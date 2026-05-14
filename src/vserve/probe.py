"""Calculate vLLM context, cache, and scheduler limits analytically.

The output keeps the historical ``limits`` table for compatibility, but the
math now follows vLLM V1 more closely:
  * KV cache capacity is rounded to PagedAttention cache blocks.
  * KV cache dtypes include FP8 and TurboQuant packed formats.
  * Scheduler profiles expose chunked-prefill-oriented defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil

from vserve.models import ModelInfo
from vserve.tools import detect_tool_parser, detect_reasoning_parser

_MIN_CTX = 4096
_DEFAULT_BLOCK_SIZE = 16


@dataclass(frozen=True)
class TurboQuantProfile:
    key_bits: int
    value_bits: int
    key_fp8: bool = False
    key_norm_bytes: int = 2
    value_scale_zero_bytes: int = 4
    quality_note: str = ""


_STANDARD_DTYPE_BYTES = {
    "auto": 2,
    "bfloat16": 2,
    "float16": 2,
    "fp16": 2,
    "fp8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "fp8_inc": 1,
}

_TURBOQUANT_PROFILES = {
    "turboquant_k8v4": TurboQuantProfile(
        key_bits=8,
        value_bits=4,
        key_fp8=True,
        key_norm_bytes=0,
        quality_note="FP8 keys + 4-bit values; safest TurboQuant capacity mode.",
    ),
    "turboquant_4bit_nc": TurboQuantProfile(
        key_bits=4,
        value_bits=4,
        quality_note="4-bit keys + 4-bit values; higher quality risk than k8v4.",
    ),
    "turboquant_k3v4_nc": TurboQuantProfile(
        key_bits=3,
        value_bits=4,
        quality_note="3-bit keys + 4-bit values; aggressive capacity mode.",
    ),
    "turboquant_3bit_nc": TurboQuantProfile(
        key_bits=3,
        value_bits=3,
        quality_note="3-bit keys + 3-bit values; highest quality risk.",
    ),
}

_DEFAULT_KV_DTYPES = (
    "auto",
    "fp8",
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
)


def _context_steps(max_pos: int) -> list[int]:
    """Generate context steps as powers of 2 from 4k up to the model's max."""
    ctx = _MIN_CTX
    steps = []
    while ctx <= max_pos:
        steps.append(ctx)
        ctx *= 2
    return steps

def _packed_bits_bytes(elements: int, bits: int) -> int:
    return ceil(elements * bits / 8)


def _align_even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def _turboquant_bytes_per_head(*, head_dim: int, dtype: str) -> int:
    profile = _TURBOQUANT_PROFILES[dtype]
    if profile.key_fp8:
        key_bytes = head_dim
    else:
        key_bytes = _packed_bits_bytes(head_dim, profile.key_bits) + profile.key_norm_bytes
    value_bytes = _packed_bits_bytes(head_dim, profile.value_bits) + profile.value_scale_zero_bytes
    return _align_even(key_bytes + value_bytes)


def kv_bytes_per_token(
    *, num_kv_heads: int, head_dim: int, num_layers: int, dtype: str,
) -> int:
    """KV cache memory per token in bytes for one resident token."""
    if dtype in _STANDARD_DTYPE_BYTES:
        return 2 * num_kv_heads * head_dim * num_layers * _STANDARD_DTYPE_BYTES[dtype]
    if dtype in _TURBOQUANT_PROFILES:
        return num_kv_heads * num_layers * _turboquant_bytes_per_head(
            head_dim=head_dim,
            dtype=dtype,
        )
    supported = ", ".join(sorted([*_STANDARD_DTYPE_BYTES, *_TURBOQUANT_PROFILES]))
    raise ValueError(f"Unsupported vLLM KV cache dtype '{dtype}'. Supported: {supported}")


def max_concurrent_sequences(
    *,
    available_kv_bytes: int | float,
    bytes_per_token: int,
    context_length: int,
    block_size: int = _DEFAULT_BLOCK_SIZE,
) -> int:
    """Return the max resident sequences after rounding each request to KV blocks."""
    if available_kv_bytes <= 0 or bytes_per_token <= 0 or context_length <= 0 or block_size <= 0:
        return 0
    total_tokens = int(available_kv_bytes // bytes_per_token)
    resident_tokens_per_sequence = ceil(context_length / block_size) * block_size
    return total_tokens // resident_tokens_per_sequence


def _kv_cache_dtype_profiles(*, num_kv_heads: int, head_dim: int, num_layers: int) -> dict[str, dict[str, object]]:
    auto_bpt = kv_bytes_per_token(
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        dtype="auto",
    )
    profiles: dict[str, dict[str, object]] = {}
    for dtype in _DEFAULT_KV_DTYPES:
        bpt = kv_bytes_per_token(
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            dtype=dtype,
        )
        entry: dict[str, object] = {
            "bytes_per_token": bpt,
            "compression_ratio_vs_auto": round(auto_bpt / bpt, 2) if bpt else None,
        }
        if dtype == "auto":
            entry["quality"] = "native"
        elif dtype == "fp8":
            entry["quality"] = "lossy; usually low risk with calibrated scales"
        elif dtype in _TURBOQUANT_PROFILES:
            entry["quality"] = _TURBOQUANT_PROFILES[dtype].quality_note
        profiles[dtype] = entry
    return profiles


def _limits_entry(limits: dict[str, dict[str, int | None]], context: int, dtype: str) -> int | None:
    return limits.get(str(context), {}).get(dtype)


def _largest_working_context(
    limits: dict[str, dict[str, int | None]],
    *,
    dtype: str,
    cap: int,
) -> int | None:
    working = sorted(
        int(ctx_str)
        for ctx_str, entry in limits.items()
        if entry.get(dtype) is not None
    )
    if not working:
        return None
    capped = [ctx for ctx in working if ctx <= cap]
    return max(capped) if capped else min(working)


def _recommended_dtype(
    limits: dict[str, dict[str, int | None]],
    *,
    context: int,
    profile: str,
) -> str | None:
    entry = limits.get(str(context), {})
    if profile == "throughput":
        fp8_slots = entry.get("fp8")
        auto_slots = entry.get("auto")
        if fp8_slots is not None and (auto_slots is None or fp8_slots >= max(auto_slots + 1, int(auto_slots * 1.5))):
            return "fp8"
    if entry.get("auto") is not None:
        return "auto"
    if entry.get("fp8") is not None:
        return "fp8"
    return next((dtype for dtype, slots in entry.items() if slots is not None), None)


def _scheduler_tokens(*, profile: str, context: int) -> int:
    if profile == "interactivity":
        return min(max(2048, min(context, 4096)), context)
    if profile == "throughput":
        return max(12288, min(32768, max(context, 12288)))
    return max(4096, min(8192, context))


def _scheduler_recommendations(
    limits: dict[str, dict[str, int | None]],
    *,
    block_size: int,
) -> dict[str, dict[str, int | str]]:
    profile_caps = {
        "interactivity": 8192,
        "balanced": 32768,
        "throughput": 8192,
    }
    recommendations: dict[str, dict[str, int | str]] = {}
    for profile, cap in profile_caps.items():
        context = _largest_working_context(limits, dtype="auto", cap=cap)
        if context is None:
            context = _largest_working_context(limits, dtype="fp8", cap=cap)
        if context is None:
            continue
        dtype = _recommended_dtype(limits, context=context, profile=profile)
        if dtype is None:
            continue
        slots = _limits_entry(limits, context, dtype)
        if slots is None:
            continue
        recommendations[profile] = {
            "context": context,
            "kv_cache_dtype": dtype,
            "max_num_seqs": slots,
            "max_num_batched_tokens": _scheduler_tokens(profile=profile, context=context),
            "performance_mode": profile,
            "optimization_level": 2,
            "block_size": block_size,
        }
    return recommendations


def calculate_limits(
    *,
    model_info: ModelInfo,
    vram_total_gb: float,
    gpu_mem_util: float = 0.90,
    block_size: int = _DEFAULT_BLOCK_SIZE,
) -> dict:
    """Calculate the full context × concurrency × kv_dtype limit table.

    Returns the same dict structure as the old probe_model() for backward compat.
    """
    if model_info.num_kv_heads is None or model_info.head_dim is None or model_info.num_layers is None:
        raise ValueError(
            f"Model {model_info.full_name} missing architecture fields "
            f"(num_kv_heads={model_info.num_kv_heads}, head_dim={model_info.head_dim}, "
            f"num_layers={model_info.num_layers}). Cannot calculate KV cache limits."
        )

    usable_gb = vram_total_gb * gpu_mem_util
    available_kv_gb = usable_gb - model_info.model_size_gb
    if available_kv_gb <= 0:
        available_kv_gb = 0

    available_kv_bytes = int(available_kv_gb * (1024**3))

    max_pos = model_info.max_position_embeddings
    steps = _context_steps(max_pos)

    limits: dict[str, dict[str, int | None]] = {}
    kv_dtype_profiles = _kv_cache_dtype_profiles(
        num_kv_heads=model_info.num_kv_heads,  # type: ignore[arg-type]
        head_dim=model_info.head_dim,  # type: ignore[arg-type]
        num_layers=model_info.num_layers,  # type: ignore[arg-type]
    )

    for ctx in steps:
        key = str(ctx)
        limits[key] = {}
        for dtype in _DEFAULT_KV_DTYPES:
            bpt = kv_bytes_per_token(
                num_kv_heads=model_info.num_kv_heads,  # type: ignore[arg-type]
                head_dim=model_info.head_dim,  # type: ignore[arg-type]
                num_layers=model_info.num_layers,  # type: ignore[arg-type]
                dtype=dtype,
            )
            max_users = max_concurrent_sequences(
                available_kv_bytes=available_kv_bytes,
                bytes_per_token=bpt,
                context_length=ctx,
                block_size=block_size,
            )
            limits[key][dtype] = max_users if max_users >= 1 else None

    recommendations = _scheduler_recommendations(limits, block_size=block_size)

    # Detect tool calling and reasoning capabilities
    tool_parser = detect_tool_parser(model_info.path)
    reasoning_parser = detect_reasoning_parser(model_info.path)

    return {
        "model_path": str(model_info.path),
        "calculated_at": datetime.now(timezone.utc).isoformat(),
        "vram_total_gb": vram_total_gb,
        "gpu_memory_utilization": gpu_mem_util,
        "model_size_gb": model_info.model_size_gb,
        "available_kv_gb": round(available_kv_gb, 1),
        "quant_method": model_info.quant_method,
        "architecture": model_info.architecture,
        "is_moe": model_info.is_moe,
        "max_position_embeddings": model_info.max_position_embeddings,
        "tool_call_parser": tool_parser,
        "reasoning_parser": reasoning_parser,
        "tuning_model": {
            "backend": "vllm-v1",
            "capacity_math": "paged-block-rounded",
            "block_size": block_size,
            "scheduler": "chunked-prefill",
            "prefix_cache_assumption": "enabled by generated vserve profiles",
        },
        "kv_cache_dtypes": kv_dtype_profiles,
        "recommendations": recommendations,
        "limits": limits,
    }
