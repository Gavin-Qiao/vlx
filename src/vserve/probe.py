"""Calculate context and concurrency limits for a model analytically.

Uses the KV cache formula:
  kv_per_token = 2 × num_kv_heads × head_dim × num_layers × dtype_bytes
  available_kv = total_vram × gpu_mem_util − model_weights − overhead
  max_concurrent = available_kv_tokens ÷ context_length
"""

from __future__ import annotations

from datetime import datetime, timezone

from vserve.models import ModelInfo
from vserve.tools import detect_tool_parser, detect_reasoning_parser

_MIN_CTX = 4096


def _context_steps(max_pos: int) -> list[int]:
    """Generate context steps as powers of 2 from 4k up to the model's max."""
    ctx = _MIN_CTX
    steps = []
    while ctx <= max_pos:
        steps.append(ctx)
        ctx *= 2
    return steps

# Bytes per element by KV cache dtype
_DTYPE_BYTES = {"auto": 2, "fp8": 1}


def kv_bytes_per_token(
    *, num_kv_heads: int, head_dim: int, num_layers: int, dtype: str,
) -> int:
    """KV cache memory per token in bytes: 2 (K+V) × heads × dim × layers × dtype."""
    return 2 * num_kv_heads * head_dim * num_layers * _DTYPE_BYTES[dtype]


def calculate_limits(
    *,
    model_info: ModelInfo,
    vram_total_gb: float,
    gpu_mem_util: float = 0.90,
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

    available_kv_bytes = available_kv_gb * (1024**3)

    max_pos = model_info.max_position_embeddings
    steps = _context_steps(max_pos)

    limits: dict[str, dict[str, int | None]] = {}

    for ctx in steps:
        key = str(ctx)
        limits[key] = {}
        for dtype in ("auto", "fp8"):
            bpt = kv_bytes_per_token(
                num_kv_heads=model_info.num_kv_heads,  # type: ignore[arg-type]
                head_dim=model_info.head_dim,  # type: ignore[arg-type]
                num_layers=model_info.num_layers,  # type: ignore[arg-type]
                dtype=dtype,
            )
            total_tokens = int(available_kv_bytes // bpt)
            max_users = total_tokens // ctx
            limits[key][dtype] = max_users if max_users >= 1 else None

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
        "limits": limits,
    }
