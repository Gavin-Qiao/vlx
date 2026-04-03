"""Tests for vserve.probe — analytical KV cache limit calculation."""

import pytest

from vserve.probe import _context_steps, calculate_limits, kv_bytes_per_token


def test_context_steps_are_powers_of_two():
    for step in _context_steps(131072):
        assert step & (step - 1) == 0, f"{step} is not a power of two"


def test_kv_bytes_per_token_auto():
    # 2 (K+V) × 4 heads × 128 dim × 32 layers × 2 bytes = 65536
    result = kv_bytes_per_token(num_kv_heads=4, head_dim=128, num_layers=32, dtype="auto")
    assert result == 2 * 4 * 128 * 32 * 2


def test_kv_bytes_per_token_fp8():
    result = kv_bytes_per_token(num_kv_heads=4, head_dim=128, num_layers=32, dtype="fp8")
    assert result == 2 * 4 * 128 * 32 * 1


def test_fp8_is_half_of_auto():
    auto = kv_bytes_per_token(num_kv_heads=8, head_dim=128, num_layers=40, dtype="auto")
    fp8 = kv_bytes_per_token(num_kv_heads=8, head_dim=128, num_layers=40, dtype="fp8")
    assert fp8 * 2 == auto


def test_calculate_limits_basic(fake_model_dir):
    from vserve.models import detect_model
    m = detect_model(fake_model_dir)

    result = calculate_limits(model_info=m, vram_total_gb=48.0, gpu_mem_util=0.90)

    assert "limits" in result
    assert result["vram_total_gb"] == 48.0
    assert result["gpu_memory_utilization"] == 0.90
    assert result["available_kv_gb"] > 0
    for ctx_str in result["limits"]:
        assert int(ctx_str) <= m.max_position_embeddings


def test_calculate_limits_fp8_doubles_capacity(fake_model_dir):
    from vserve.models import detect_model
    m = detect_model(fake_model_dir)

    result = calculate_limits(model_info=m, vram_total_gb=48.0, gpu_mem_util=0.90)

    for ctx_str, entry in result["limits"].items():
        auto = entry.get("auto")
        fp8 = entry.get("fp8")
        if auto and fp8:
            assert fp8 >= auto * 2 - 1  # allow rounding


def test_calculate_limits_higher_context_fewer_users(fake_model_dir):
    from vserve.models import detect_model
    m = detect_model(fake_model_dir)

    result = calculate_limits(model_info=m, vram_total_gb=48.0, gpu_mem_util=0.90)

    prev_auto = None
    for ctx_str in sorted(result["limits"], key=int):
        auto = result["limits"][ctx_str].get("auto")
        if auto and prev_auto:
            assert auto <= prev_auto
        if auto:
            prev_auto = auto


def test_calculate_limits_tiny_vram(fake_model_dir):
    from vserve.models import detect_model
    m = detect_model(fake_model_dir)

    result = calculate_limits(model_info=m, vram_total_gb=0.1, gpu_mem_util=0.90)
    for entry in result["limits"].values():
        for v in entry.values():
            assert v is None


def test_calculate_limits_missing_arch_fields(fake_model_dir):
    from vserve.models import detect_model
    m = detect_model(fake_model_dir)
    m.num_kv_heads = None

    with pytest.raises(ValueError, match="missing architecture"):
        calculate_limits(model_info=m, vram_total_gb=48.0)


def test_calculate_limits_gpu_util(fake_model_dir):
    from vserve.models import detect_model
    m = detect_model(fake_model_dir)

    low = calculate_limits(model_info=m, vram_total_gb=48.0, gpu_mem_util=0.80)
    high = calculate_limits(model_info=m, vram_total_gb=48.0, gpu_mem_util=0.95)

    assert high["available_kv_gb"] > low["available_kv_gb"]


def test_calculate_limits_output_format(fake_model_dir):
    from vserve.models import detect_model
    m = detect_model(fake_model_dir)

    result = calculate_limits(model_info=m, vram_total_gb=48.0)

    for key in ["model_path", "calculated_at", "vram_total_gb", "gpu_memory_utilization",
                "model_size_gb", "quant_method", "architecture", "is_moe",
                "max_position_embeddings", "tool_call_parser", "reasoning_parser",
                "limits"]:
        assert key in result

    for entry in result["limits"].values():
        assert "auto" in entry
        assert "fp8" in entry


def test_sanity_qwen3_27b_on_rtx_pro_5000():
    """Verify math against a known real-world model on known hardware.

    Qwen3-27B-FP8: 28 layers, 4 kv_heads, 128 head_dim, 28.7 GB weights
    RTX PRO 5000: 48 GB VRAM
    """
    from vserve.models import ModelInfo
    from pathlib import Path

    m = ModelInfo(
        path=Path("/fake"),
        provider="Qwen",
        model_name="Qwen3.5-27B-FP8",
        architecture="Qwen3ForCausalLM",
        model_type="qwen3",
        quant_method="fp8",
        max_position_embeddings=131072,
        is_moe=False,
        model_size_gb=28.7,
        num_kv_heads=4,
        num_layers=28,
        head_dim=128,
    )
    result = calculate_limits(model_info=m, vram_total_gb=47.8, gpu_mem_util=0.90)

    # KV per token (auto): 2 * 4 * 128 * 28 * 2 = 57344 bytes = ~56 KB
    bpt_auto = kv_bytes_per_token(num_kv_heads=4, head_dim=128, num_layers=28, dtype="auto")
    assert bpt_auto == 57344

    # Available KV: 47.8 * 0.90 - 28.7 = 14.32 GB
    # (overhead is in gpu_mem_util, not subtracted again here)
    assert 13 < result["available_kv_gb"] < 16

    # At 8K context, auto: ~14.3 GB / (57344 * 8192) = ~32 users
    ctx_8k = result["limits"]["8192"]
    assert ctx_8k["auto"] is not None
    assert 20 < ctx_8k["auto"] < 40  # reasonable range

    # FP8 should roughly double it
    assert ctx_8k["fp8"] is not None
    assert ctx_8k["fp8"] >= ctx_8k["auto"] * 2 - 1

    # 131072 context should have very few or no users
    ctx_128k = result["limits"].get("131072")
    if ctx_128k:
        auto_128k = ctx_128k.get("auto")
        # At 128K with auto: 14.3 GB / (57344 * 131072) = ~2 users
        assert auto_128k is None or auto_128k <= 2


def test_calculate_limits_model_exceeds_vram():
    """Model weights alone exceed available VRAM — all limits should be None."""
    from vserve.models import ModelInfo
    from pathlib import Path

    m = ModelInfo(
        path=Path("/fake"), provider="big", model_name="Big-70B",
        architecture="LlamaForCausalLM", model_type="llama",
        quant_method=None, max_position_embeddings=32768,
        is_moe=False, model_size_gb=70.0,
        num_kv_heads=8, num_layers=80, head_dim=128,
    )
    result = calculate_limits(model_info=m, vram_total_gb=24.0, gpu_mem_util=0.90)
    assert result["available_kv_gb"] == 0
    for entry in result["limits"].values():
        for v in entry.values():
            assert v is None


def test_calculate_limits_max_pos_below_smallest_step():
    """Model with 2K context — no CONTEXT_STEPS apply, limits should be empty."""
    from vserve.models import ModelInfo
    from pathlib import Path

    m = ModelInfo(
        path=Path("/fake"), provider="tiny", model_name="Tiny-1B",
        architecture="GPT2LMHeadModel", model_type="gpt2",
        quant_method=None, max_position_embeddings=2048,
        is_moe=False, model_size_gb=2.0,
        num_kv_heads=8, num_layers=24, head_dim=64,
    )
    result = calculate_limits(model_info=m, vram_total_gb=48.0)
    assert result["limits"] == {}


def test_calculate_limits_moe_model(fake_moe_model_dir):
    """MoE models should calculate limits normally when arch fields are present."""
    from vserve.models import detect_model
    m = detect_model(fake_moe_model_dir)

    result = calculate_limits(model_info=m, vram_total_gb=48.0, gpu_mem_util=0.90)
    assert result["is_moe"] is True
    assert len(result["limits"]) > 0
    # Should have entries — MoE fixture now has arch fields
    some_entry = list(result["limits"].values())[0]
    assert "auto" in some_entry
    assert "fp8" in some_entry
