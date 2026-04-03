import json
from pathlib import Path

import pytest


@pytest.fixture
def fake_model_dir(tmp_path: Path) -> Path:
    """Create a fake model directory with config.json."""
    model_dir = tmp_path / "models" / "testprovider" / "TestModel-7B-FP8"
    model_dir.mkdir(parents=True)

    config = {
        "architectures": ["TestForCausalLM"],
        "model_type": "test",
        "quantization_config": {"quant_method": "fp8"},
        "text_config": {
            "max_position_embeddings": 131072,
            "num_key_value_heads": 4,
            "num_attention_heads": 32,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "model.safetensors").write_bytes(b"\0" * 1024)

    return model_dir


@pytest.fixture
def fake_moe_model_dir(tmp_path: Path) -> Path:
    """Create a fake MoE model directory."""
    model_dir = tmp_path / "models" / "testprovider" / "TestMoE-35B-FP8"
    model_dir.mkdir(parents=True)

    config = {
        "architectures": ["TestMoeForCausalLM"],
        "model_type": "test_moe",
        "quantization_config": {"quant_method": "fp8"},
        "text_config": {
            "max_position_embeddings": 262144,
            "num_experts": 256,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
            "hidden_size": 4096,
            "num_hidden_layers": 64,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "model.safetensors").write_bytes(b"\0" * 2048)

    return model_dir


@pytest.fixture
def fake_limits() -> dict:
    """Sample limits.json data."""
    return {
        "model_path": "/opt/vllm/models/testprovider/TestModel-7B-FP8",
        "calculated_at": "2026-03-27T10:00:00",
        "vram_total_gb": 48.0,
        "gpu_memory_utilization": 0.9270833333333334,
        "available_kv_gb": 37.3,  # 48.0 * 0.927 - 7.2
        "model_size_gb": 7.2,
        "quant_method": "fp8",
        "architecture": "TestForCausalLM",
        "is_moe": False,
        "max_position_embeddings": 131072,
        "limits": {
            "4096": {"auto": 64, "fp8": 128},
            "8192": {"auto": 32, "fp8": 64},
            "16384": {"auto": 16, "fp8": 32},
            "32768": {"auto": 8, "fp8": 16},
            "65536": {"auto": None, "fp8": 8},
            "131072": {"auto": None, "fp8": None},
        },
    }


@pytest.fixture
def models_root(tmp_path: Path, fake_model_dir: Path) -> Path:
    """Return the models root containing fake_model_dir."""
    return tmp_path / "models"
