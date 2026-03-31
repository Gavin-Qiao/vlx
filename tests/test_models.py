import pytest

from vserve.models import (
    detect_model,
    scan_models,
    fuzzy_match,
    quant_flag,
)


def test_detect_model_basic(fake_model_dir):
    info = detect_model(fake_model_dir)
    assert info.architecture == "TestForCausalLM"
    assert info.quant_method == "fp8"
    assert info.max_position_embeddings == 131072
    assert info.is_moe is False
    assert info.provider == "testprovider"
    assert info.model_name == "TestModel-7B-FP8"


def test_detect_model_arch_fields(fake_model_dir):
    info = detect_model(fake_model_dir)
    assert info.num_kv_heads == 4
    assert info.num_layers == 32
    assert info.head_dim == 128  # 4096 / 32 attention heads


def test_detect_model_head_dim_from_hidden_size(tmp_path):
    """head_dim is derived from hidden_size / num_attention_heads when not explicit."""
    import json
    model_dir = tmp_path / "models" / "test" / "HeadDimTest"
    model_dir.mkdir(parents=True)
    config = {
        "architectures": ["TestLM"],
        "model_type": "test",
        "text_config": {
            "hidden_size": 8192,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "num_hidden_layers": 48,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "model.safetensors").write_bytes(b"\0" * 512)
    info = detect_model(model_dir)
    assert info.head_dim == 128  # 8192 / 64


def test_detect_model_missing_arch_fields(tmp_path):
    """Models without kv_heads/layers get None fields."""
    import json
    model_dir = tmp_path / "models" / "test" / "MinimalModel"
    model_dir.mkdir(parents=True)
    config = {
        "architectures": ["MinimalLM"],
        "model_type": "test",
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "model.safetensors").write_bytes(b"\0" * 512)
    info = detect_model(model_dir)
    assert info.num_kv_heads is None
    assert info.num_layers is None
    assert info.head_dim is None


def test_detect_model_explicit_head_dim(tmp_path):
    """When head_dim is explicit in config, use it instead of deriving."""
    import json
    model_dir = tmp_path / "models" / "test" / "ExplicitHeadDim"
    model_dir.mkdir(parents=True)
    config = {
        "architectures": ["TestLM"],
        "model_type": "test",
        "text_config": {
            "head_dim": 64,
            "hidden_size": 8192,
            "num_attention_heads": 64,  # derived would be 128, but explicit wins
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "model.safetensors").write_bytes(b"\0" * 512)
    info = detect_model(model_dir)
    assert info.head_dim == 64  # explicit, not 8192/64=128


def test_detect_model_top_level_arch_fields(tmp_path):
    """Arch fields at top level (no text_config) — common in many HF models."""
    import json
    model_dir = tmp_path / "models" / "test" / "TopLevel"
    model_dir.mkdir(parents=True)
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "num_key_value_heads": 8,
        "num_attention_heads": 32,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "max_position_embeddings": 4096,
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "model.safetensors").write_bytes(b"\0" * 512)
    info = detect_model(model_dir)
    assert info.num_kv_heads == 8
    assert info.num_layers == 32
    assert info.head_dim == 128  # 4096 / 32


def test_detect_model_num_local_experts(tmp_path):
    """Mixtral-style models use num_local_experts instead of num_experts."""
    import json
    model_dir = tmp_path / "models" / "test" / "MixtralStyle"
    model_dir.mkdir(parents=True)
    config = {
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "num_local_experts": 8,
        "max_position_embeddings": 32768,
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "model.safetensors").write_bytes(b"\0" * 512)
    info = detect_model(model_dir)
    assert info.is_moe is True


def test_detect_model_moe(fake_moe_model_dir):
    info = detect_model(fake_moe_model_dir)
    assert info.is_moe is True
    assert info.max_position_embeddings == 262144


def test_detect_model_missing_config(tmp_path):
    empty = tmp_path / "models" / "test" / "empty"
    empty.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        detect_model(empty)


def test_quant_flag_mapping():
    assert quant_flag("gptq") == "--quantization gptq_marlin"
    assert quant_flag("awq") == "--quantization awq_marlin"
    assert quant_flag("fp8") == "--quantization fp8"
    assert quant_flag("compressed-tensors") == ""
    assert quant_flag("none") == ""
    assert quant_flag(None) == ""


def test_scan_models(models_root):
    models = scan_models(models_root)
    assert len(models) == 1
    assert models[0].model_name == "TestModel-7B-FP8"


def test_scan_models_empty(tmp_path):
    empty_root = tmp_path / "models"
    empty_root.mkdir()
    assert scan_models(empty_root) == []


@pytest.mark.parametrize(
    "query,expected_count",
    [
        ("TestModel-7B-FP8", 1),
        ("testmodel", 1),
        ("7b-fp8", 1),
        ("fp8", 1),
        ("nonexistent", 0),
    ],
)
def test_fuzzy_match(models_root, query, expected_count):
    models = scan_models(models_root)
    matches = fuzzy_match(query, models)
    assert len(matches) == expected_count


def test_fuzzy_match_exact_provider_model(models_root):
    models = scan_models(models_root)
    matches = fuzzy_match("testprovider/TestModel-7B-FP8", models)
    assert len(matches) == 1


def test_fuzzy_match_full_path(models_root):
    models = scan_models(models_root)
    full_path = str(models[0].path)
    matches = fuzzy_match(full_path, models)
    assert len(matches) == 1
