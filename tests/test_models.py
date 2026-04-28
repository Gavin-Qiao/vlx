import logging

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


def test_detect_model_size_includes_bin_weights(tmp_path):
    """Legacy PyTorch .bin weights count toward model size."""
    import json
    model_dir = tmp_path / "models" / "test" / "BinModel"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["TestLM"],
        "model_type": "test",
    }))
    with (model_dir / "pytorch_model.bin").open("wb") as f:
        f.truncate(2 * 1024**3)

    info = detect_model(model_dir)

    assert info.model_size_gb == 2.0


def test_detect_model_mha_uses_attention_heads_when_kv_heads_missing(tmp_path):
    """Standard MHA configs omit num_key_value_heads; KV heads equal attention heads."""
    import json
    model_dir = tmp_path / "models" / "test" / "MhaModel"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["TestLM"],
        "model_type": "test",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_hidden_layers": 24,
    }))
    (model_dir / "model.safetensors").write_bytes(b"\0" * 512)

    info = detect_model(model_dir)

    assert info.num_kv_heads == 32
    assert info.head_dim == 128


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


def test_scan_models_skips_ignored_materialization_sources(tmp_path):
    source = tmp_path / "models" / "provider" / "Model"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    (source / ".vserve-ignore").write_text("materialized variants live in sibling roots\n")

    assert scan_models(tmp_path / "models") == []


def test_scan_models_logs_invalid_candidate(tmp_path, caplog):
    bad_dir = tmp_path / "models" / "broken" / "BadModel"
    bad_dir.mkdir(parents=True)
    (bad_dir / "config.json").write_text("{broken")

    with caplog.at_level(logging.WARNING):
        assert scan_models(tmp_path / "models") == []

    assert "Skipping model directory" in caplog.text
    assert str(bad_dir) in caplog.text


def test_scan_models_skips_unreadable_provider(tmp_path, caplog, monkeypatch):
    from pathlib import Path

    root = tmp_path / "models"
    provider = root / "secret"
    provider.mkdir(parents=True)

    real_iterdir = Path.iterdir

    def fake_iterdir(self):
        if self == provider:
            raise PermissionError("denied")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)

    with caplog.at_level(logging.WARNING):
        assert scan_models(root) == []

    assert "Skipping provider directory" in caplog.text
    assert str(provider) in caplog.text


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


def test_fuzzy_match_tokenized_query_ranks_quant_and_provider():
    from pathlib import Path

    from vserve.models import ModelInfo

    models = [
        ModelInfo(
            path=Path("/models/Qwen/Qwen3.5-27B-FP8"),
            provider="Qwen",
            model_name="Qwen3.5-27B-FP8",
            architecture="Qwen3ForCausalLM",
            model_type="qwen3",
            quant_method="fp8",
            max_position_embeddings=131072,
            is_moe=False,
            model_size_gb=27,
        ),
        ModelInfo(
            path=Path("/models/Qwen/Qwen3.5-27B-BF16"),
            provider="Qwen",
            model_name="Qwen3.5-27B-BF16",
            architecture="Qwen3ForCausalLM",
            model_type="qwen3",
            quant_method=None,
            max_position_embeddings=131072,
            is_moe=False,
            model_size_gb=54,
        ),
    ]

    matches = fuzzy_match("qwen fp8", models)

    assert [m.model_name for m in matches] == ["Qwen3.5-27B-FP8"]


# --- GGUF model detection ---


def test_detect_gguf_model(fake_gguf_model_dir):
    m = detect_model(fake_gguf_model_dir)
    assert m.provider == "bartowski"
    assert m.model_name == "TestModel-8B-GGUF"
    assert m.model_size_gb >= 0  # fake file is tiny
    assert m.is_gguf is True
    assert m.architecture == "gguf"


def test_scan_models_finds_gguf(tmp_path, fake_gguf_model_dir):
    models = scan_models(tmp_path / "models")
    gguf_models = [m for m in models if m.is_gguf]
    assert len(gguf_models) == 1


def test_safetensors_model_not_gguf(fake_model_dir):
    m = detect_model(fake_model_dir)
    assert m.is_gguf is False


def test_gguf_with_config_json(tmp_path):
    """GGUF model with config.json alongside (common in HF repos) is detected as GGUF."""
    import json
    model_dir = tmp_path / "models" / "user" / "Model-GGUF"
    model_dir.mkdir(parents=True)
    (model_dir / "model-Q4_K_M.gguf").write_bytes(b"\0" * 8192)
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
    }))
    m = detect_model(model_dir)
    assert m.is_gguf is True
    assert m.model_size_gb >= 0


def test_mixed_safetensors_and_gguf_prefers_safetensors(tmp_path):
    """Dir with both safetensors and GGUF — detected as safetensors (vLLM)."""
    import json
    model_dir = tmp_path / "models" / "user" / "Mixed"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"\0" * 4096)
    (model_dir / "model-Q4_K_M.gguf").write_bytes(b"\0" * 8192)
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "max_position_embeddings": 4096,
    }))
    m = detect_model(model_dir)
    assert m.is_gguf is False


def test_is_embedding_by_name():
    from vserve.models import ModelInfo
    from pathlib import Path

    for name, expected in [
        ("nomic-embed-text-v1.5-GGUF", True),
        ("Qwen3-Embedding-8B", True),
        ("bge-base-en-v1.5-gguf", True),
        ("e5-small-v2-gguf", True),
        ("jina-reranker-v2-GGUF", True),
        ("Qwen3.5-27B-FP8", False),
        ("Llama-3-8B", False),
    ]:
        m = ModelInfo(
            path=Path(f"/fake/{name}"), provider="test", model_name=name,
            architecture="gguf", model_type="gguf", quant_method=None,
            max_position_embeddings=0, is_moe=False, model_size_gb=1.0,
        )
        assert m.is_embedding is expected, f"{name}: expected {expected}"


def test_is_embedding_by_architecture():
    """is_embedding detects from architecture field."""
    from vserve.models import ModelInfo
    from pathlib import Path

    m = ModelInfo(
        path=Path("/fake/SomeModel"), provider="test", model_name="SomeModel",
        architecture="BertForEmbedding", model_type="bert", quant_method=None,
        max_position_embeddings=512, is_moe=False, model_size_gb=0.5,
    )
    assert m.is_embedding is True

    m2 = ModelInfo(
        path=Path("/fake/LLM"), provider="test", model_name="LLM",
        architecture="LlamaForCausalLM", model_type="llama", quant_method=None,
        max_position_embeddings=32768, is_moe=False, model_size_gb=7.0,
    )
    assert m2.is_embedding is False


def test_is_embedding_edge_cases():
    """is_embedding handles hyphenated names correctly."""
    from vserve.models import ModelInfo
    from pathlib import Path

    for name, expected in [
        ("bge-m3-GGUF", True),
        ("my-bge-model", True),
        ("e5-large-v2", True),
        ("Qwen3-Embedding-8B", True),
        ("e5x-model", False),  # "e5" must be a whole segment
        ("biggest-model", False),  # contains "e5" but not as a segment
    ]:
        m = ModelInfo(
            path=Path(f"/fake/{name}"), provider="test", model_name=name,
            architecture="gguf", model_type="gguf", quant_method=None,
            max_position_embeddings=0, is_moe=False, model_size_gb=1.0,
        )
        assert m.is_embedding is expected, f"{name}: expected {expected}"


def test_detect_gguf_model_no_config(tmp_path):
    """GGUF model without config.json still detected."""
    model_dir = tmp_path / "models" / "user" / "Model-GGUF"
    model_dir.mkdir(parents=True)
    (model_dir / "model-Q4_K_M.gguf").write_bytes(b"\0" * 8192)
    # No config.json, no tokenizer_config.json
    m = detect_model(model_dir)
    assert m.is_gguf is True
    assert m.architecture == "gguf"
    assert m.model_size_gb >= 0


def test_guess_pooling():
    from vserve.backends.llamacpp import LlamaCppBackend
    assert LlamaCppBackend._guess_pooling("bge-base-en-v1.5-gguf") == "cls"
    assert LlamaCppBackend._guess_pooling("nomic-embed-text-v1.5") == "mean"
    assert LlamaCppBackend._guess_pooling("jina-reranker-v2") == "rank"
    assert LlamaCppBackend._guess_pooling("e5-small-v2") == "mean"
