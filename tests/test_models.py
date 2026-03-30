import pytest

from vlx.models import (
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
