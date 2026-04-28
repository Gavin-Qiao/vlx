from vserve.variants import discover_variants, Variant


def test_single_index_standard_repo():
    """Standard repo: one index.json + shards + shared files (e.g., Qwen/Qwen3-8B)."""
    files = {
        "config.json": 1200,
        "tokenizer.json": 4_000_000,
        "generation_config.json": 200,
        "model.safetensors.index.json": 50_000,
        "model-00001-of-00003.safetensors": 5_000_000_000,
        "model-00002-of-00003.safetensors": 5_000_000_000,
        "model-00003-of-00003.safetensors": 4_000_000_000,
    }
    index_contents = {
        "model.safetensors.index.json": {
            "metadata": {"total_size": 14_000_000_000},
            "weight_map": {
                "model.layers.0.weight": "model-00001-of-00003.safetensors",
                "model.layers.1.weight": "model-00002-of-00003.safetensors",
                "model.layers.2.weight": "model-00003-of-00003.safetensors",
            },
        }
    }

    variants, shared = discover_variants(files, index_contents)

    assert len(variants) == 1
    v = variants[0]
    assert v.label == "default"
    assert v.num_files == 4  # 3 shards + index
    assert v.total_bytes == 5_000_000_000 + 5_000_000_000 + 4_000_000_000 + 50_000
    assert "model-00001-of-00003.safetensors" in v.files
    assert "model.safetensors.index.json" in v.files

    # Shared files
    assert "config.json" in shared
    assert "tokenizer.json" in shared
    assert "generation_config.json" in shared


def test_multi_variant_repo():
    """Repo with default + subdirectory variant (e.g., openai/gpt-oss-20b)."""
    files = {
        "config.json": 1200,
        "tokenizer.json": 4_000_000,
        "model.safetensors.index.json": 50_000,
        "model-00001-of-00002.safetensors": 7_000_000_000,
        "model-00002-of-00002.safetensors": 7_000_000_000,
        "original/model.safetensors.index.json": 40_000,
        "original/model--00001-of-00001.safetensors": 38_000_000_000,
    }
    index_contents = {
        "model.safetensors.index.json": {
            "metadata": {"total_size": 14_000_000_000},
            "weight_map": {
                "layer.0": "model-00001-of-00002.safetensors",
                "layer.1": "model-00002-of-00002.safetensors",
            },
        },
        "original/model.safetensors.index.json": {
            "metadata": {"total_size": 38_000_000_000},
            "weight_map": {
                "layer.0": "model--00001-of-00001.safetensors",
            },
        },
    }

    variants, shared = discover_variants(files, index_contents)

    assert len(variants) == 2
    labels = {v.label for v in variants}
    assert "default" in labels
    assert "original" in labels

    default = next(v for v in variants if v.label == "default")
    assert default.num_files == 3
    original = next(v for v in variants if v.label == "original")
    assert original.num_files == 2
    assert "original/model--00001-of-00001.safetensors" in original.files
    assert "config.json" in shared


def test_variant_tag_in_index_filename():
    """Repo using Transformers variant naming (model.safetensors.index.fp16.json)."""
    files = {
        "config.json": 1000,
        "model.safetensors.index.json": 30_000,
        "model-00001-of-00001.safetensors": 8_000_000_000,
        "model.safetensors.index.fp16.json": 30_000,
        "model.fp16-00001-of-00001.safetensors": 4_000_000_000,
    }
    index_contents = {
        "model.safetensors.index.json": {
            "metadata": {},
            "weight_map": {"w": "model-00001-of-00001.safetensors"},
        },
        "model.safetensors.index.fp16.json": {
            "metadata": {},
            "weight_map": {"w": "model.fp16-00001-of-00001.safetensors"},
        },
    }

    variants, shared = discover_variants(files, index_contents)

    assert len(variants) == 2
    labels = {v.label for v in variants}
    assert "default" in labels
    assert "fp16" in labels


def test_pytorch_bin_index():
    """Legacy repo with pytorch_model.bin shards."""
    files = {
        "config.json": 1000,
        "pytorch_model.bin.index.json": 20_000,
        "pytorch_model-00001-of-00002.bin": 5_000_000_000,
        "pytorch_model-00002-of-00002.bin": 5_000_000_000,
    }
    index_contents = {
        "pytorch_model.bin.index.json": {
            "metadata": {},
            "weight_map": {
                "layer.0": "pytorch_model-00001-of-00002.bin",
                "layer.1": "pytorch_model-00002-of-00002.bin",
            },
        },
    }

    variants, shared = discover_variants(files, index_contents)

    assert len(variants) == 1
    assert variants[0].label == "default"
    assert variants[0].num_files == 3


def test_orphan_safetensors():
    """Mistral-style: sharded + consolidated.safetensors with no index."""
    files = {
        "config.json": 1000,
        "model.safetensors.index.json": 30_000,
        "model-00001-of-00002.safetensors": 5_000_000_000,
        "model-00002-of-00002.safetensors": 5_000_000_000,
        "consolidated.safetensors": 10_000_000_000,
    }
    index_contents = {
        "model.safetensors.index.json": {
            "metadata": {},
            "weight_map": {
                "layer.0": "model-00001-of-00002.safetensors",
                "layer.1": "model-00002-of-00002.safetensors",
            },
        },
    }

    variants, shared = discover_variants(files, index_contents)

    assert len(variants) == 2
    labels = {v.label for v in variants}
    assert "default" in labels
    assert "consolidated.safetensors" in labels

    orphan = next(v for v in variants if v.label == "consolidated.safetensors")
    assert orphan.num_files == 1
    assert orphan.total_bytes == 10_000_000_000


def test_gguf_variants():
    """GGUF-only repo with multiple quant levels."""
    files = {
        "config.json": 1000,
        "Qwen3-8B-Q4_K_M.gguf": 4_000_000_000,
        "Qwen3-8B-Q5_K_S.gguf": 5_000_000_000,
        "Qwen3-8B-Q8_0.gguf": 8_000_000_000,
    }

    variants, shared = discover_variants(files, {})

    assert len(variants) == 3
    labels = {v.label for v in variants}
    assert "Q4_K_M" in labels
    assert "Q5_K_S" in labels
    assert "Q8_0" in labels


def test_split_gguf_shards_are_one_variant():
    """Split GGUF shards should be grouped as one downloadable variant."""
    files = {
        "config.json": 1000,
        "Model-Q4_K_M-00001-of-00002.gguf": 2_000_000_000,
        "Model-Q4_K_M-00002-of-00002.gguf": 2_000_000_000,
        "Model-Q8_0.gguf": 8_000_000_000,
    }

    variants, shared = discover_variants(files, {})

    labels = {v.label for v in variants}
    assert labels == {"Q4_K_M", "Q8_0"}
    q4 = next(v for v in variants if v.label == "Q4_K_M")
    assert q4.num_files == 2
    assert set(q4.files) == {
        "Model-Q4_K_M-00001-of-00002.gguf",
        "Model-Q4_K_M-00002-of-00002.gguf",
    }


def test_incomplete_split_gguf_shards_are_not_offered():
    """Do not offer a split GGUF group unless every shard is present."""
    files = {
        "config.json": 1000,
        "Model-Q4_K_M-00001-of-00002.gguf": 2_000_000_000,
        "Model-Q8_0.gguf": 8_000_000_000,
    }

    variants, shared = discover_variants(files, {})

    labels = {v.label for v in variants}
    assert labels == {"Q8_0"}
    assert "Model-Q4_K_M-00001-of-00002.gguf" not in shared


def test_incomplete_indexed_safetensors_variant_is_not_offered():
    files = {
        "model.safetensors.index.json": 50_000,
        "model-00001-of-00002.safetensors": 1_000_000,
    }
    index_contents = {
        "model.safetensors.index.json": {
            "weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors",
            },
        },
    }

    variants, shared = discover_variants(files, index_contents)

    assert variants == []
    assert "model-00001-of-00002.safetensors" not in shared
    assert "model.safetensors.index.json" not in shared


def test_gguf_iq_quant():
    """GGUF with IQ quant naming."""
    files = {"model-IQ4_XS.gguf": 3_000_000_000}
    variants, shared = discover_variants(files, {})
    assert len(variants) == 1
    assert variants[0].label == "IQ4_XS"


def test_skipped_files():
    """Files in metal/ and .git are skipped entirely."""
    files = {
        "config.json": 1000,
        "model.safetensors": 5_000_000_000,
        "metal/model.bin": 5_000_000_000,
        ".gitattributes": 500,
    }

    variants, shared = discover_variants(files, {})

    assert len(variants) == 1  # just the orphan safetensors
    assert "metal/model.bin" not in shared
    assert ".gitattributes" not in shared
    assert "config.json" in shared


def test_empty_repo_no_weights():
    """Repo with no weight files."""
    files = {
        "config.json": 1000,
        "README.md": 5000,
    }
    variants, shared = discover_variants(files, {})
    assert len(variants) == 0
    assert "config.json" in shared


def test_default_label_uses_quant_method():
    """Default variant label uses quantization_config.quant_method when available."""
    files = {
        "config.json": 1200,
        "model.safetensors.index.json": 50_000,
        "model-00001-of-00001.safetensors": 14_000_000_000,
    }
    index_contents = {
        "model.safetensors.index.json": {
            "metadata": {},
            "weight_map": {"w": "model-00001-of-00001.safetensors"},
        },
    }
    config = {"quantization_config": {"quant_method": "mxfp4"}}

    variants, shared = discover_variants(files, index_contents, config=config)

    assert variants[0].label == "mxfp4"


def test_no_quant_method_stays_default():
    """Without quantization_config, label remains 'default'."""
    files = {
        "config.json": 1200,
        "model.safetensors.index.json": 50_000,
        "model-00001-of-00001.safetensors": 14_000_000_000,
    }
    index_contents = {
        "model.safetensors.index.json": {
            "metadata": {},
            "weight_map": {"w": "model-00001-of-00001.safetensors"},
        },
    }

    variants, shared = discover_variants(files, index_contents, config={})
    assert variants[0].label == "default"


def test_fetch_repo_variants(mocker):
    """fetch_repo_variants wires HfApi calls to discover_variants."""
    import json
    import tempfile
    from unittest.mock import MagicMock
    from vserve.variants import fetch_repo_variants

    mock_api = MagicMock()

    class FakeFile:
        def __init__(self, path, size):
            self.path = path
            self.size = size

    class FakeFolder:
        def __init__(self, path):
            self.path = path

    mock_api.list_repo_tree.return_value = [
        FakeFile("config.json", 1200),
        FakeFile("tokenizer.json", 4_000_000),
        FakeFile("model.safetensors.index.json", 50_000),
        FakeFile("model-00001-of-00001.safetensors", 14_000_000_000),
        FakeFolder("some_dir"),
    ]

    index_data = {"metadata": {}, "weight_map": {"w": "model-00001-of-00001.safetensors"}}
    config_data = {"quantization_config": {"quant_method": "fp8"}}

    def fake_download(repo_id, filename, **kwargs):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        if "index" in filename:
            json.dump(index_data, tmp)
        elif filename == "config.json":
            json.dump(config_data, tmp)
        tmp.close()
        return tmp.name

    mocker.patch("huggingface_hub.hf_hub_download", side_effect=fake_download)

    variants, shared = fetch_repo_variants("test/model", mock_api)

    assert len(variants) == 1
    assert variants[0].label == "fp8"
    assert "config.json" in shared
    mock_api.list_repo_tree.assert_called_once_with("test/model", repo_type="model", recursive=True)


def test_format_variant_line_sharded():
    from vserve.variants import format_variant_line

    v = Variant(label="mxfp4", files={
        "model.safetensors.index.json": 50_000,
        "model-00001-of-00003.safetensors": 5_000_000_000,
        "model-00002-of-00003.safetensors": 5_000_000_000,
        "model-00003-of-00003.safetensors": 4_000_000_000,
    })
    line = format_variant_line(v, index=1)
    assert "1)" in line
    assert "mxfp4" in line
    assert "3 shards" in line
    assert "13.0 GB" in line


def test_format_variant_line_single_file():
    from vserve.variants import format_variant_line

    v = Variant(label="consolidated.safetensors", files={
        "consolidated.safetensors": 10_000_000_000,
    })
    line = format_variant_line(v, index=2)
    assert "2)" in line
    assert "1 file" in line
