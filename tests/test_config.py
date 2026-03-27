from pathlib import Path
from vx.config import (
    MODELS_DIR,
    CONFIGS_DIR,
    BENCHMARKS_DIR,
    limits_path,
    profile_path,
    active_yaml_path,
    read_limits,
    write_limits,
    read_profile_yaml,
    write_profile_yaml,
)


def test_paths_are_under_opt_vllm():
    assert str(MODELS_DIR).startswith("/opt/vllm")
    assert str(CONFIGS_DIR).startswith("/opt/vllm")
    assert str(BENCHMARKS_DIR).startswith("/opt/vllm")


def test_limits_path():
    p = limits_path("apolo13x", "Qwen3.5-27B-NVFP4")
    assert p == CONFIGS_DIR / "apolo13x--Qwen3.5-27B-NVFP4.limits.json"


def test_profile_path():
    p = profile_path("apolo13x", "Qwen3.5-27B-NVFP4", "interactive")
    assert p == CONFIGS_DIR / "apolo13x--Qwen3.5-27B-NVFP4.interactive.yaml"


def test_active_yaml_path():
    assert active_yaml_path() == Path("/opt/vllm/configs/active.yaml")


def test_write_and_read_limits(tmp_path):
    data = {
        "model_path": "/opt/vllm/models/test/model",
        "limits": {"4096": {"auto": 64, "fp8": 128}},
    }
    path = tmp_path / "test.limits.json"
    write_limits(path, data)
    loaded = read_limits(path)
    assert loaded["limits"]["4096"]["fp8"] == 128


def test_read_limits_missing(tmp_path):
    path = tmp_path / "nonexistent.json"
    assert read_limits(path) is None


def test_write_and_read_profile_yaml(tmp_path):
    data = {
        "model": "/opt/vllm/models/test/model",
        "host": "0.0.0.0",
        "port": 8888,
        "gpu-memory-utilization": 0.958,
    }
    path = tmp_path / "test.yaml"
    write_profile_yaml(path, data, comment="test profile")
    content = path.read_text()
    assert "# test profile" in content
    loaded = read_profile_yaml(path)
    assert loaded["port"] == 8888
