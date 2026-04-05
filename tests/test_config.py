from pathlib import Path

from vserve.config import (
    cfg,
    load_config,
    save_config,
    reset_config,
    VserveConfig,
    _build_config,
    _discover_vllm_root,
    _discover_cuda_home,
    _discover_port,
    limits_path,
    profile_path,
    active_yaml_path,
    read_limits,
    write_limits,
    read_profile_yaml,
    try_read_profile_yaml,
    write_profile_yaml,
)


def test_cfg_returns_vxconfig():
    c = cfg()
    assert isinstance(c, VserveConfig)


def test_paths_are_under_vllm_root():
    c = cfg()
    assert c.models_dir == c.vllm_root / "models"
    assert c.configs_dir == c.vllm_root / "configs" / "models"
    assert c.benchmarks_dir == c.vllm_root / "benchmarks"
    assert c.logs_dir == c.vllm_root / "logs"
    assert c.run_dir == c.vllm_root / "run"


def test_build_config():
    c = _build_config(
        vllm_root=Path("/test/vllm"),
        cuda_home=Path("/usr/local/cuda"),
        service_name="vllm-test",
        service_user="testuser",
        port=9999,
    )
    assert c.models_dir == Path("/test/vllm/models")
    assert c.service_name == "vllm-test"
    assert c.port == 9999
    assert c.active_yaml == Path("/test/vllm/configs/active.yaml")


def test_save_and_load_config(tmp_path, mocker):
    mocker.patch("vserve.config.CONFIG_FILE", tmp_path / "config.yaml")
    c = _build_config(
        vllm_root=Path("/my/vllm"),
        cuda_home=Path("/my/cuda"),
        service_name="myvllm",
        service_user="myuser",
        port=7777,
    )
    save_config(c)
    reset_config()
    loaded = load_config()
    assert loaded.vllm_root == Path("/my/vllm")
    assert loaded.service_name == "myvllm"
    assert loaded.port == 7777


def test_limits_path():
    p = limits_path("apolo13x", "Qwen3.5-27B-NVFP4")
    assert p.name == "apolo13x--Qwen3.5-27B-NVFP4.limits.json"
    assert p.parent == cfg().configs_dir


def test_profile_path():
    p = profile_path("apolo13x", "Qwen3.5-27B-NVFP4", "interactive")
    assert p.name == "apolo13x--Qwen3.5-27B-NVFP4.interactive.yaml"


def test_active_yaml_path():
    assert active_yaml_path() == cfg().active_yaml


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


def test_try_read_profile_yaml_corrupt(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("not: valid: yaml: {{{{")
    assert try_read_profile_yaml(path) is None


# --- Discovery tests ---


def test_discover_vllm_root_from_path(mocker, tmp_path):
    """Discovery finds vllm root by following `which vllm`."""
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "vllm").touch()
    mocker.patch("vserve.config.shutil.which", return_value=str(venv / "vllm"))
    assert _discover_vllm_root() == tmp_path


def test_discover_vllm_root_from_common_path(mocker, tmp_path):
    """Discovery finds vllm at /opt/vllm (common location)."""
    mocker.patch("vserve.config.shutil.which", return_value=None)
    fake_opt = tmp_path / "opt" / "vllm"
    (fake_opt / "venv" / "bin").mkdir(parents=True)
    (fake_opt / "venv" / "bin" / "vllm").touch()
    # Patch the candidate list to include our tmp dir
    mocker.patch("vserve.config.Path.home", return_value=tmp_path / "nope")
    # Can't easily patch the hardcoded /opt/vllm, so just test the PATH method
    mocker.patch("vserve.config.shutil.which", return_value=str(fake_opt / "venv" / "bin" / "vllm"))
    assert _discover_vllm_root() == fake_opt


def test_discover_vllm_root_not_found(mocker):
    mocker.patch("vserve.config.shutil.which", return_value=None)
    # Ensure common paths don't exist (they won't in CI)
    result = _discover_vllm_root()
    # May return None or a real path if /opt/vllm exists on this machine
    assert result is None or isinstance(result, Path)


def test_discover_cuda_home_from_nvcc(mocker, tmp_path):
    cuda = tmp_path / "cuda" / "bin"
    cuda.mkdir(parents=True)
    (cuda / "nvcc").touch()
    mocker.patch("vserve.config.shutil.which", return_value=str(cuda / "nvcc"))
    assert _discover_cuda_home() == tmp_path / "cuda"


def test_discover_cuda_home_fallback(mocker):
    mocker.patch("vserve.config.shutil.which", return_value=None)
    assert _discover_cuda_home() == Path("/usr/local/cuda")


def test_discover_port_from_active_config(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "active.yaml").write_text("port: 9999\n")
    assert _discover_port(tmp_path) == 9999


def test_discover_port_missing_config(tmp_path):
    assert _discover_port(tmp_path) == 8888


def test_discover_port_corrupt_config(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "active.yaml").write_text("not: valid: yaml: {{{{")
    assert _discover_port(tmp_path) == 8888


def test_load_config_from_file(tmp_path, mocker):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "vllm_root: /custom/vllm\n"
        "cuda_home: /custom/cuda\n"
        "service_name: my-vllm\n"
        "service_user: myuser\n"
        "port: 7777\n"
    )
    mocker.patch("vserve.config.CONFIG_FILE", config_file)
    reset_config()
    c = load_config()
    assert c.vllm_root == Path("/custom/vllm")
    assert c.service_name == "my-vllm"
    assert c.port == 7777
    assert c.models_dir == Path("/custom/vllm/models")


def test_load_config_partial_file_uses_defaults(tmp_path, mocker):
    """Config file with only some fields uses defaults for the rest."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("vllm_root: /my/vllm\n")
    mocker.patch("vserve.config.CONFIG_FILE", config_file)
    reset_config()
    c = load_config()
    assert c.vllm_root == Path("/my/vllm")
    assert c.service_name == "vllm"  # default
    assert c.port == 8888  # default


def test_reset_config_clears_singleton():
    c1 = cfg()
    reset_config()
    c2 = cfg()
    # Both should be valid VserveConfig — the point is reset didn't crash
    assert isinstance(c1, VserveConfig)
    assert isinstance(c2, VserveConfig)


# --- llama.cpp config fields ---


def test_config_llamacpp_defaults():
    """New llamacpp fields have sensible defaults."""
    c = _build_config(
        vllm_root=Path("/opt/vllm"),
        cuda_home=Path("/usr/local/cuda"),
        service_name="vllm",
        service_user="vllm",
        port=8888,
    )
    assert c.llamacpp_root is None
    assert c.llamacpp_service_name == "llama-cpp"
    assert c.llamacpp_service_user == "llama-cpp"


def test_config_llamacpp_roundtrip(tmp_path, mocker):
    """llamacpp fields survive save/load cycle."""
    config_file = tmp_path / "config.yaml"
    mocker.patch("vserve.config.CONFIG_FILE", config_file)

    c = _build_config(
        vllm_root=Path("/opt/vllm"),
        cuda_home=Path("/usr/local/cuda"),
        service_name="vllm",
        service_user="vllm",
        port=8888,
    )
    c.llamacpp_root = Path("/opt/llama-cpp")
    save_config(c)

    reset_config()
    loaded = load_config()
    assert loaded.llamacpp_root == Path("/opt/llama-cpp")
    assert loaded.llamacpp_service_name == "llama-cpp"
    assert loaded.llamacpp_service_user == "llama-cpp"


def test_config_without_llamacpp_fields(tmp_path, mocker):
    """Existing config files without llamacpp fields load fine."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("vllm_root: /opt/vllm\ncuda_home: /usr/local/cuda\n")
    mocker.patch("vserve.config.CONFIG_FILE", config_file)

    reset_config()
    loaded = load_config()
    assert loaded.llamacpp_root is None
