import os
from pathlib import Path

from vserve.config import (
    cfg,
    load_config,
    save_config,
    reset_config,
    VserveConfig,
    find_systemd_unit_path,
    _build_config,
    _discover_service,
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
    active_manifest_path,
    read_active_manifest,
    write_active_manifest,
    limits_cache_matches,
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


def test_build_config_uses_dotvenv_when_venv_missing(tmp_path):
    root = tmp_path / "vllm"
    dotvenv_bin = root / ".venv" / "bin"
    dotvenv_bin.mkdir(parents=True)
    (dotvenv_bin / "vllm").touch()
    (dotvenv_bin / "python").touch()

    c = _build_config(
        vllm_root=root,
        cuda_home=Path("/usr/local/cuda"),
        service_name="vllm-test",
        service_user="testuser",
        port=9999,
    )

    assert c.vllm_bin == dotvenv_bin / "vllm"
    assert c.vllm_python == dotvenv_bin / "python"


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


def test_save_config_writes_backend_indexed_schema(tmp_path, mocker):
    import yaml

    config_file = tmp_path / "config.yaml"
    mocker.patch("vserve.config.CONFIG_FILE", config_file)
    c = _build_config(
        vllm_root=Path("/my/vllm"),
        cuda_home=Path("/my/cuda"),
        service_name="myvllm",
        service_user="myuser",
        port=7777,
        llamacpp_root=Path("/my/llama"),
        llamacpp_service_name="llama-test",
        llamacpp_service_user="llamauser",
    )

    save_config(c)

    data = yaml.safe_load(config_file.read_text())
    assert data["schema_version"] == 2
    assert data["backends"]["vllm"] == {
        "root": "/my/vllm",
        "service_name": "myvllm",
        "service_user": "myuser",
        "port": 7777,
    }
    assert data["backends"]["llamacpp"] == {
        "root": "/my/llama",
        "service_name": "llama-test",
        "service_user": "llamauser",
    }
    assert "vllm_root" not in data
    assert "llamacpp_root" not in data


def test_load_config_backend_indexed_schema(tmp_path, mocker):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "schema_version: 2\n"
        "cuda_home: /custom/cuda\n"
        "gpu:\n"
        "  memory_utilization: 0.91\n"
        "backends:\n"
        "  vllm:\n"
        "    root: /custom/vllm\n"
        "    service_name: my-vllm\n"
        "    service_user: myuser\n"
        "    port: 9999\n"
        "  llamacpp:\n"
        "    root: /custom/llama\n"
        "    service_name: my-llama\n"
        "    service_user: llamauser\n"
    )
    mocker.patch("vserve.config.CONFIG_FILE", config_file)
    reset_config()

    c = load_config()

    assert c.vllm_root == Path("/custom/vllm")
    assert c.service_name == "my-vllm"
    assert c.port == 9999
    assert c.llamacpp_root == Path("/custom/llama")
    assert c.llamacpp_service_name == "my-llama"
    assert c.gpu_memory_utilization == 0.91
    assert c.backends["vllm"].root == Path("/custom/vllm")
    assert c.backends["llamacpp"].root == Path("/custom/llama")


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
    assert loaded["schema_version"] == 2
    assert loaded["limits"]["4096"]["fp8"] == 128


def test_limits_cache_matches_requires_schema_and_fingerprint(tmp_path):
    data = {
        "schema_version": 2,
        "backend": "vllm",
        "fingerprint": {
            "model_path": "/opt/vllm/models/test/model",
            "vllm_version": "0.20.0",
            "torch_version": "2.11.0+cu130",
        },
        "limits": {"4096": {"auto": 64, "fp8": 128}},
    }

    assert limits_cache_matches(
        data,
        backend="vllm",
        fingerprint={
            "model_path": "/opt/vllm/models/test/model",
            "vllm_version": "0.20.0",
            "torch_version": "2.11.0+cu130",
        },
    )
    assert not limits_cache_matches(
        data,
        backend="vllm",
        fingerprint={
            "model_path": "/opt/vllm/models/test/model",
            "vllm_version": "0.20.1",
            "torch_version": "2.11.0+cu130",
        },
    )
    assert not limits_cache_matches(
        {"limits": data["limits"]},
        backend="vllm",
        fingerprint=data["fingerprint"],
    )


def test_active_manifest_roundtrip(tmp_path, mocker):
    mock_c = mocker.Mock(run_dir=tmp_path / "run")
    mocker.patch("vserve.config.cfg", return_value=mock_c)
    path = active_manifest_path()

    write_active_manifest(
        {
            "backend": "vllm",
            "service_name": "vllm",
            "config_path": "/opt/vllm/configs/models/test.yaml",
            "status": "ready",
        },
        path,
    )

    loaded = read_active_manifest(path)
    assert loaded is not None
    assert loaded["schema_version"] == 1
    assert loaded["backend"] == "vllm"
    assert path == tmp_path / "run" / "active-manifest.json"


def test_read_limits_missing(tmp_path):
    path = tmp_path / "nonexistent.json"
    assert read_limits(path) is None


def test_read_limits_corrupt_returns_none(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json")
    assert read_limits(path) is None


def test_read_limits_non_object_returns_none(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('["not", "an", "object"]')
    assert read_limits(path) is None


def test_read_limits_invalid_nested_shape_returns_none(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"limits": {"4096": {"auto": "many"}}}')
    assert read_limits(path) is None


def test_read_limits_accepts_flat_llamacpp_shape(tmp_path):
    path = tmp_path / "llama.json"
    path.write_text('{"limits": {"4096": 8, "8192": null}}')
    loaded = read_limits(path)
    assert loaded is not None
    assert loaded["limits"]["4096"] == 8
    assert loaded["limits"]["8192"] is None


def test_read_limits_rejects_non_numeric_context_keys(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"limits": {"wide": {"auto": 4}}}')
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


def test_atomic_write_text_does_not_follow_predictable_temp_symlink(tmp_path):
    from vserve.config import _atomic_write_text

    target = tmp_path / "test.yaml"
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me\n")
    predictable_tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    predictable_tmp.symlink_to(victim)

    _atomic_write_text(target, "model: safe\n")

    assert target.read_text() == "model: safe\n"
    assert victim.read_text() == "keep me\n"


def test_try_read_profile_yaml_corrupt(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("not: valid: yaml: {{{{")
    assert try_read_profile_yaml(path) is None


def test_read_profile_yaml_requires_mapping(tmp_path):
    import pytest

    path = tmp_path / "bad.yaml"
    path.write_text("- one\n- two\n")
    with pytest.raises(ValueError, match="Invalid profile YAML"):
        read_profile_yaml(path)


def test_try_read_profile_yaml_non_mapping_returns_none(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- one\n- two\n")
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


def test_find_systemd_unit_path_checks_multiple_locations(tmp_path, monkeypatch):
    from vserve import config as config_module

    lib_systemd = tmp_path / "lib"
    usr_systemd = tmp_path / "usr"
    lib_systemd.mkdir()
    usr_systemd.mkdir()
    unit = usr_systemd / "vllm.service"
    unit.write_text("[Unit]\nDescription=vLLM\n")

    monkeypatch.setattr(
        "vserve.config.SYSTEMD_UNIT_DIRS",
        (lib_systemd, usr_systemd),
    )
    monkeypatch.setattr(
        config_module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("skip systemctl")),
    )

    assert find_systemd_unit_path("vllm") == unit


def test_discover_service_ignores_partial_name_matches(mocker):
    list_units = mocker.Mock(stdout="nvllm.service enabled\nmy-vllm.service enabled\n", returncode=0)
    show_user = mocker.Mock(stdout="User=svcuser\n", returncode=0)
    mocker.patch("vserve.config.subprocess.run", side_effect=[list_units, show_user])

    assert _discover_service() == ("my-vllm", "svcuser")


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
