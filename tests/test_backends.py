"""Tests for the backend registry and protocol."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from vserve.backends import register, get_backend, get_backend_by_name, available_backends
from vserve.backends.protocol import Backend
from vserve.backends.vllm import VllmBackend


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry before each test."""
    from vserve.backends import _BACKENDS
    saved = _BACKENDS.copy()
    _BACKENDS.clear()
    yield
    _BACKENDS.clear()
    _BACKENDS.extend(saved)


def _make_mock_backend(name: str, can_serve_result: bool = True, has_binary: bool = True) -> Mock:
    b = Mock(spec=Backend)
    b.name = name
    b.display_name = name
    b.can_serve.return_value = can_serve_result
    b.find_entrypoint.return_value = Path("/fake/bin") if has_binary else None
    return b


def test_register_and_get_backend_by_name():
    b = _make_mock_backend("test")
    register(b)
    assert get_backend_by_name("test") is b


def test_get_backend_by_name_unknown():
    with pytest.raises(KeyError):
        get_backend_by_name("nonexistent")


def test_get_backend_auto_detect():
    b1 = _make_mock_backend("a", can_serve_result=False)
    b2 = _make_mock_backend("b", can_serve_result=True)
    register(b1)
    register(b2)

    model = Mock()
    assert get_backend(model) is b2


def test_get_backend_no_match():
    b = _make_mock_backend("a", can_serve_result=False)
    register(b)

    model = Mock()
    with pytest.raises(ValueError, match="No backend"):
        get_backend(model)


def test_available_backends_filters_missing():
    b1 = _make_mock_backend("installed", has_binary=True)
    b2 = _make_mock_backend("missing", has_binary=False)
    register(b1)
    register(b2)

    result = available_backends()
    assert len(result) == 1
    assert result[0].name == "installed"


# --- VllmBackend tests ---


class TestVllmBackend:
    def test_identity(self):
        b = VllmBackend()
        assert b.name == "vllm"
        assert b.display_name == "vLLM"

    def test_can_serve_safetensors(self, fake_model_dir):
        b = VllmBackend()
        from vserve.models import detect_model
        m = detect_model(fake_model_dir)
        assert b.can_serve(m) is True

    def test_cannot_serve_gguf_only(self, tmp_path):
        b = VllmBackend()
        model_dir = tmp_path / "models" / "user" / "Model-GGUF"
        model_dir.mkdir(parents=True)
        (model_dir / "model-Q4_K_M.gguf").write_bytes(b"\0" * 100)

        from vserve.models import ModelInfo
        m = ModelInfo(
            path=model_dir, provider="user", model_name="Model-GGUF",
            architecture="unknown", model_type="unknown", quant_method=None,
            max_position_embeddings=4096, is_moe=False, model_size_gb=0.1,
        )
        assert b.can_serve(m) is False

    def test_health_url(self):
        b = VllmBackend()
        assert b.health_url(8888) == "http://localhost:8888/health"

    def test_quant_flag(self):
        b = VllmBackend()
        assert "gptq_marlin" in b.quant_flag("gptq")
        assert "fp8" in b.quant_flag("fp8")
        assert b.quant_flag(None) == ""
        assert b.quant_flag("compressed-tensors") == ""

    def test_tune_delegates_to_probe(self, fake_model_dir):
        b = VllmBackend()
        from vserve.models import detect_model
        m = detect_model(fake_model_dir)

        mock_gpu = Mock()
        mock_gpu.vram_total_gb = 48.0

        result = b.tune(m, mock_gpu, gpu_mem_util=0.90)
        assert "limits" in result
        assert "tool_call_parser" in result

    def test_detect_tools(self, fake_model_dir):
        b = VllmBackend()
        result = b.detect_tools(fake_model_dir)
        assert "tool_call_parser" in result
        assert "reasoning_parser" in result

    def test_start_stop_delegates(self, mocker, tmp_path):
        b = VllmBackend()
        mock_start = mocker.patch("vserve.serve.start_vllm")
        mock_stop = mocker.patch("vserve.serve.stop_vllm")
        mocker.patch("vserve.serve.is_vllm_running", return_value=True)

        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text("model: test\n")
        b.start(cfg_path)
        mock_start.assert_called_once_with(cfg_path)

        b.stop()
        mock_stop.assert_called_once()

        assert b.is_running() is True

    def test_build_config_basic(self, fake_model_dir):
        b = VllmBackend()
        from vserve.models import detect_model
        m = detect_model(fake_model_dir)

        choices = {
            "context": 8192,
            "kv_dtype": "auto",
            "slots": 4,
            "batched_tokens": None,
            "gpu_mem_util": 0.90,
            "port": 8888,
            "tools": False,
            "tool_parser": None,
            "reasoning_parser": None,
        }
        cfg = b.build_config(m, choices)
        assert cfg["max-model-len"] == 8192
        assert cfg["max-num-seqs"] == 4
        assert cfg["kv-cache-dtype"] == "auto"
        assert cfg["gpu-memory-utilization"] == 0.90
        assert "enable-auto-tool-choice" not in cfg

    def test_build_config_with_tools(self, fake_model_dir):
        b = VllmBackend()
        from vserve.models import detect_model
        m = detect_model(fake_model_dir)

        choices = {
            "context": 8192,
            "kv_dtype": "auto",
            "slots": 4,
            "batched_tokens": 4096,
            "gpu_mem_util": 0.90,
            "port": 8888,
            "tools": True,
            "tool_parser": "hermes",
            "reasoning_parser": "qwen3",
        }
        cfg = b.build_config(m, choices)
        assert cfg["enable-auto-tool-choice"] is True
        assert cfg["tool-call-parser"] == "hermes"
        assert cfg["reasoning-parser"] == "qwen3"
        assert cfg["max-num-batched-tokens"] == 4096

    def test_find_entrypoint_missing(self, mocker):
        b = VllmBackend()
        mock_c = Mock()
        mock_c.vllm_bin = Path("/nonexistent/vllm")
        mocker.patch("vserve.config.cfg", return_value=mock_c)
        assert b.find_entrypoint() is None

    def test_find_entrypoint_exists(self, mocker, tmp_path):
        b = VllmBackend()
        vllm_bin = tmp_path / "venv" / "bin" / "vllm"
        vllm_bin.parent.mkdir(parents=True)
        vllm_bin.touch()
        mock_c = Mock()
        mock_c.vllm_bin = vllm_bin
        mocker.patch("vserve.config.cfg", return_value=mock_c)
        assert b.find_entrypoint() == vllm_bin

    def test_doctor_checks_returns_callables(self):
        b = VllmBackend()
        checks = b.doctor_checks()
        assert len(checks) >= 2
        for desc, fn in checks:
            assert isinstance(desc, str)
            assert callable(fn)

    def test_service_identity(self):
        b = VllmBackend()
        assert b.service_name == "vllm"
        assert b.service_user == "vllm"


# --- Auto-registration ---


def test_default_backends_registered():
    """Built-in backends are registered after _register_defaults."""
    from vserve.backends import _register_defaults
    _register_defaults()

    from vserve.backends import _BACKENDS
    names = [b.name for b in _BACKENDS]
    assert "vllm" in names
    assert "llamacpp" in names


def test_duplicate_registration_prevented():
    """register() with same name is idempotent."""
    b1 = _make_mock_backend("dedup")
    b2 = _make_mock_backend("dedup")
    register(b1)
    register(b2)
    from vserve.backends import _BACKENDS
    assert sum(1 for b in _BACKENDS if b.name == "dedup") == 1
