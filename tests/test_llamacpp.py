"""Tests for the llama.cpp backend."""

from unittest.mock import Mock

from vserve.backends.llamacpp import LlamaCppBackend


class TestLlamaCppCanServe:
    def test_can_serve_gguf(self, fake_gguf_model_dir):
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)
        assert b.can_serve(m) is True

    def test_cannot_serve_safetensors(self, fake_model_dir):
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_model_dir)
        assert b.can_serve(m) is False


class TestLlamaCppIdentity:
    def test_name(self):
        b = LlamaCppBackend()
        assert b.name == "llamacpp"
        assert b.display_name == "llama.cpp"
        assert b.service_name == "llama-cpp"
        assert b.service_user == "llama-cpp"


class TestLlamaCppHealthUrl:
    def test_health_url(self):
        b = LlamaCppBackend()
        assert b.health_url(8888) == "http://localhost:8888/health"
        assert b.health_url(9999) == "http://localhost:9999/health"


class TestLlamaCppBuildConfig:
    def test_basic_config(self, fake_gguf_model_dir):
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)

        choices = {
            "context": 8192,
            "n_gpu_layers": 35,
            "parallel": 4,
            "port": 8888,
            "tools": False,
        }
        cfg = b.build_config(m, choices)
        assert cfg["ctx_size"] == 8192
        assert cfg["n_gpu_layers"] == 35
        assert cfg["parallel"] == 4
        assert cfg["flash_attn"] is True
        assert "cont_batching" not in cfg
        assert "cont-batching" not in cfg
        assert "jinja" not in cfg

    def test_config_with_tools(self, fake_gguf_model_dir):
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)

        choices = {
            "context": 8192,
            "n_gpu_layers": 35,
            "parallel": 4,
            "port": 8888,
            "tools": True,
        }
        cfg = b.build_config(m, choices)
        assert cfg["jinja"] is True

    def test_config_model_path_is_gguf_file(self, fake_gguf_model_dir):
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)

        choices = {"context": 4096, "n_gpu_layers": 10, "parallel": 1, "port": 8888, "tools": False}
        cfg = b.build_config(m, choices)
        assert cfg["model"].endswith(".gguf")


class TestLlamaCppQuant:
    def test_quant_flag_always_empty(self):
        b = LlamaCppBackend()
        assert b.quant_flag("gptq") == ""
        assert b.quant_flag(None) == ""
        assert b.quant_flag("fp8") == ""


class TestLlamaCppDetectTools:
    def test_detect_tools_with_template(self, fake_gguf_model_dir):
        b = LlamaCppBackend()
        result = b.detect_tools(fake_gguf_model_dir)
        assert "supports_tools" in result
        assert result["supports_tools"] is True

    def test_detect_tools_no_template(self, tmp_path):
        b = LlamaCppBackend()
        model_dir = tmp_path / "models" / "u" / "m"
        model_dir.mkdir(parents=True)
        (model_dir / "model.gguf").write_bytes(b"\0" * 100)
        result = b.detect_tools(model_dir)
        assert result["supports_tools"] is False


class TestLlamaCppTune:
    def _mock_metadata(self, mocker):
        mocker.patch.object(
            LlamaCppBackend, "_read_gguf_metadata",
            return_value={
                "arch": "llama",
                "num_layers": 32,
                "max_context": 8192,
                "num_kv_heads": 8,
                "head_dim": 128,
            },
        )

    def test_tune_basic(self, fake_gguf_model_dir, mocker):
        """Tune produces expected output structure."""
        self._mock_metadata(mocker)
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)

        gpu = Mock()
        gpu.vram_total_gb = 48.0

        result = b.tune(m, gpu, gpu_mem_util=0.90)
        assert "n_gpu_layers" in result
        assert "limits" in result
        assert "model_path" in result
        assert "supports_tools" in result
        assert result["backend"] == "llamacpp"

    def test_tune_full_offload_with_tiny_model(self, fake_gguf_model_dir, mocker):
        """Tiny model fully fits on GPU."""
        self._mock_metadata(mocker)
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)

        gpu = Mock()
        gpu.vram_total_gb = 48.0

        result = b.tune(m, gpu, gpu_mem_util=0.90)
        assert result["full_offload"] is True


class TestLlamaCppLifecycle:
    def test_start_calls_systemctl(self, mocker, tmp_path):
        b = LlamaCppBackend()
        mock_run = mocker.patch("vserve.backends.llamacpp.subprocess.run",
                                return_value=Mock(returncode=0, stdout="", stderr=""))
        mocker.patch("vserve.backends.llamacpp.shutil.copy2")

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text('{"model": "test"}')

        # Mock active config path
        active = tmp_path / "configs" / "active.json"
        active.parent.mkdir(parents=True)
        mocker.patch.object(b, "_active_config_path", return_value=active)

        b.start(cfg_path)
        calls = [c for c in mock_run.call_args_list if "start" in str(c)]
        assert len(calls) >= 1

    def test_stop_calls_systemctl(self, mocker):
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=0, stdout="", stderr=""))
        b.stop()

    def test_is_running(self, mocker):
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=0, stdout="active", stderr=""))
        assert b.is_running() is True

    def test_is_not_running(self, mocker):
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=3, stdout="inactive", stderr=""))
        assert b.is_running() is False


class TestLlamaCppFindEntrypoint:
    def test_find_on_path(self, mocker):
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.shutil.which", return_value="/usr/bin/llama-server")
        # Mock root_dir to a nonexistent path
        mocker.patch.object(type(b), "root_dir", new_callable=lambda: property(lambda self: __import__("pathlib").Path("/nonexistent")))
        result = b.find_entrypoint()
        assert result is not None

    def test_not_found(self, mocker):
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.shutil.which", return_value=None)
        mocker.patch.object(type(b), "root_dir", new_callable=lambda: property(lambda self: __import__("pathlib").Path("/nonexistent")))
        assert b.find_entrypoint() is None


class TestLlamaCppDoctorChecks:
    def test_returns_callables(self):
        b = LlamaCppBackend()
        checks = b.doctor_checks()
        assert len(checks) >= 2
        for desc, fn in checks:
            assert isinstance(desc, str)
            assert callable(fn)


class TestLlamaCppStartFailure:
    def test_start_raises_on_failure(self, mocker, tmp_path):
        import pytest
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=1, stdout="", stderr="Unit not found"))
        mocker.patch("vserve.backends.llamacpp.shutil.copy2")
        active = tmp_path / "active.json"
        mocker.patch.object(b, "_active_config_path", return_value=active)

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text('{}')
        with pytest.raises(RuntimeError, match="systemctl start"):
            b.start(cfg_path)
