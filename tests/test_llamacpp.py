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

    def test_configured_service_identity(self, mocker):
        b = LlamaCppBackend()
        mocker.patch(
            "vserve.config.cfg",
            return_value=Mock(
                llamacpp_service_name="custom-llama",
                llamacpp_service_user="svc-llama",
                llamacpp_root=None,
            ),
        )

        assert b.service_name == "custom-llama"
        assert b.service_user == "svc-llama"


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

    def test_config_selects_one_coherent_split_shard_set(self, tmp_path):
        b = LlamaCppBackend()
        model_dir = tmp_path / "models" / "provider" / "Model"
        model_dir.mkdir(parents=True)
        for name, size in {
            "Model-Q4_K_M-00001-of-00002.gguf": 1,
            "Model-Q4_K_M-00002-of-00002.gguf": 1,
            "Model-Q8_0-00001-of-00002.gguf": 3,
            "Model-Q8_0-00002-of-00002.gguf": 3,
        }.items():
            (model_dir / name).write_bytes(b"\0" * size)
        from vserve.models import detect_model
        m = detect_model(model_dir)

        choices = {"context": 4096, "n_gpu_layers": 10, "parallel": 1, "port": 8888, "tools": False}
        cfg = b.build_config(m, choices)

        assert cfg["model"].endswith("Model-Q8_0-00001-of-00002.gguf")

    def test_config_rejects_incomplete_split_shard_set(self, tmp_path):
        import pytest

        b = LlamaCppBackend()
        model_dir = tmp_path / "models" / "provider" / "Model"
        model_dir.mkdir(parents=True)
        (model_dir / "Model-Q4_K_M-00001-of-00002.gguf").write_bytes(b"\0")
        from vserve.models import detect_model
        m = detect_model(model_dir)

        choices = {"context": 4096, "n_gpu_layers": 10, "parallel": 1, "port": 8888, "tools": False}

        with pytest.raises(ValueError, match="Incomplete split GGUF"):
            b.build_config(m, choices)

    def test_config_embedding_mode(self, fake_gguf_model_dir):
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)

        choices = {
            "context": 512,
            "n_gpu_layers": 10,
            "parallel": 8,
            "port": 8888,
            "embedding": True,
            "pooling": "mean",
        }
        cfg = b.build_config(m, choices)
        assert cfg["embedding"] is True
        assert cfg["pooling"] == "mean"
        assert "jinja" not in cfg  # no tool calling in embedding mode

    def test_config_embedding_cls_pooling(self, fake_gguf_model_dir):
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)

        choices = {
            "context": 512,
            "n_gpu_layers": 10,
            "parallel": 1,
            "port": 8888,
            "embedding": True,
            "pooling": "cls",
        }
        cfg = b.build_config(m, choices)
        assert cfg["pooling"] == "cls"


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

    def test_tune_uses_one_coherent_split_shard_set(self, tmp_path, mocker):
        """Multiple split GGUF variants in one dir should not be summed together."""
        self._mock_metadata(mocker)
        mocker.patch.object(LlamaCppBackend, "detect_tools", return_value={})
        b = LlamaCppBackend()
        model_dir = tmp_path / "models" / "provider" / "Model"
        model_dir.mkdir(parents=True)
        for name, size_gb in {
            "Model-Q4_K_M-00001-of-00002.gguf": 1,
            "Model-Q4_K_M-00002-of-00002.gguf": 1,
            "Model-Q8_0-00001-of-00002.gguf": 3,
            "Model-Q8_0-00002-of-00002.gguf": 3,
        }.items():
            path = model_dir / name
            with path.open("wb") as f:
                f.truncate(size_gb * 1024**3)
        from vserve.models import detect_model
        m = detect_model(model_dir)

        gpu = Mock()
        gpu.vram_total_gb = 48.0

        result = b.tune(m, gpu, gpu_mem_util=0.90)

        assert result["model_size_gb"] == 6.0


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

    def test_start_rejects_invalid_json_config(self, tmp_path):
        import pytest

        b = LlamaCppBackend()
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text("{not json")

        with pytest.raises(RuntimeError, match="Invalid llama.cpp config"):
            b.start(cfg_path)

    def test_stop_calls_systemctl(self, mocker):
        b = LlamaCppBackend()
        mock_run = mocker.patch("vserve.backends.llamacpp.subprocess.run",
                                return_value=Mock(returncode=0, stdout="", stderr=""))
        b.stop()
        assert mock_run.call_args[0][0][-1] == "llama-cpp"

    def test_is_running(self, mocker):
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=0, stdout="active", stderr=""))
        assert b.is_running() is True

    def test_is_running_raises_on_service_error(self, mocker):
        import pytest

        b = LlamaCppBackend()
        mocker.patch(
            "vserve.backends.llamacpp.subprocess.run",
            return_value=Mock(returncode=1, stdout="", stderr="dbus error"),
        )
        with pytest.raises(RuntimeError, match="systemctl is-active"):
            b.is_running()

    def test_is_running_missing_unit_is_false(self, mocker):
        b = LlamaCppBackend()
        mocker.patch(
            "vserve.backends.llamacpp.subprocess.run",
            return_value=Mock(returncode=1, stdout="", stderr="Unit llama-cpp.service could not be found."),
        )
        assert b.is_running() is False

    def test_is_running_activating_is_uncertain(self, mocker):
        import pytest

        b = LlamaCppBackend()
        mocker.patch(
            "vserve.backends.llamacpp.subprocess.run",
            return_value=Mock(returncode=3, stdout="activating", stderr=""),
        )
        with pytest.raises(RuntimeError, match="is transitional"):
            b.is_running()

    def test_start_uses_configured_service_name(self, mocker, tmp_path):
        b = LlamaCppBackend()
        mock_run = mocker.patch(
            "vserve.backends.llamacpp.subprocess.run",
            return_value=Mock(returncode=0, stdout="", stderr=""),
        )
        mocker.patch(
            "vserve.config.cfg",
            return_value=Mock(
                llamacpp_service_name="custom-llama",
                llamacpp_service_user="svc-llama",
                llamacpp_root=None,
            ),
        )

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text('{"model": "test"}')
        active = tmp_path / "configs" / "active.json"
        active.parent.mkdir(parents=True)
        mocker.patch.object(b, "_active_config_path", return_value=active)

        b.start(cfg_path)

        start_calls = [call for call in mock_run.call_args_list if "start" in call.args[0]]
        assert start_calls
        assert start_calls[-1].args[0][-1] == "custom-llama"

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


class TestLlamaCppRuntimeInfo:
    def test_runtime_info_calls_llama_server_version(self, mocker, tmp_path):
        b = LlamaCppBackend()
        exe = tmp_path / "llama-server"
        exe.write_text("")
        mocker.patch.object(b, "find_entrypoint", return_value=exe)
        run = mocker.patch(
            "vserve.backends.llamacpp.subprocess.run",
            return_value=Mock(returncode=0, stdout="llama-server 2026\n", stderr=""),
        )

        info = b.runtime_info()

        assert info["llama_server_version"] == "llama-server 2026"
        run.assert_called_once_with([str(exe), "--version"], capture_output=True, text=True, timeout=10)

    def test_compatibility_fails_without_entrypoint(self, mocker):
        b = LlamaCppBackend()
        mocker.patch.object(b, "find_entrypoint", return_value=None)

        result = b.compatibility()

        assert result["supported"] is False
        assert "entrypoint not found" in result["errors"][0]


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
        active = tmp_path / "active.sh"
        mocker.patch.object(b, "_active_config_path", return_value=active)

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text('{}')
        with pytest.raises(RuntimeError, match="systemctl start"):
            b.start(cfg_path)

    def test_start_failure_rolls_back_active_links_and_writes_manifest(self, mocker, tmp_path):
        import json
        import pytest

        from vserve.config import read_active_manifest

        b = LlamaCppBackend()
        mocker.patch(
            "vserve.backends.llamacpp.subprocess.run",
            return_value=Mock(returncode=1, stdout="", stderr="Unit not found"),
        )
        mocker.patch.object(b, "find_entrypoint", return_value="/opt/llama-cpp/bin/llama-server")

        active = tmp_path / "configs" / "active.sh"
        active.parent.mkdir(parents=True)
        previous_script = tmp_path / "configs" / "models" / "previous.sh"
        previous_json = tmp_path / "configs" / "models" / "previous.json"
        previous_script.parent.mkdir(parents=True)
        previous_script.write_text("#!/bin/bash\n")
        previous_json.write_text("{}")
        active.symlink_to(previous_script)
        active_json = active.with_suffix(".json")
        active_json.symlink_to(previous_json)
        manifest_path = tmp_path / "run" / "active-manifest.json"

        mocker.patch.object(b, "_active_config_path", return_value=active)
        mocker.patch.object(b, "active_manifest_path", return_value=manifest_path)

        cfg_path = tmp_path / "configs" / "models" / "next.json"
        cfg_path.write_text(json.dumps({"model": "test.gguf", "port": 8888}))

        with pytest.raises(RuntimeError, match="systemctl start"):
            b.start(cfg_path)

        assert active.resolve() == previous_script.resolve()
        assert active_json.resolve() == previous_json.resolve()
        manifest = read_active_manifest(manifest_path)
        assert manifest is not None
        assert manifest["status"] == "failed"
        assert "Unit not found" in manifest["error"]


class TestLlamaCppLaunchScript:
    def test_script_content(self, mocker, tmp_path):
        """Launch script has correct flags and is shell-safe."""
        import json
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=0, stdout="", stderr=""))

        active = tmp_path / "configs" / "active.sh"
        mocker.patch.object(b, "_active_config_path", return_value=active)
        mocker.patch.object(b, "find_entrypoint", return_value="/opt/llama-cpp/bin/llama-server")

        cfg = {
            "model": "/opt/llama-cpp/models/test/model.gguf",
            "host": "0.0.0.0",
            "port": 8888,
            "ctx_size": 8192,
            "n_gpu_layers": 32,
            "parallel": 4,
            "flash_attn": True,
            "jinja": True,
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        b.start(cfg_path)

        script = active.read_text()
        assert script.startswith("#!/bin/bash\nexport CUDA_VISIBLE_DEVICES=0\nexec ")
        assert "-m /opt/llama-cpp/models/test/model.gguf" in script
        assert "-c 8192" in script
        assert "-ngl 32" in script
        assert "-np 4" in script
        assert "-fa on" in script
        assert "--jinja" in script
        assert "--host 0.0.0.0" in script
        assert "--port 8888" in script

    def test_script_exports_configured_gpu_index(self, mocker, tmp_path):
        import json
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=0, stdout="", stderr=""))
        mocker.patch("vserve.config.cfg", return_value=Mock(gpu_index=2))

        active = tmp_path / "configs" / "active.sh"
        mocker.patch.object(b, "_active_config_path", return_value=active)
        mocker.patch.object(b, "find_entrypoint", return_value="/opt/llama-cpp/bin/llama-server")

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({
            "model": "/m/model.gguf",
            "host": "0.0.0.0",
            "port": 8888,
            "ctx_size": 4096,
            "n_gpu_layers": 10,
            "parallel": 1,
        }))

        b.start(cfg_path)

        script = active.read_text()
        assert "export CUDA_VISIBLE_DEVICES=2\n" in script

    def test_script_quoting_spaces(self, mocker, tmp_path):
        """Model paths with spaces are properly quoted in script."""
        import json
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=0, stdout="", stderr=""))

        active = tmp_path / "configs" / "active.sh"
        mocker.patch.object(b, "_active_config_path", return_value=active)
        mocker.patch.object(b, "find_entrypoint", return_value="/opt/llama-cpp/bin/llama-server")

        cfg = {
            "model": "/opt/models/My Model Dir/model file.gguf",
            "host": "0.0.0.0",
            "port": 8888,
            "ctx_size": 4096,
            "n_gpu_layers": 10,
            "parallel": 1,
            "flash_attn": True,
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        b.start(cfg_path)

        script = active.read_text()
        # shlex.join should quote the path with spaces using single quotes
        assert "My Model Dir" in script
        assert "model file.gguf" in script
        # The path should be single-quoted by shlex
        assert "'/opt/models/My Model Dir/model file.gguf'" in script


class TestLlamaCppEmbedding:
    def _mock_metadata(self, mocker, pooling=None):
        meta = {
            "arch": "nomic-bert",
            "num_layers": 12,
            "max_context": 8192,
            "num_kv_heads": 12,
            "head_dim": 64,
            "pooling": pooling,
        }
        mocker.patch.object(LlamaCppBackend, "_read_gguf_metadata", return_value=meta)

    def test_tune_embedding_model(self, fake_embedding_model_dir, mocker):
        """tune() returns is_embedding and pooling for embedding models."""
        self._mock_metadata(mocker, pooling="mean")
        mocker.patch.object(LlamaCppBackend, "detect_tools", return_value={})
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_embedding_model_dir)

        gpu = Mock()
        gpu.vram_total_gb = 48.0

        result = b.tune(m, gpu, gpu_mem_util=0.90)
        assert result["is_embedding"] is True
        assert result["pooling"] == "mean"
        assert result["supports_tools"] is False

    def test_tune_embedding_guesses_pooling(self, fake_embedding_model_dir, mocker):
        """tune() guesses pooling when GGUF metadata lacks it."""
        self._mock_metadata(mocker, pooling=None)
        mocker.patch.object(LlamaCppBackend, "detect_tools", return_value={})
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_embedding_model_dir)

        gpu = Mock()
        gpu.vram_total_gb = 48.0

        result = b.tune(m, gpu, gpu_mem_util=0.90)
        assert result["is_embedding"] is True
        assert result["pooling"] == "mean"  # nomic → mean

    def test_tune_non_embedding_has_no_embedding_key(self, fake_gguf_model_dir, mocker):
        """tune() for non-embedding models has no is_embedding key."""
        mocker.patch.object(LlamaCppBackend, "_read_gguf_metadata", return_value={
            "arch": "llama", "num_layers": 32, "max_context": 8192,
            "num_kv_heads": 8, "head_dim": 128, "pooling": None,
        })
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)

        gpu = Mock()
        gpu.vram_total_gb = 48.0

        result = b.tune(m, gpu, gpu_mem_util=0.90)
        assert "is_embedding" not in result
        assert "pooling" not in result

    def test_build_config_embedding_no_jinja(self, fake_embedding_model_dir):
        """Embedding config has --embedding but not --jinja."""
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_embedding_model_dir)

        choices = {
            "context": 512, "n_gpu_layers": 12, "parallel": 8,
            "port": 8888, "embedding": True, "pooling": "mean",
        }
        cfg = b.build_config(m, choices)
        assert cfg["embedding"] is True
        assert cfg["pooling"] == "mean"
        assert "jinja" not in cfg

    def test_start_script_embedding_flags(self, mocker, tmp_path):
        """Launch script includes --embedding and --pooling flags."""
        import json
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=0, stdout="", stderr=""))

        active = tmp_path / "configs" / "active.sh"
        mocker.patch.object(b, "_active_config_path", return_value=active)
        mocker.patch.object(b, "find_entrypoint", return_value="/opt/llama-cpp/bin/llama-server")

        cfg = {
            "model": "/opt/llama-cpp/models/nomic/embed.gguf",
            "host": "0.0.0.0", "port": 8888,
            "ctx_size": 8192, "n_gpu_layers": 12, "parallel": 8,
            "flash_attn": True,
            "embedding": True, "pooling": "mean",
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        b.start(cfg_path)

        script = active.read_text()
        assert "--embedding" in script
        assert "--pooling mean" in script
        assert "--jinja" not in script

    def test_start_script_cls_pooling(self, mocker, tmp_path):
        """Launch script respects cls pooling for BGE models."""
        import json
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=0, stdout="", stderr=""))

        active = tmp_path / "configs" / "active.sh"
        mocker.patch.object(b, "_active_config_path", return_value=active)
        mocker.patch.object(b, "find_entrypoint", return_value="/opt/llama-cpp/bin/llama-server")

        cfg = {
            "model": "/opt/llama-cpp/models/bge/model.gguf",
            "host": "0.0.0.0", "port": 8888,
            "ctx_size": 512, "n_gpu_layers": 12, "parallel": 1,
            "flash_attn": True,
            "embedding": True, "pooling": "cls",
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        b.start(cfg_path)

        script = active.read_text()
        assert "--pooling cls" in script

    def test_start_script_no_pooling_when_absent(self, mocker, tmp_path):
        """No --pooling flag when pooling is not set."""
        import json
        b = LlamaCppBackend()
        mocker.patch("vserve.backends.llamacpp.subprocess.run",
                     return_value=Mock(returncode=0, stdout="", stderr=""))

        active = tmp_path / "configs" / "active.sh"
        mocker.patch.object(b, "_active_config_path", return_value=active)
        mocker.patch.object(b, "find_entrypoint", return_value="/opt/llama-cpp/bin/llama-server")

        cfg = {
            "model": "/m/model.gguf", "host": "0.0.0.0", "port": 8888,
            "ctx_size": 4096, "n_gpu_layers": 10, "parallel": 1,
            "flash_attn": True,
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        b.start(cfg_path)

        script = active.read_text()
        assert "--pooling" not in script
        assert "--embedding" not in script

    def test_build_config_no_tools_no_embedding(self, fake_gguf_model_dir):
        """Config with neither tools nor embedding has no jinja or embedding flags."""
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)

        choices = {
            "context": 4096, "n_gpu_layers": 10, "parallel": 1,
            "port": 8888, "tools": False,
        }
        cfg = b.build_config(m, choices)
        assert "jinja" not in cfg
        assert "embedding" not in cfg
        assert "pooling" not in cfg

    def test_guess_pooling_case_insensitive(self):
        assert LlamaCppBackend._guess_pooling("BGE-Large-EN-v1.5") == "cls"
        assert LlamaCppBackend._guess_pooling("NOMIC-EMBED-TEXT") == "mean"
        assert LlamaCppBackend._guess_pooling("Jina-Reranker-v2") == "rank"
