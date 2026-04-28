from pathlib import Path

from vserve.runtime import (
    SUPPORTED_VLLM_RANGE,
    DETECTOR_SCHEMA_VERSION,
    RuntimeInfo,
    build_tuning_fingerprint,
    check_vllm_compatibility,
    collect_vllm_runtime_info,
    upgrade_vllm_stable,
)


def test_check_vllm_compatibility_accepts_stable_020():
    info = RuntimeInfo(
        backend="vllm",
        executable=Path("/opt/vllm/venv/bin/vllm"),
        python=Path("/opt/vllm/venv/bin/python"),
        vllm_version="0.20.0",
        torch_version="2.11.0+cu130",
        torch_cuda="13.0",
        transformers_version="5.6.2",
        huggingface_hub_version="1.12.0",
        pip_check_ok=True,
        pip_check_output="No broken requirements found",
    )

    result = check_vllm_compatibility(info)

    assert result.supported is True
    assert result.range == SUPPORTED_VLLM_RANGE
    assert result.errors == []
    assert any("0.20.0" in message for message in result.messages)


def test_check_vllm_compatibility_rejects_beta_and_wrong_minor():
    beta = RuntimeInfo(
        backend="vllm",
        executable=Path("/opt/vllm/venv/bin/vllm"),
        python=Path("/opt/vllm/venv/bin/python"),
        vllm_version="0.20.0rc1",
        torch_version="2.11.0+cu130",
        torch_cuda="13.0",
        transformers_version="5.6.2",
        huggingface_hub_version="1.12.0",
        pip_check_ok=True,
        pip_check_output="No broken requirements found",
    )
    older = RuntimeInfo(
        backend="vllm",
        executable=Path("/opt/vllm/venv/bin/vllm"),
        python=Path("/opt/vllm/venv/bin/python"),
        vllm_version="0.19.2",
        torch_version="2.10.0",
        torch_cuda="12.8",
        transformers_version="4.56.0",
        huggingface_hub_version="0.36.0",
        pip_check_ok=True,
        pip_check_output="No broken requirements found",
    )

    assert check_vllm_compatibility(beta).supported is False
    assert "pre-release" in " ".join(check_vllm_compatibility(beta).errors)
    assert check_vllm_compatibility(older).supported is False
    assert ">=0.20,<0.21" in " ".join(check_vllm_compatibility(older).errors)


def test_collect_vllm_runtime_info_uses_vllm_python(mocker, tmp_path):
    vllm_bin = tmp_path / "venv" / "bin" / "vllm"
    vllm_python = tmp_path / "venv" / "bin" / "python"
    vllm_bin.parent.mkdir(parents=True)
    vllm_bin.touch()
    vllm_python.touch()
    cfg = mocker.Mock(vllm_bin=vllm_bin, vllm_python=vllm_python)

    def fake_run(cmd, *args, **kwargs):
        if cmd == [str(vllm_bin), "--version"]:
            return mocker.Mock(returncode=0, stdout="vLLM 0.20.0\n", stderr="")
        if cmd[:3] == [str(vllm_python), "-c", mocker.ANY]:
            return mocker.Mock(
                returncode=0,
                stdout=(
                    '{"vllm":"0.20.0","torch":"2.11.0+cu130","torch_cuda":"13.0",'
                    '"transformers":"5.6.2","huggingface_hub":"1.12.0"}\n'
                ),
                stderr="",
            )
        if cmd[:3] == [str(vllm_python), "-m", "pip"]:
            return mocker.Mock(returncode=0, stdout="No broken requirements found.\n", stderr="")
        raise AssertionError(cmd)

    mocker.patch("vserve.runtime.subprocess.run", side_effect=fake_run)

    info = collect_vllm_runtime_info(cfg)

    assert info.vllm_version == "0.20.0"
    assert info.torch_version == "2.11.0+cu130"
    assert info.transformers_version == "5.6.2"
    assert info.pip_check_ok is True


def test_upgrade_vllm_stable_force_reinstalls_pinned_stable(mocker, tmp_path):
    vllm_python = tmp_path / "venv" / "bin" / "python"
    vllm_python.parent.mkdir(parents=True)
    vllm_python.touch()
    cfg = mocker.Mock(vllm_python=vllm_python)
    run = mocker.patch("vserve.runtime.subprocess.run", return_value=mocker.Mock(returncode=0, stdout="", stderr=""))

    upgrade_vllm_stable(cfg)

    run.assert_called_once()
    cmd = run.call_args.args[0]
    assert cmd[:4] == [str(vllm_python), "-m", "pip", "install"]
    assert "--force-reinstall" in cmd
    assert "vllm==0.20.0" in cmd


def test_build_tuning_fingerprint_includes_template_detector_runtime_and_files(tmp_path):
    from vserve.models import ModelInfo

    model_dir = tmp_path / "models" / "provider" / "Model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.safetensors").write_bytes(b"weights-v1")
    (model_dir / "tokenizer_config.json").write_text('{"chat_template": "hello {{ tools }}"}')
    model = ModelInfo(
        path=model_dir,
        provider="provider",
        model_name="Model",
        architecture="TestLM",
        model_type="test",
        quant_method="fp8",
        max_position_embeddings=4096,
        is_moe=False,
        model_size_gb=1.0,
    )
    gpu = type("GPU", (), {
        "name": "GPU",
        "driver": "590",
        "cuda": "13.1",
        "vram_total_gb": 48.0,
    })()
    runtime = RuntimeInfo(
        backend="vllm",
        executable=Path("/opt/vllm/venv/bin/vllm"),
        python=Path("/opt/vllm/venv/bin/python"),
        vllm_version="0.20.0",
        torch_version="2.11.0+cu130",
        torch_cuda="13.0",
        transformers_version="5.6.2",
    )

    fp1 = build_tuning_fingerprint(
        model_info=model,
        gpu=gpu,
        backend="vllm",
        gpu_mem_util=0.91,
        runtime_info=runtime,
    )
    (model_dir / "tokenizer_config.json").write_text('{"chat_template": "changed {{ tools }}"}')
    fp2 = build_tuning_fingerprint(
        model_info=model,
        gpu=gpu,
        backend="vllm",
        gpu_mem_util=0.91,
        runtime_info=runtime,
    )

    assert fp1["detector_schema_version"] == DETECTOR_SCHEMA_VERSION
    assert fp1["tokenizer_template_hash"] != fp2["tokenizer_template_hash"]
    assert fp1["vllm_version"] == "0.20.0"
    assert fp1["model_file_identity"][0]["path"] == "model.safetensors"


def test_build_tuning_fingerprint_includes_gguf_metadata_hash(tmp_path):
    from vserve.models import ModelInfo

    model_dir = tmp_path / "models" / "provider" / "Model-GGUF"
    model_dir.mkdir(parents=True)
    gguf = model_dir / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF-metadata-v1" + b"\0" * 128)
    model = ModelInfo(
        path=model_dir,
        provider="provider",
        model_name="Model-GGUF",
        architecture="gguf",
        model_type="gguf",
        quant_method=None,
        max_position_embeddings=0,
        is_moe=False,
        model_size_gb=1.0,
        is_gguf=True,
    )
    gpu = type("GPU", (), {
        "name": "GPU",
        "driver": "590",
        "cuda": "13.1",
        "vram_total_gb": 48.0,
    })()

    fp = build_tuning_fingerprint(
        model_info=model,
        gpu=gpu,
        backend="llamacpp",
        gpu_mem_util=0.91,
    )

    assert fp["gguf_metadata_hash"]


def test_build_tuning_fingerprint_includes_llamacpp_runtime_identity(tmp_path):
    from vserve.models import ModelInfo

    model_dir = tmp_path / "models" / "provider" / "Model-GGUF"
    model_dir.mkdir(parents=True)
    (model_dir / "model-Q4_K_M.gguf").write_bytes(b"GGUF")
    model = ModelInfo(
        path=model_dir,
        provider="provider",
        model_name="Model-GGUF",
        architecture="gguf",
        model_type="gguf",
        quant_method=None,
        max_position_embeddings=0,
        is_moe=False,
        model_size_gb=1.0,
        is_gguf=True,
    )
    gpu = type("GPU", (), {
        "name": "GPU",
        "driver": "590",
        "cuda": "13.1",
        "vram_total_gb": 48.0,
    })()

    fp = build_tuning_fingerprint(
        model_info=model,
        gpu=gpu,
        backend="llamacpp",
        gpu_mem_util=0.91,
        runtime_info={
            "backend": "llamacpp",
            "executable": "/opt/llama-cpp/bin/llama-server",
            "llama_server_version": "llama-server 2026",
        },
    )

    assert fp["llama_server_version"] == "llama-server 2026"
    assert fp["runtime_executable"] == "/opt/llama-cpp/bin/llama-server"
