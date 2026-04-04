"""Tests for vserve tune CLI command — analytical limit calculation."""

from typer.testing import CliRunner
from vserve.cli import app

runner = CliRunner()


def test_tune_no_model_found(mocker):
    mocker.patch("vserve.cli._all_models", return_value=[])
    mocker.patch("vserve.cli.scan_models", return_value=[])
    result = runner.invoke(app, ["tune", "nonexistent"])
    assert result.exit_code == 1
    assert "No model" in result.output


def _mock_vllm_backend(mocker):
    """Create a mock vLLM backend for tune tests."""
    from unittest.mock import Mock
    from vserve.probe import calculate_limits
    backend = Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.tune.side_effect = lambda m, gpu, **kw: calculate_limits(
        model_info=m, vram_total_gb=gpu.vram_total_gb, gpu_mem_util=kw.get("gpu_mem_util", 0.90),
    )
    mocker.patch("vserve.backends.get_backend", return_value=backend)
    return backend


def test_tune_calculates_limits(mocker, fake_model_dir):
    from vserve.models import detect_model
    m = detect_model(fake_model_dir)

    _mock_vllm_backend(mocker)
    mocker.patch("vserve.cli._all_models", return_value=[m])
    mocker.patch("vserve.cli.scan_models", return_value=[m])
    mocker.patch("vserve.cli.fuzzy_match", return_value=[m])
    mocker.patch("vserve.gpu.get_gpu_info", return_value=type("GPU", (), {
        "name": "Test GPU", "vram_total_gb": 48.0, "cuda": "13.1",
        "vram_total_mb": 49152, "vram_used_mb": 0, "vram_free_mb": 49152,
        "driver": "590", "vram_used_gb": 0.0,
    })())
    mocker.patch("vserve.config.write_limits")

    result = runner.invoke(app, ["tune", "TestModel"])
    if result.exit_code != 0:
        print(f"TUNE OUTPUT:\n{result.output}")
        if result.exception:
            import traceback
            traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)
    assert result.exit_code == 0
    assert "slots" in result.output


def test_tune_all_models(mocker, fake_model_dir):
    from vserve.models import detect_model
    m = detect_model(fake_model_dir)

    _mock_vllm_backend(mocker)
    mocker.patch("vserve.cli._all_models", return_value=[m])
    mocker.patch("vserve.cli.scan_models", return_value=[m])
    mocker.patch("vserve.gpu.get_gpu_info", return_value=type("GPU", (), {
        "name": "Test GPU", "vram_total_gb": 48.0, "cuda": "13.1",
        "vram_total_mb": 49152, "vram_used_mb": 0, "vram_free_mb": 49152,
        "driver": "590", "vram_used_gb": 0.0,
    })())
    mocker.patch("vserve.config.write_limits")

    result = runner.invoke(app, ["tune", "--all"])
    if result.exit_code != 0:
        print(f"TUNE ALL OUTPUT:\n{result.output}")
        if result.exception:
            import traceback
            traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)
    assert result.exit_code == 0


def test_tune_cached_shows_table(mocker, fake_model_dir, fake_limits):
    from vserve.models import detect_model
    m = detect_model(fake_model_dir)

    mocker.patch("vserve.cli._all_models", return_value=[m])
    mocker.patch("vserve.cli.scan_models", return_value=[m])
    mocker.patch("vserve.cli.fuzzy_match", return_value=[m])
    mocker.patch("vserve.cli.read_limits", return_value=fake_limits)
    mocker.patch("vserve.gpu.get_gpu_info", return_value=type("GPU", (), {
        "name": "Test GPU", "vram_total_gb": 48.0, "cuda": "13.1",
        "vram_total_mb": 49152, "vram_used_mb": 0, "vram_free_mb": 49152,
        "driver": "590", "vram_used_gb": 0.0,
    })())
    # Mock config to not have custom gpu settings (so gpu_util matches fake_limits)
    mock_cfg = mocker.MagicMock()
    mock_cfg.gpu_memory_utilization = fake_limits["gpu_memory_utilization"]
    mock_cfg.gpu_overhead_gb = None
    mocker.patch("vserve.config.cfg", return_value=mock_cfg)

    result = runner.invoke(app, ["tune", "TestModel"])
    assert result.exit_code == 0
    assert "cached" in result.output


def test_tune_recalc_ignores_cache(mocker, fake_model_dir, fake_limits):
    from vserve.models import detect_model
    m = detect_model(fake_model_dir)

    _mock_vllm_backend(mocker)
    mocker.patch("vserve.cli._all_models", return_value=[m])
    mocker.patch("vserve.cli.scan_models", return_value=[m])
    mocker.patch("vserve.cli.fuzzy_match", return_value=[m])
    mocker.patch("vserve.cli.read_limits", return_value=fake_limits)
    mocker.patch("vserve.config.write_limits")
    mocker.patch("vserve.gpu.get_gpu_info", return_value=type("GPU", (), {
        "name": "Test GPU", "vram_total_gb": 48.0, "cuda": "13.1",
        "vram_total_mb": 49152, "vram_used_mb": 0, "vram_free_mb": 49152,
        "driver": "590", "vram_used_gb": 0.0,
    })())

    result = runner.invoke(app, ["tune", "TestModel", "--recalc"])
    assert result.exit_code == 0
    assert "cached" not in result.output
    assert "slots" in result.output


def test_tune_gpu_util_out_of_range(mocker, fake_model_dir):
    """gpu_util outside 0.5-0.99 should fail with a clear error."""
    result = runner.invoke(app, ["tune", "TestModel", "--gpu-util", "0.3"])
    assert result.exit_code == 1
    assert "--gpu-util must be between 0.5 and 0.99" in result.output

    result = runner.invoke(app, ["tune", "TestModel", "--gpu-util", "1.0"])
    assert result.exit_code == 1
    assert "--gpu-util must be between 0.5 and 0.99" in result.output


def test_tune_missing_arch_fields(mocker, tmp_path):
    """Model without architecture fields shows clear error, doesn't crash."""
    import json
    from vserve.models import detect_model
    model_dir = tmp_path / "models" / "test" / "Minimal"
    model_dir.mkdir(parents=True)
    config = {"architectures": ["MinimalLM"], "model_type": "test"}
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "model.safetensors").write_bytes(b"\0" * 512)
    m = detect_model(model_dir)

    mocker.patch("vserve.cli._all_models", return_value=[m])
    mocker.patch("vserve.cli.scan_models", return_value=[m])
    mocker.patch("vserve.cli.fuzzy_match", return_value=[m])
    mocker.patch("vserve.cli.read_limits", return_value=None)
    mocker.patch("vserve.gpu.get_gpu_info", return_value=type("GPU", (), {
        "name": "Test GPU", "vram_total_gb": 48.0, "cuda": "13.1",
        "vram_total_mb": 49152, "vram_used_mb": 0, "vram_free_mb": 49152,
        "driver": "590", "vram_used_gb": 0.0,
    })())

    result = runner.invoke(app, ["tune", "Minimal"])
    assert result.exit_code == 0  # continues past bad model
    assert "Missing architecture" in result.output
