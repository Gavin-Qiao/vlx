"""Tests for vlx tune CLI command — analytical limit calculation."""

from typer.testing import CliRunner
from vlx.cli import app

runner = CliRunner()


def test_tune_no_model_found(mocker):
    mocker.patch("vlx.cli.scan_models", return_value=[])
    result = runner.invoke(app, ["tune", "nonexistent"])
    assert result.exit_code == 1
    assert "No model" in result.output


def test_tune_calculates_limits(mocker, fake_model_dir):
    from vlx.models import detect_model
    m = detect_model(fake_model_dir)

    mocker.patch("vlx.cli.scan_models", return_value=[m])
    mocker.patch("vlx.cli.fuzzy_match", return_value=[m])
    mocker.patch("vlx.gpu.get_gpu_info", return_value=type("GPU", (), {
        "name": "Test GPU", "vram_total_gb": 48.0, "cuda": "13.1",
        "vram_total_mb": 49152, "vram_used_mb": 0, "vram_free_mb": 49152,
        "driver": "590", "vram_used_gb": 0.0,
    })())
    mocker.patch("vlx.config.write_limits")

    result = runner.invoke(app, ["tune", "TestModel"])
    assert result.exit_code == 0
    assert "slots" in result.output


def test_tune_all_models(mocker, fake_model_dir):
    from vlx.models import detect_model
    m = detect_model(fake_model_dir)

    mocker.patch("vlx.cli.scan_models", return_value=[m])
    mocker.patch("vlx.gpu.get_gpu_info", return_value=type("GPU", (), {
        "name": "Test GPU", "vram_total_gb": 48.0, "cuda": "13.1",
        "vram_total_mb": 49152, "vram_used_mb": 0, "vram_free_mb": 49152,
        "driver": "590", "vram_used_gb": 0.0,
    })())
    mocker.patch("vlx.config.write_limits")

    result = runner.invoke(app, ["tune", "--all"])
    assert result.exit_code == 0


def test_tune_cached_shows_table(mocker, fake_model_dir, fake_limits):
    from vlx.models import detect_model
    m = detect_model(fake_model_dir)

    mocker.patch("vlx.cli.scan_models", return_value=[m])
    mocker.patch("vlx.cli.fuzzy_match", return_value=[m])
    mocker.patch("vlx.cli.read_limits", return_value=fake_limits)
    mocker.patch("vlx.gpu.get_gpu_info", return_value=type("GPU", (), {
        "name": "Test GPU", "vram_total_gb": 48.0, "cuda": "13.1",
        "vram_total_mb": 49152, "vram_used_mb": 0, "vram_free_mb": 49152,
        "driver": "590", "vram_used_gb": 0.0,
    })())

    result = runner.invoke(app, ["tune", "TestModel"])
    assert result.exit_code == 0
    assert "cached" in result.output


def test_tune_recalc_ignores_cache(mocker, fake_model_dir, fake_limits):
    from vlx.models import detect_model
    m = detect_model(fake_model_dir)

    mocker.patch("vlx.cli.scan_models", return_value=[m])
    mocker.patch("vlx.cli.fuzzy_match", return_value=[m])
    mocker.patch("vlx.cli.read_limits", return_value=fake_limits)
    mocker.patch("vlx.config.write_limits")
    mocker.patch("vlx.gpu.get_gpu_info", return_value=type("GPU", (), {
        "name": "Test GPU", "vram_total_gb": 48.0, "cuda": "13.1",
        "vram_total_mb": 49152, "vram_used_mb": 0, "vram_free_mb": 49152,
        "driver": "590", "vram_used_gb": 0.0,
    })())

    result = runner.invoke(app, ["tune", "TestModel", "--recalc"])
    assert result.exit_code == 0
    assert "cached" not in result.output
    assert "slots" in result.output


def test_tune_missing_arch_fields(mocker, tmp_path):
    """Model without architecture fields shows clear error, doesn't crash."""
    import json
    from vlx.models import detect_model
    model_dir = tmp_path / "models" / "test" / "Minimal"
    model_dir.mkdir(parents=True)
    config = {"architectures": ["MinimalLM"], "model_type": "test"}
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "model.safetensors").write_bytes(b"\0" * 512)
    m = detect_model(model_dir)

    mocker.patch("vlx.cli.scan_models", return_value=[m])
    mocker.patch("vlx.cli.fuzzy_match", return_value=[m])
    mocker.patch("vlx.cli.read_limits", return_value=None)
    mocker.patch("vlx.gpu.get_gpu_info", return_value=type("GPU", (), {
        "name": "Test GPU", "vram_total_gb": 48.0, "cuda": "13.1",
        "vram_total_mb": 49152, "vram_used_mb": 0, "vram_free_mb": 49152,
        "driver": "590", "vram_used_gb": 0.0,
    })())

    result = runner.invoke(app, ["tune", "Minimal"])
    assert result.exit_code == 0  # continues past bad model
    assert "Missing architecture" in result.output
