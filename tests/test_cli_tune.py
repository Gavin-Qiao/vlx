from typer.testing import CliRunner
from vx.cli import app

runner = CliRunner()


def test_tune_no_model_found(mocker):
    mocker.patch("vx.cli.scan_models", return_value=[])
    result = runner.invoke(app, ["tune", "nonexistent", "interactive"])
    assert result.exit_code == 1
    assert "No models found" in result.output


def test_tune_dry_run_shows_commands(mocker, fake_model_dir):
    from vx.models import detect_model
    m = detect_model(fake_model_dir)

    mocker.patch("vx.cli.scan_models", return_value=[m])
    mocker.patch("vx.cli.fuzzy_match", return_value=[m])
    mocker.patch("vx.cli.read_limits", return_value=None)

    # Mock GPU
    mocker.patch("vx.gpu.get_gpu_info", return_value=type("GPU", (), {
        "name": "Test GPU", "vram_total_gb": 48.0, "cuda": "13.1",
        "vram_total_mb": 49152, "vram_used_mb": 0, "vram_free_mb": 49152,
        "driver": "590", "vram_used_gb": 0.0,
    })())

    result = runner.invoke(app, ["tune", "TestModel", "interactive", "--dry-run"])
    assert result.exit_code == 0
    assert "DRY RUN" in result.output
    assert "bench sweep" in result.output


def test_tune_dry_run_with_existing_limits(mocker, fake_model_dir, fake_limits):
    """When limits exist, probe is skipped — only benchmark job runs."""
    from vx.models import detect_model
    m = detect_model(fake_model_dir)

    mocker.patch("vx.cli.scan_models", return_value=[m])
    mocker.patch("vx.cli.fuzzy_match", return_value=[m])
    mocker.patch("vx.cli.read_limits", return_value=fake_limits)

    mocker.patch("vx.gpu.get_gpu_info", return_value=type("GPU", (), {
        "name": "Test GPU", "vram_total_gb": 48.0, "cuda": "13.1",
        "vram_total_mb": 49152, "vram_used_mb": 0, "vram_free_mb": 49152,
        "driver": "590", "vram_used_gb": 0.0,
    })())

    result = runner.invoke(app, ["tune", "TestModel", "interactive", "--dry-run"])
    assert result.exit_code == 0
    assert "DRY RUN" in result.output
    # No probe job when limits already exist
    assert "probe" not in result.output.lower() or "→ probe" not in result.output


def test_tune_dry_run_all_profiles(mocker, fake_model_dir, fake_limits):
    from vx.models import detect_model
    m = detect_model(fake_model_dir)

    mocker.patch("vx.cli.scan_models", return_value=[m])
    mocker.patch("vx.cli.fuzzy_match", return_value=[m])
    mocker.patch("vx.cli.read_limits", return_value=fake_limits)

    mocker.patch("vx.gpu.get_gpu_info", return_value=type("GPU", (), {
        "name": "Test GPU", "vram_total_gb": 48.0, "cuda": "13.1",
        "vram_total_mb": 49152, "vram_used_mb": 0, "vram_free_mb": 49152,
        "driver": "590", "vram_used_gb": 0.0,
    })())

    result = runner.invoke(app, ["tune", "TestModel", "all", "--dry-run"])
    assert result.exit_code == 0
    # Should show all 4 profiles
    assert "interactive" in result.output
    assert "batch" in result.output
    assert "agentic" in result.output
    assert "creative" in result.output
