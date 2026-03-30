from pathlib import Path
from typer.testing import CliRunner
from vlx.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "vLLM model manager" in result.output


def test_models_command_exists():
    result = runner.invoke(app, ["models", "--help"])
    assert result.exit_code == 0
    assert "List downloaded models" in result.output


def test_download_command_exists():
    result = runner.invoke(app, ["download", "--help"])
    assert result.exit_code == 0


def test_tune_command_exists():
    result = runner.invoke(app, ["tune", "--help"])
    assert result.exit_code == 0


def test_start_command_exists():
    result = runner.invoke(app, ["start", "--help"])
    assert result.exit_code == 0


def test_stop_command_exists():
    result = runner.invoke(app, ["stop", "--help"])
    assert result.exit_code == 0



def test_models_runs(mocker):
    from vlx.models import ModelInfo

    fake = ModelInfo(
        path=Path("/opt/vllm/models/test/Model-7B"),
        provider="test",
        model_name="Model-7B",
        architecture="TestLM",
        model_type="test",
        quant_method="fp8",
        max_position_embeddings=32768,
        is_moe=False,
        model_size_gb=7.2,
    )
    mocker.patch("vlx.cli.scan_models", return_value=[fake])
    mocker.patch("vlx.cli.read_limits", return_value=None)

    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "Model-7B" in result.output
