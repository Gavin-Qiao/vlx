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


def test_download_shows_variant_picker(mocker):
    """Download with full repo ID shows variant picker."""
    from vlx.variants import Variant

    mock_api_cls = mocker.patch("huggingface_hub.HfApi")
    mock_api = mock_api_cls.return_value
    mock_api.repo_exists.return_value = True

    mocker.patch("vlx.variants.fetch_repo_variants", return_value=(
        [
            Variant(label="mxfp4", files={
                "model.safetensors.index.json": 50_000,
                "model-00001-of-00001.safetensors": 14_000_000_000,
            }),
        ],
        {"config.json": 1200, "tokenizer.json": 4_000_000},
    ))
    mocker.patch("vlx.cli._has_gum", return_value=False)
    mocker.patch("huggingface_hub.snapshot_download")
    mocker.patch("vlx.cli.VlxLock")
    mocker.patch("vlx.models.detect_model", side_effect=FileNotFoundError)

    result = runner.invoke(app, ["download", "test/model"], input="1\ny\n")

    assert "mxfp4" in result.output
    assert "13.0 GB" in result.output


def test_download_nonexistent_repo_falls_to_search(mocker):
    """Download with non-existent repo ID falls through to search."""
    mock_api_cls = mocker.patch("huggingface_hub.HfApi")
    mock_api = mock_api_cls.return_value
    mock_api.repo_exists.return_value = False
    mock_api.list_models.return_value = []

    mocker.patch("vlx.cli._has_gum", return_value=False)

    result = runner.invoke(app, ["download", "nonexistent/model"], input="\n")

    assert "Searching" in result.output
