from pathlib import Path
from typer.testing import CliRunner
from vserve.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "LLM inference manager" in result.output


def test_list_command_exists():
    result = runner.invoke(app, ["list", "--help"])
    assert result.exit_code == 0
    assert "List downloaded models" in result.output


def test_ls_alias_works():
    result = runner.invoke(app, ["ls", "--help"])
    assert result.exit_code == 0


def test_add_command_exists():
    result = runner.invoke(app, ["add", "--help"])
    assert result.exit_code == 0


def test_rm_command_exists():
    result = runner.invoke(app, ["rm", "--help"])
    assert result.exit_code == 0
    assert "Remove a downloaded model" in result.output


def test_tune_command_exists():
    result = runner.invoke(app, ["tune", "--help"])
    assert result.exit_code == 0


def test_run_command_exists():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0


def test_stop_command_exists():
    result = runner.invoke(app, ["stop", "--help"])
    assert result.exit_code == 0



def test_list_runs(mocker):
    from vserve.models import ModelInfo

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
    mocker.patch("vserve.cli.scan_models", return_value=[fake])
    mocker.patch("vserve.cli.read_limits", return_value=None)

    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "Model-7B" in result.output


def test_add_shows_variant_picker(mocker):
    """add with full repo ID shows variant picker."""
    from vserve.variants import Variant

    mock_api_cls = mocker.patch("huggingface_hub.HfApi")
    mock_api = mock_api_cls.return_value
    mock_api.repo_exists.return_value = True

    mocker.patch("vserve.variants.fetch_repo_variants", return_value=(
        [
            Variant(label="mxfp4", files={
                "model.safetensors.index.json": 50_000,
                "model-00001-of-00001.safetensors": 14_000_000_000,
            }),
        ],
        {"config.json": 1200, "tokenizer.json": 4_000_000},
    ))
    mocker.patch("huggingface_hub.snapshot_download")
    mocker.patch("vserve.cli.VserveLock")
    mocker.patch("vserve.models.detect_model", side_effect=FileNotFoundError)

    result = runner.invoke(app, ["add", "test/model"], input="1\ny\n")

    assert "mxfp4" in result.output
    assert "13.0 GB" in result.output


def test_add_nonexistent_repo_falls_to_search(mocker):
    """add with non-existent repo ID falls through to search."""
    mock_api_cls = mocker.patch("huggingface_hub.HfApi")
    mock_api = mock_api_cls.return_value
    mock_api.repo_exists.return_value = False
    mock_api.list_models.return_value = []

    result = runner.invoke(app, ["add", "nonexistent/model"], input="\n")

    assert "Searching" in result.output


def test_rm_deletes(mocker, tmp_path):
    """rm deletes the model dir and cached limits."""
    from vserve.models import ModelInfo

    model_dir = tmp_path / "models" / "test" / "Model-7B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")

    fake = ModelInfo(
        path=model_dir,
        provider="test",
        model_name="Model-7B",
        architecture="TestLM",
        model_type="test",
        quant_method="fp8",
        max_position_embeddings=32768,
        is_moe=False,
        model_size_gb=7.2,
    )
    mocker.patch("vserve.cli._all_models", return_value=[fake])
    mocker.patch("vserve.cli.fuzzy_match", return_value=[fake])
    mocker.patch("vserve.cli.limits_path", return_value=tmp_path / "nonexistent.yaml")
    mocker.patch("vserve.cli.profile_path", return_value=tmp_path / "nonexistent_profile.yaml")

    result = runner.invoke(app, ["rm", "Model-7B", "--force"])
    assert result.exit_code == 0
    assert "Deleted" in result.output
    assert not model_dir.exists()


def test_rm_cancelled(mocker, tmp_path):
    """rm without --force prompts and can be cancelled."""
    from vserve.models import ModelInfo

    model_dir = tmp_path / "models" / "test" / "Model-7B"
    model_dir.mkdir(parents=True)

    fake = ModelInfo(
        path=model_dir,
        provider="test",
        model_name="Model-7B",
        architecture="TestLM",
        model_type="test",
        quant_method="fp8",
        max_position_embeddings=32768,
        is_moe=False,
        model_size_gb=7.2,
    )
    mocker.patch("vserve.cli._all_models", return_value=[fake])
    mocker.patch("vserve.cli.fuzzy_match", return_value=[fake])

    result = runner.invoke(app, ["rm", "Model-7B"], input="n\n")
    assert model_dir.exists()
    assert "Cancelled" in result.output


def test_remove_alias_works():
    """remove is a hidden alias for rm."""
    result = runner.invoke(app, ["remove", "--help"])
    assert result.exit_code == 0
    assert "alias for rm" in result.output


def test_status_command_exists():
    result = runner.invoke(app, ["status", "--help"])
    assert result.exit_code == 0


def test_doctor_command_exists():
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0


def test_init_command_exists():
    result = runner.invoke(app, ["init", "--help"])
    assert result.exit_code == 0


# --- Old command names must not work ---

def test_old_start_command_removed():
    result = runner.invoke(app, ["start", "--help"])
    assert result.exit_code != 0


def test_old_models_command_removed():
    result = runner.invoke(app, ["models", "--help"])
    assert result.exit_code != 0


def test_old_download_command_removed():
    result = runner.invoke(app, ["download", "--help"])
    assert result.exit_code != 0


# --- Dashboard shows new command names ---

def test_dashboard_shows_new_commands(mocker):
    mocker.patch("vserve.cli._all_models", return_value=[])
    mocker.patch("vserve.cli.read_limits", return_value=None)
    mocker.patch("vserve.cli.CONFIG_FILE", new_callable=lambda: type("P", (), {"exists": lambda self: True})())
    result = runner.invoke(app, [])
    assert "run [model]" in result.output or "run" in result.output
    assert "list" in result.output
    assert "add" in result.output
    assert "rm" in result.output


# --- list command ---

def test_list_detail_view(mocker):
    """list <model> shows detail view."""
    from vserve.models import ModelInfo

    fake = ModelInfo(
        path=Path("/opt/vllm/models/test/Model-7B"),
        provider="test", model_name="Model-7B",
        architecture="TestLM", model_type="test", quant_method="fp8",
        max_position_embeddings=32768, is_moe=False, model_size_gb=7.2,
    )
    mocker.patch("vserve.cli._all_models", return_value=[fake])
    mocker.patch("vserve.cli.fuzzy_match", return_value=[fake])
    mocker.patch("vserve.cli.read_limits", return_value=None)

    result = runner.invoke(app, ["list", "Model-7B"])
    assert result.exit_code == 0
    assert "Model-7B" in result.output
    assert "Not probed" in result.output or "not tuned" in result.output.lower() or "TestLM" in result.output


def test_list_empty(mocker):
    mocker.patch("vserve.cli.scan_models", return_value=[])
    mocker.patch("vserve.cli._all_models", return_value=[])
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "No models" in result.output


def test_list_with_limits(mocker):
    """list shows tuned status from cached limits."""
    from vserve.models import ModelInfo

    fake = ModelInfo(
        path=Path("/opt/vllm/models/test/Model-7B"),
        provider="test", model_name="Model-7B",
        architecture="TestLM", model_type="test", quant_method="fp8",
        max_position_embeddings=32768, is_moe=False, model_size_gb=7.2,
    )
    mocker.patch("vserve.cli.scan_models", return_value=[fake])
    mocker.patch("vserve.cli.read_limits", return_value={
        "limits": {"4096": {"auto": 64, "fp8": 128}},
        "tool_call_parser": "hermes",
    })
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "Model-7B" in result.output


# --- rm edge cases ---

def test_rm_cleans_empty_parent(mocker, tmp_path):
    """rm removes empty parent provider dir."""
    from vserve.models import ModelInfo

    model_dir = tmp_path / "models" / "orphan-provider" / "Model-7B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")

    fake = ModelInfo(
        path=model_dir, provider="orphan-provider", model_name="Model-7B",
        architecture="TestLM", model_type="test", quant_method=None,
        max_position_embeddings=32768, is_moe=False, model_size_gb=1.0,
    )
    mocker.patch("vserve.cli._all_models", return_value=[fake])
    mocker.patch("vserve.cli.fuzzy_match", return_value=[fake])
    mocker.patch("vserve.cli.limits_path", return_value=tmp_path / "no.yaml")
    mocker.patch("vserve.cli.profile_path", return_value=tmp_path / "no.yaml")

    result = runner.invoke(app, ["rm", "Model-7B", "--force"])
    assert result.exit_code == 0
    assert not model_dir.exists()
    assert not model_dir.parent.exists()  # empty provider dir removed


def test_rm_interactive_picker(mocker, tmp_path):
    """rm without args shows picker."""
    from vserve.models import ModelInfo

    model_dir = tmp_path / "models" / "test" / "Model-7B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}")

    fake = ModelInfo(
        path=model_dir, provider="test", model_name="Model-7B",
        architecture="TestLM", model_type="test", quant_method=None,
        max_position_embeddings=32768, is_moe=False, model_size_gb=1.0,
    )
    mocker.patch("vserve.cli._all_models", return_value=[fake])
    mocker.patch("vserve.cli._pick", return_value=0)
    mocker.patch("vserve.cli.limits_path", return_value=tmp_path / "no.yaml")
    mocker.patch("vserve.cli.profile_path", return_value=tmp_path / "no.yaml")

    result = runner.invoke(app, ["rm", "--force"])
    assert result.exit_code == 0
    assert "Deleted" in result.output


# --- run command ---

def test_run_no_models(mocker):
    mocker.patch("vserve.cli._all_models", return_value=[])
    mocker.patch("vserve.cli.check_session")
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 1
    assert "No models" in result.output


def test_run_backend_override(mocker):
    """run --backend forces specific backend."""
    from vserve.models import ModelInfo

    fake = ModelInfo(
        path=Path("/opt/vllm/models/test/Model-7B"),
        provider="test", model_name="Model-7B",
        architecture="TestLM", model_type="test", quant_method="fp8",
        max_position_embeddings=32768, is_moe=False, model_size_gb=7.2,
    )
    mocker.patch("vserve.cli._all_models", return_value=[fake])
    mocker.patch("vserve.cli.fuzzy_match", return_value=[fake])
    mocker.patch("vserve.cli.check_session")
    mock_get = mocker.patch("vserve.backends.get_backend_by_name", side_effect=KeyError("nope"))

    runner.invoke(app, ["run", "Model-7B", "--backend", "nope"])
    mock_get.assert_called_once_with("nope")
