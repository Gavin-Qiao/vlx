from pathlib import Path
import click
import pytest
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


def test_stop_failure_preserves_session(mocker):
    backend = mocker.Mock()
    backend.display_name = "vLLM"
    backend.is_running.return_value = True
    backend.stop.side_effect = RuntimeError("stop failed")
    lock = mocker.Mock()

    mocker.patch("vserve.backends._BACKENDS", [backend])
    mocker.patch("vserve.cli._session_or_exit")
    clear_session = mocker.patch("vserve.cli.clear_session")
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)

    result = runner.invoke(app, ["stop"])
    assert result.exit_code == 1
    clear_session.assert_not_called()
    lock.release.assert_called_once()



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


def test_status_handles_unreadable_active_config(mocker, tmp_path):
    active = tmp_path / "active.yaml"
    active.write_text("not: valid: yaml: {{{{")

    backend = mocker.Mock()
    backend.name = "vllm"
    backend.display_name = "vLLM"
    backend.is_running.return_value = True

    mocker.patch("vserve.backends._BACKENDS", [backend])
    mocker.patch("vserve.config.active_yaml_path", return_value=active)

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "active config unreadable" in result.output
    assert str(active) in result.output


def test_doctor_command_exists():
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0


def test_init_command_exists():
    result = runner.invoke(app, ["init", "--help"])
    assert result.exit_code == 0


def test_init_reports_nvml_fan_control_without_coolbits(mocker, tmp_path):
    config_file = tmp_path / "config.yaml"
    vllm_root = tmp_path / "vllm"

    fake_gpu = mocker.Mock(name="GPU")
    fake_gpu.name = "NVIDIA RTX PRO 5000 Blackwell"
    fake_gpu.vram_total_gb = 48
    fake_gpu.driver = "595.58.03"
    fake_gpu.cuda = "13.1"

    mocker.patch("vserve.gpu.get_gpu_info", return_value=fake_gpu)
    mocker.patch("vserve.gpu.get_fan_count", return_value=1)
    mocker.patch("vserve.config._discover_cuda_home", return_value=tmp_path / "cuda")
    mocker.patch("vserve.config._discover_vllm_root", return_value=vllm_root)
    mocker.patch("vserve.config._discover_service", return_value=("vllm", "vllm"))
    mocker.patch("vserve.config._discover_port", return_value=8888)
    mocker.patch("vserve.config._build_config", return_value=mocker.Mock())
    mocker.patch("vserve.config.save_config")
    mocker.patch("vserve.config.reset_config")
    mocker.patch("vserve.config.CONFIG_FILE", config_file)
    mocker.patch("typer.confirm", return_value=False)

    def _which(name: str) -> str | None:
        if name == "nvcc":
            return "/usr/bin/nvcc"
        return None

    mocker.patch("shutil.which", side_effect=_which)
    mocker.patch("subprocess.run", return_value=mocker.Mock(returncode=0, stdout="vLLM 0.1.0", stderr=""))

    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert "Fan control 1 fan(s) exposed via NVML" in result.output
    assert "Coolbits" not in result.output
    assert "Install login banner?" not in result.output


def test_init_llamacpp_only_skips_vllm_prompt_and_banner_install(mocker, tmp_path):
    config_file = tmp_path / "config.yaml"

    fake_gpu = mocker.Mock(name="GPU")
    fake_gpu.name = "NVIDIA RTX PRO 5000 Blackwell"
    fake_gpu.vram_total_gb = 48
    fake_gpu.driver = "595.58.03"
    fake_gpu.cuda = "13.1"

    mocker.patch("vserve.gpu.get_gpu_info", return_value=fake_gpu)
    mocker.patch("vserve.gpu.get_fan_count", return_value=1)
    mocker.patch("vserve.config._discover_cuda_home", return_value=tmp_path / "cuda")
    mocker.patch("vserve.config._discover_vllm_root", return_value=None)
    mocker.patch("vserve.config._discover_service", return_value=("vllm", "vllm"))
    mocker.patch("vserve.config._discover_port", return_value=8888)
    mocker.patch("vserve.config._build_config", return_value=mocker.Mock())
    mocker.patch("vserve.config.save_config")
    mocker.patch("vserve.config.reset_config")
    mocker.patch("vserve.config.CONFIG_FILE", config_file)
    mocker.patch("typer.confirm", return_value=False)
    prompt = mocker.patch("typer.prompt")

    def _which(name: str) -> str | None:
        if name == "nvcc":
            return "/usr/bin/nvcc"
        if name == "llama-server":
            return "/opt/llama-cpp/bin/llama-server"
        return None

    mocker.patch("shutil.which", side_effect=_which)
    mocker.patch("subprocess.run", return_value=mocker.Mock(returncode=0, stdout="ok", stderr=""))

    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0
    prompt.assert_not_called()
    assert "optional — needed for safetensors models" in result.output
    assert "required for vserve run/stop" not in result.output
    assert "Install login banner?" not in result.output


def test_doctor_summary_label_counts_warnings():
    from vserve.cli import _doctor_summary_label

    assert _doctor_summary_label(0, 0) == "All clear"
    assert _doctor_summary_label(1, 0) == "1 warning(s) found"
    assert _doctor_summary_label(2, 1) == "1 issue(s) and 2 warning(s) found"


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


def test_launch_backend_aborts_on_restart_stop_failure(mocker, tmp_path):
    from vserve.cli import _launch_backend

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("port: 8888\nmax-model-len: 4096\n")

    backend = mocker.Mock()
    backend.display_name = "vLLM"
    backend.service_name = "vllm"
    backend.name = "vllm"
    backend.is_running.return_value = True
    backend.stop.side_effect = RuntimeError("stop failed")

    lock = mocker.Mock()
    clear_session = mocker.patch("vserve.cli.clear_session")
    mocker.patch("vserve.cli._lock_or_exit", return_value=lock)
    mocker.patch("vserve.cli._session_or_exit")
    mocker.patch("vserve.config.read_profile_yaml", return_value={"port": 8888, "max-model-len": 4096})
    mocker.patch("typer.confirm", return_value=True)
    mocker.patch("time.sleep")

    with pytest.raises(click.exceptions.Exit):
        _launch_backend(backend, cfg_path, "test-model")

    backend.start.assert_not_called()
    clear_session.assert_not_called()
    lock.release.assert_called_once()
