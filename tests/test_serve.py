from pathlib import Path
from unittest.mock import Mock

from vserve.serve import start_vllm, stop_vllm, is_vllm_running


def _mock_cfg(mocker, service_name="vllm", active_yaml=None):
    mock = Mock()
    mock.service_name = service_name
    mock.active_yaml = active_yaml or Path("/tmp/active.yaml")
    mock.run_dir = mock.active_yaml.parent / "run"
    mocker.patch("vserve.serve.cfg", return_value=mock)
    return mock


def test_is_vllm_running_active(mocker):
    _mock_cfg(mocker)
    mocker.patch("vserve.serve._systemctl", return_value=(True, "active", ""))
    assert is_vllm_running() is True


def test_is_vllm_running_inactive(mocker):
    _mock_cfg(mocker)
    mocker.patch("vserve.serve._systemctl", return_value=(True, "inactive", ""))
    assert is_vllm_running() is False


def test_is_vllm_running_activating_is_uncertain(mocker):
    import pytest

    _mock_cfg(mocker)
    mocker.patch("vserve.serve._systemctl", return_value=(False, "activating", ""))
    with pytest.raises(RuntimeError, match="is transitional"):
        is_vllm_running()


def test_stop_vllm(mocker):
    _mock_cfg(mocker)
    mock_run = mocker.patch("vserve.serve._systemctl", return_value=(True, "", ""))
    stop_vllm()
    mock_run.assert_called_once_with("stop")


def test_start_vllm_with_config(mocker, tmp_path):
    config_file = tmp_path / "test.yaml"
    config_file.write_text("model: /opt/vllm/models/test\nport: 8888\n")

    active = tmp_path / "active.yaml"
    _mock_cfg(mocker, active_yaml=active)
    mock_systemctl = mocker.patch("vserve.serve._systemctl", return_value=(True, "", ""))

    start_vllm(config_path=config_file)
    assert active.is_symlink()
    mock_systemctl.assert_called_with("start")


def test_start_vllm_writes_active_manifest(mocker, tmp_path):
    config_file = tmp_path / "test.yaml"
    config_file.write_text("model: /opt/vllm/models/test\nport: 8888\n")

    active = tmp_path / "active.yaml"
    mock_c = _mock_cfg(mocker, active_yaml=active)
    mocker.patch("vserve.serve._systemctl", return_value=(True, "", ""))

    start_vllm(config_path=config_file)

    from vserve.config import read_active_manifest

    manifest = read_active_manifest(mock_c.run_dir / "active-manifest.json")
    assert manifest is not None
    assert manifest["backend"] == "vllm"
    assert manifest["service_name"] == "vllm"
    assert manifest["config_path"] == str(config_file.resolve())
    assert manifest["status"] == "starting"


def test_start_vllm_rolls_back_active_link_on_systemctl_failure(mocker, tmp_path):
    previous_config = tmp_path / "previous.yaml"
    previous_config.write_text("model: previous\n")
    next_config = tmp_path / "next.yaml"
    next_config.write_text("model: next\n")
    active = tmp_path / "active.yaml"
    active.symlink_to(previous_config)

    mock_c = _mock_cfg(mocker, active_yaml=active)
    mocker.patch("vserve.serve._systemctl", return_value=(False, "", "Unit not found"))

    import pytest

    with pytest.raises(RuntimeError, match="systemctl start failed"):
        start_vllm(next_config)

    assert active.resolve() == previous_config.resolve()

    from vserve.config import read_active_manifest

    manifest = read_active_manifest(mock_c.run_dir / "active-manifest.json")
    assert manifest is not None
    assert manifest["status"] == "failed"
    assert "Unit not found" in manifest["error"]


def test_systemctl_uses_service_name(mocker):
    _mock_cfg(mocker, service_name="my-vllm")
    mock_run = mocker.patch("vserve.serve.subprocess.run", return_value=Mock(returncode=0, stdout="active", stderr=""))
    is_vllm_running()
    args = mock_run.call_args[0][0]
    assert "my-vllm" in args


def test_systemctl_uses_timeout(mocker):
    _mock_cfg(mocker)
    mock_run = mocker.patch("vserve.serve.subprocess.run", return_value=Mock(returncode=0, stdout="active", stderr=""))
    is_vllm_running()
    assert mock_run.call_args.kwargs["timeout"] == 5


def test_is_vllm_running_probe_error_is_false(mocker):
    import pytest

    _mock_cfg(mocker)
    mocker.patch("vserve.serve._systemctl", return_value=(False, "", "dbus error"))
    with pytest.raises(RuntimeError, match="systemctl is-active"):
        is_vllm_running()


def test_is_vllm_running_missing_unit_is_false(mocker):
    _mock_cfg(mocker)
    mocker.patch("vserve.serve._systemctl", return_value=(False, "", "Unit vllm.service could not be found."))
    assert is_vllm_running() is False


def test_systemctl_timeout_returns_error(mocker):
    _mock_cfg(mocker)
    mocker.patch("vserve.serve.subprocess.run", side_effect=__import__("subprocess").TimeoutExpired(cmd="systemctl", timeout=5))
    ok, out, err = __import__("vserve.serve", fromlist=["_systemctl"])._systemctl("start", timeout=5)
    assert ok is False
    assert out == ""
    assert "timed out" in err


def test_start_vllm_failure(mocker, tmp_path):
    import pytest

    config_file = tmp_path / "config.yaml"
    config_file.write_text("model: test\n")
    active = tmp_path / "active.yaml"
    _mock_cfg(mocker, active_yaml=active)
    mocker.patch("vserve.serve._systemctl", return_value=(False, "", "Unit not found"))
    with pytest.raises(RuntimeError, match="systemctl start failed"):
        start_vllm(config_file)


def test_stop_vllm_failure(mocker):
    import pytest

    _mock_cfg(mocker)
    mocker.patch("vserve.serve._systemctl", return_value=(False, "", "Failed to stop"))
    with pytest.raises(RuntimeError, match="systemctl stop failed"):
        stop_vllm()
