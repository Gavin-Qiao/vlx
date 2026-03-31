from pathlib import Path
from unittest.mock import Mock

from vserve.serve import start_vllm, stop_vllm, is_vllm_running


def _mock_cfg(mocker, service_name="vllm", active_yaml=None):
    mock = Mock()
    mock.service_name = service_name
    mock.active_yaml = active_yaml or Path("/tmp/active.yaml")
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


def test_systemctl_uses_service_name(mocker):
    _mock_cfg(mocker, service_name="my-vllm")
    mock_run = mocker.patch("vserve.serve.subprocess.run", return_value=Mock(returncode=0, stdout="active", stderr=""))
    is_vllm_running()
    args = mock_run.call_args[0][0]
    assert "my-vllm" in args


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
