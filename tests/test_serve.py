from vx.serve import start_vllm, stop_vllm, is_vllm_running


def test_is_vllm_running_active(mocker):
    mocker.patch("vx.serve._systemctl", return_value=(True, "active"))
    assert is_vllm_running() is True


def test_is_vllm_running_inactive(mocker):
    mocker.patch("vx.serve._systemctl", return_value=(True, "inactive"))
    assert is_vllm_running() is False


def test_stop_vllm(mocker):
    mock_run = mocker.patch("vx.serve._systemctl", return_value=(True, ""))
    stop_vllm()
    mock_run.assert_called_once_with("stop")


def test_start_vllm_with_config(mocker, tmp_path):
    config_file = tmp_path / "test.yaml"
    config_file.write_text("model: /opt/vllm/models/test\nport: 8888\n")

    mock_symlink = mocker.patch("vx.serve._update_active_symlink")
    mock_systemctl = mocker.patch("vx.serve._systemctl", return_value=(True, ""))

    start_vllm(config_path=config_file)
    mock_symlink.assert_called_once_with(config_file)
    mock_systemctl.assert_called_with("start")
