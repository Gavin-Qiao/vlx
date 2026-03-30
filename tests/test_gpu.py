import pytest

from vx.gpu import get_gpu_info, compute_gpu_memory_utilization, get_fan_speed, set_fan_speed


def test_parse_gpu_info(mocker):
    mocker.patch(
        "vx.gpu._run_nvidia_smi",
        return_value="NVIDIA RTX PRO 5000 Blackwell, 48935, 0, 1024, 590.48.01",
    )
    mocker.patch("vx.gpu._get_cuda_version", return_value="13.1")
    info = get_gpu_info()
    assert info.name == "NVIDIA RTX PRO 5000 Blackwell"
    assert info.vram_total_mb == 48935
    assert abs(info.vram_total_gb - 47.8) < 0.1
    assert info.driver == "590.48.01"
    assert info.cuda == "13.1"


def test_compute_gpu_memory_utilization():
    util = compute_gpu_memory_utilization(vram_total_gb=47.8, overhead_gb=2.0)
    assert abs(util - 0.958) < 0.001


def test_compute_gpu_memory_utilization_small_gpu():
    util = compute_gpu_memory_utilization(vram_total_gb=8.0, overhead_gb=2.0)
    assert abs(util - 0.75) < 0.001


def test_get_fan_speed(mocker):
    mocker.patch(
        "vx.gpu.subprocess.run",
        return_value=mocker.Mock(stdout="80\n", returncode=0),
    )
    assert get_fan_speed() == 80


def test_set_fan_speed_valid(mocker):
    mock_popen = mocker.Mock()
    mock_popen.wait.return_value = None
    mocker.patch("vx.gpu.subprocess.Popen", return_value=mock_popen)
    mock_run = mocker.patch("vx.gpu.subprocess.run", return_value=mocker.Mock(returncode=0))
    mocker.patch("vx.gpu.time.sleep")

    set_fan_speed(80)

    calls = mock_run.call_args_list
    assert any("[gpu:0]/GPUFanControlState=1" in str(c) for c in calls)
    assert any("[fan:0]/GPUTargetFanSpeed=80" in str(c) for c in calls)
    mock_popen.terminate.assert_called_once()


def test_set_fan_speed_rejects_out_of_range():
    with pytest.raises(ValueError, match="30-100"):
        set_fan_speed(10)
    with pytest.raises(ValueError, match="30-100"):
        set_fan_speed(101)
