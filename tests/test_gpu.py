import pytest
from unittest.mock import MagicMock, patch

from vserve.gpu import get_gpu_info, compute_gpu_memory_utilization, get_fan_speed, set_fan_speed, restore_fan_auto


def test_parse_gpu_info(mocker):
    mocker.patch(
        "vserve.gpu._run_nvidia_smi",
        return_value="NVIDIA RTX PRO 5000 Blackwell, 48935, 0, 1024, 590.48.01",
    )
    mocker.patch("vserve.gpu._get_cuda_version", return_value="13.1")
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


def test_compute_gpu_memory_utilization_zero_vram():
    assert compute_gpu_memory_utilization(0.0) == 0.90
    assert compute_gpu_memory_utilization(-1.0) == 0.90


def _mock_pynvml():
    """Create a mock pynvml module for testing."""
    mock = MagicMock()
    mock.NVML_TEMPERATURE_GPU = 0
    handle = MagicMock()
    mock.nvmlDeviceGetHandleByIndex.return_value = handle
    return mock, handle


def test_get_fan_speed():
    mock_nvml, handle = _mock_pynvml()
    mock_nvml.nvmlDeviceGetFanSpeed_v2.return_value = 80
    with patch.dict("sys.modules", {"pynvml": mock_nvml}):
        speed = get_fan_speed()
    assert speed == 80
    mock_nvml.nvmlInit.assert_called_once()
    mock_nvml.nvmlShutdown.assert_called_once()


def test_set_fan_speed_valid():
    mock_nvml, handle = _mock_pynvml()
    with patch.dict("sys.modules", {"pynvml": mock_nvml}):
        set_fan_speed(80)
    mock_nvml.nvmlDeviceSetFanSpeed_v2.assert_called_once_with(handle, 0, 80)


def test_set_fan_speed_rejects_out_of_range():
    with pytest.raises(ValueError, match="30-100"):
        set_fan_speed(10)
    with pytest.raises(ValueError, match="30-100"):
        set_fan_speed(101)


def test_restore_fan_auto():
    mock_nvml, handle = _mock_pynvml()
    mock_nvml.nvmlDeviceGetNumFans.return_value = 1
    with patch.dict("sys.modules", {"pynvml": mock_nvml}):
        restore_fan_auto()
    mock_nvml.nvmlDeviceSetFanControlPolicy.assert_called_once_with(handle, 0, 0)
