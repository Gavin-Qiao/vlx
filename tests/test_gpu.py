from vx.gpu import get_gpu_info, compute_gpu_memory_utilization


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
