"""GPU info via nvidia-smi."""

import subprocess
from dataclasses import dataclass


@dataclass
class GpuInfo:
    name: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    driver: str
    cuda: str

    @property
    def vram_total_gb(self) -> float:
        return self.vram_total_mb / 1024

    @property
    def vram_used_gb(self) -> float:
        return self.vram_used_mb / 1024


def _run_nvidia_smi() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,utilization.gpu,memory.used,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _get_cuda_version() -> str:
    result = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in result.stdout.splitlines():
        if "CUDA Version:" in line:
            parts = line.split("CUDA Version:")
            if len(parts) > 1:
                return parts[1].strip().split()[0]
    return "unknown"


def get_gpu_info() -> GpuInfo:
    raw = _run_nvidia_smi()
    parts = [p.strip() for p in raw.split(",")]
    name = parts[0]
    vram_total = int(parts[1])
    vram_used = int(parts[3])
    driver = parts[4]
    cuda = _get_cuda_version()
    return GpuInfo(
        name=name,
        vram_total_mb=vram_total,
        vram_used_mb=vram_used,
        vram_free_mb=vram_total - vram_used,
        driver=driver,
        cuda=cuda,
    )


def compute_gpu_memory_utilization(
    vram_total_gb: float, overhead_gb: float = 2.0
) -> float:
    return (vram_total_gb - overhead_gb) / vram_total_gb
