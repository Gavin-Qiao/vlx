"""GPU info via nvidia-smi."""

import subprocess
import time
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
        timeout=10,
    )
    return result.stdout.strip()


def _get_cuda_version() -> str:
    result = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
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


def get_fan_speed() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=fan.speed", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return int(result.stdout.strip())


def set_fan_speed(percent: int) -> None:
    """Set GPU fan to a fixed percentage (30-100) via nvidia-settings + Xvfb."""
    if not 30 <= percent <= 100:
        raise ValueError(f"Fan speed must be 30-100, got {percent}")

    import os

    xvfb = subprocess.Popen(
        ["sudo", "Xvfb", ":98", "-screen", "0", "1024x768x24", "-nolisten", "tcp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)

    env = {**os.environ, "DISPLAY": ":98"}
    try:
        subprocess.run(
            ["sudo", "nvidia-settings", "-a", "[gpu:0]/GPUFanControlState=1"],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            timeout=10,
        )
        subprocess.run(
            ["sudo", "nvidia-settings", "-a", f"[fan:0]/GPUTargetFanSpeed={percent}"],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            timeout=10,
        )
    finally:
        xvfb.terminate()
        try:
            xvfb.wait(timeout=5)
        except subprocess.TimeoutExpired:
            xvfb.kill()
            try:
                xvfb.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
