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
    vram_total_gb: float, overhead_gb: float = 3.5
) -> float:
    """Reserve overhead for CUDA context, activations, CUDA graphs, and sampler."""
    if vram_total_gb <= 0:
        return 0.90
    return (vram_total_gb - overhead_gb) / vram_total_gb


def resolve_gpu_memory_utilization(
    vram_total_gb: float,
    *,
    requested: float | None = None,
    config: object | None = None,
) -> float:
    """Resolve the effective GPU memory policy used by add, tune, and run."""
    if requested is not None:
        return requested
    configured_util = getattr(config, "gpu_memory_utilization", None)
    if configured_util is not None:
        return float(configured_util)
    configured_overhead = getattr(config, "gpu_overhead_gb", None)
    if configured_overhead is not None:
        return compute_gpu_memory_utilization(vram_total_gb, float(configured_overhead))
    return compute_gpu_memory_utilization(vram_total_gb)


def get_fan_speed() -> int:
    """Read current fan speed via NVML."""
    import pynvml  # type: ignore[import-untyped]
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetFanSpeed_v2(h, 0)
    finally:
        pynvml.nvmlShutdown()


def get_fan_count() -> int:
    """Return the number of fans exposed by NVML for the primary GPU."""
    import pynvml
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetNumFans(h)
    finally:
        pynvml.nvmlShutdown()


def set_fan_speed(percent: int) -> None:
    """Set GPU fan to a fixed percentage (30-100) via NVML. Requires root."""
    if not 30 <= percent <= 100:
        raise ValueError(f"Fan speed must be 30-100, got {percent}")

    import pynvml
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        for i in range(pynvml.nvmlDeviceGetNumFans(h)):
            pynvml.nvmlDeviceSetFanSpeed_v2(h, i, percent)
    finally:
        pynvml.nvmlShutdown()


def restore_fan_auto() -> None:
    """Restore automatic (vBIOS-controlled) fan control via NVML. Requires root."""
    import pynvml
    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        restore_default = getattr(pynvml, "nvmlDeviceSetDefaultFanSpeed_v2", None)
        for i in range(pynvml.nvmlDeviceGetNumFans(h)):
            if restore_default is not None:
                try:
                    restore_default(h, i)
                    continue
                except Exception:
                    pass
            pynvml.nvmlDeviceSetFanControlPolicy(h, i, 0)
    finally:
        pynvml.nvmlShutdown()
