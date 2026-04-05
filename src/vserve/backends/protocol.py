"""Backend Protocol — the interface every inference engine must implement."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vserve.gpu import GpuInfo
    from vserve.models import ModelInfo


@runtime_checkable
class Backend(Protocol):
    """What every inference backend must provide."""

    name: str
    display_name: str

    @property
    def service_name(self) -> str:
        """Systemd unit name for this backend."""
        ...

    @property
    def service_user(self) -> str:
        """System user that owns this backend service."""
        ...

    @property
    def root_dir(self) -> Path:
        """Root directory for this backend (e.g. /opt/vllm, /opt/llama-cpp)."""
        ...

    def can_serve(self, model: ModelInfo) -> bool:
        """Can this backend serve this model? (format check)"""
        ...

    def find_entrypoint(self) -> Path | str | None:
        """Locate the server binary or launch command. None = not installed."""
        ...

    def tune(self, model: ModelInfo, gpu: GpuInfo, *, gpu_mem_util: float) -> dict:
        """Calculate optimal serving parameters. Returns limits dict."""
        ...

    def build_config(self, model: ModelInfo, choices: dict) -> dict:
        """Build serving config from interactive wizard choices."""
        ...

    def quant_flag(self, method: str | None) -> str:
        """Return CLI quantization flag string. Empty if N/A."""
        ...

    def start(self, config_path: Path) -> None:
        """Start the inference server with the given config."""
        ...

    def stop(self) -> None:
        """Stop the inference server."""
        ...

    def is_running(self) -> bool:
        """Check if the inference server is currently active."""
        ...

    def health_url(self, port: int) -> str:
        """Return the health check URL for this backend."""
        ...

    def detect_tools(self, model_path: Path) -> dict:
        """Detect tool calling capabilities. Keys vary by backend."""
        ...

    def doctor_checks(self) -> list[tuple[str, Callable[[], bool]]]:
        """Return (description, check_fn) pairs for diagnostics."""
        ...
