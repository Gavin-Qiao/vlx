"""Backend Protocol — the interface every inference engine must implement."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vserve.gpu import GpuInfo
    from vserve.models import ModelInfo


@dataclass(frozen=True)
class RuntimeIdentity:
    backend: str
    executable: Path | str | None
    version: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompatibilityResult:
    backend: str
    supported: bool
    messages: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    supported_range: str | None = None


@dataclass(frozen=True)
class BackendCapabilities:
    tools: bool = False
    reasoning: bool = False
    embedding: bool = False
    tool_parser: str | None = None
    reasoning_parser: str | None = None


@dataclass(frozen=True)
class ServiceStatus:
    backend: str
    running: bool | None
    service_name: str
    health_url: str | None = None
    error: str | None = None


BackendConfig = dict[str, Any]
TuningResult = dict[str, Any]
ActiveManifest = Mapping[str, Any]
CacheFingerprint = Mapping[str, Any]
ProfileData = Mapping[str, Any]


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

    def runtime_info(self) -> RuntimeIdentity | Any:
        """Collect version and dependency facts for this backend runtime."""
        ...

    def compatibility(self) -> CompatibilityResult | Any:
        """Return this backend's runtime compatibility check."""
        ...

    def tune(self, model: ModelInfo, gpu: GpuInfo, *, gpu_mem_util: float) -> TuningResult:
        """Calculate optimal serving parameters. Returns limits dict."""
        ...

    def build_config(self, model: ModelInfo, choices: dict) -> BackendConfig:
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

    def active_manifest_path(self) -> Path:
        """Return the backend-owned manifest path for active runtime state."""
        ...

    def detect_tools(self, model_path: Path) -> Mapping[str, Any]:
        """Detect tool calling capabilities. Keys vary by backend."""
        ...

    def doctor_checks(self) -> list[tuple[str, Callable[[], bool]]]:
        """Return (description, check_fn) pairs for diagnostics."""
        ...
