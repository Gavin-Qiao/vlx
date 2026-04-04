"""Backend registry — auto-detect and manage inference backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vserve.backends.protocol import Backend
    from vserve.models import ModelInfo

_BACKENDS: list[Backend] = []


def register(backend: Backend) -> None:
    """Register a backend with the registry."""
    if any(b.name == backend.name for b in _BACKENDS):
        return
    _BACKENDS.append(backend)


def any_backend_running() -> bool:
    """Check if any registered backend has a running server."""
    for b in _BACKENDS:
        try:
            if b.is_running():
                return True
        except Exception:
            continue
    return False


def running_backend() -> Backend | None:
    """Return the currently running backend, or None."""
    for b in _BACKENDS:
        try:
            if b.is_running():
                return b
        except Exception:
            continue
    return None


def get_backend(model: ModelInfo) -> Backend:
    """Auto-detect the right backend for a model."""
    candidates = [b for b in _BACKENDS if b.can_serve(model)]
    if len(candidates) == 0:
        raise ValueError(
            f"No backend can serve {model.full_name}. "
            f"Registered: {[b.name for b in _BACKENDS]}"
        )
    return candidates[0]


def get_backend_by_name(name: str) -> Backend:
    """Look up a backend by name. Raises KeyError if not found."""
    for b in _BACKENDS:
        if b.name == name:
            return b
    raise KeyError(f"Unknown backend: {name!r}. Available: {[b.name for b in _BACKENDS]}")


def available_backends() -> list[Backend]:
    """Return only backends whose entrypoint is installed."""
    return [b for b in _BACKENDS if b.find_entrypoint() is not None]


# Auto-register built-in backends on import
def _register_defaults() -> None:
    from vserve.backends.vllm import VllmBackend
    from vserve.backends.llamacpp import LlamaCppBackend
    register(VllmBackend())
    register(LlamaCppBackend())


_register_defaults()
