"""Paths, YAML/JSON read/write, and configuration discovery."""

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_FILE = Path.home() / ".config" / "vserve" / "config.yaml"

# Hardcoded defaults — used only when discovery fails and no config file exists.
_DEFAULT_VLLM_ROOT = "/opt/vllm"
_DEFAULT_SERVICE_NAME = "vllm"
_DEFAULT_SERVICE_USER = "vllm"
_DEFAULT_PORT = 8888
_DEFAULT_CUDA_HOME = "/usr/local/cuda"


@dataclass
class VserveConfig:
    vllm_root: Path
    models_dir: Path
    configs_dir: Path
    benchmarks_dir: Path
    logs_dir: Path
    run_dir: Path
    vllm_bin: Path
    vllm_python: Path
    cuda_home: Path
    service_name: str
    service_user: str
    port: int
    gpu_memory_utilization: float | None = None
    gpu_overhead_gb: float | None = None
    llamacpp_root: Path | None = None
    llamacpp_service_name: str = "llama-cpp"
    llamacpp_service_user: str = "llama-cpp"

    @property
    def active_yaml(self) -> Path:
        return self.vllm_root / "configs" / "active.yaml"


def _discover_vllm_root() -> Path | None:
    """Try to find the vLLM root by locating the vllm binary or checking common paths."""
    # Method 1: find vllm on PATH
    vllm_path = shutil.which("vllm")
    if vllm_path:
        real = Path(vllm_path).resolve()
        # e.g. /opt/vllm/venv/bin/vllm → /opt/vllm
        if real.parent.name == "bin" and real.parent.parent.name in ("venv", ".venv"):
            return real.parent.parent.parent

    # Method 2: check common locations
    for candidate in [
        Path("/opt/vllm"),
        Path.home() / "vllm",
        Path.home() / ".vllm",
    ]:
        if (candidate / "venv" / "bin" / "vllm").exists() or \
           (candidate / ".venv" / "bin" / "vllm").exists():
            return candidate

    return None


def _discover_cuda_home() -> Path:
    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).resolve().parent.parent
    return Path(_DEFAULT_CUDA_HOME)


def _discover_service() -> tuple[str, str]:
    """Find vLLM systemd service name and user."""
    try:
        result = subprocess.run(
            ["systemctl", "list-unit-files", "--type=service", "--no-pager"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if "vllm" in line.lower():
                name = line.split()[0].removesuffix(".service")
                # Try to read User= from the unit
                show = subprocess.run(
                    ["systemctl", "show", name, "--property=User"],
                    capture_output=True, text=True, timeout=5,
                )
                user = show.stdout.strip().removeprefix("User=") or _DEFAULT_SERVICE_USER
                return name, user
    except Exception:
        pass
    return _DEFAULT_SERVICE_NAME, _DEFAULT_SERVICE_USER


def _discover_port(vllm_root: Path) -> int:
    """Try to read port from active config."""
    active = vllm_root / "configs" / "active.yaml"
    if active.exists():
        try:
            with open(active) as f:
                data = yaml.safe_load(f) or {}
            return int(str(data.get("port", _DEFAULT_PORT)))
        except Exception:
            pass
    return _DEFAULT_PORT


def _build_config(vllm_root: Path, cuda_home: Path, service_name: str,
                  service_user: str, port: int, *,
                  gpu_memory_utilization: float | None = None,
                  gpu_overhead_gb: float | None = None,
                  llamacpp_root: Path | None = None,
                  llamacpp_service_name: str = "llama-cpp",
                  llamacpp_service_user: str = "llama-cpp") -> VserveConfig:
    return VserveConfig(
        vllm_root=vllm_root,
        models_dir=vllm_root / "models",
        configs_dir=vllm_root / "configs" / "models",
        benchmarks_dir=vllm_root / "benchmarks",
        logs_dir=vllm_root / "logs",
        run_dir=vllm_root / "run",
        vllm_bin=vllm_root / "venv" / "bin" / "vllm",
        vllm_python=vllm_root / "venv" / "bin" / "python",
        cuda_home=cuda_home,
        service_name=service_name,
        service_user=service_user,
        port=port,
        gpu_memory_utilization=gpu_memory_utilization,
        gpu_overhead_gb=gpu_overhead_gb,
        llamacpp_root=llamacpp_root,
        llamacpp_service_name=llamacpp_service_name,
        llamacpp_service_user=llamacpp_service_user,
    )


def load_config() -> VserveConfig:
    """Load config: file > auto-discovery > defaults."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            data: dict[str, str] = yaml.safe_load(f) or {}
        root = Path(str(data.get("vllm_root", _DEFAULT_VLLM_ROOT)))
        gpu_util_raw = data.get("gpu_memory_utilization")
        gpu_overhead_raw = data.get("gpu_overhead_gb")
        llamacpp_root_raw = data.get("llamacpp_root")
        return _build_config(
            vllm_root=root,
            cuda_home=Path(str(data.get("cuda_home", _DEFAULT_CUDA_HOME))),
            service_name=str(data.get("service_name", _DEFAULT_SERVICE_NAME)),
            service_user=str(data.get("service_user", _DEFAULT_SERVICE_USER)),
            port=int(str(data.get("port", _DEFAULT_PORT))),
            gpu_memory_utilization=float(gpu_util_raw) if gpu_util_raw is not None else None,
            gpu_overhead_gb=float(gpu_overhead_raw) if gpu_overhead_raw is not None else None,
            llamacpp_root=Path(str(llamacpp_root_raw)) if llamacpp_root_raw else None,
            llamacpp_service_name=str(data.get("llamacpp_service_name", "llama-cpp")),
            llamacpp_service_user=str(data.get("llamacpp_service_user", "llama-cpp")),
        )

    # Auto-discover
    root = _discover_vllm_root() or Path(_DEFAULT_VLLM_ROOT)
    cuda_home = _discover_cuda_home()
    service_name, service_user = _discover_service()
    port = _discover_port(root)
    return _build_config(root, cuda_home, service_name, service_user, port)


def save_config(cfg: VserveConfig) -> Path:
    """Write config to ~/.config/vserve/config.yaml."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {
        "vllm_root": str(cfg.vllm_root),
        "cuda_home": str(cfg.cuda_home),
        "service_name": cfg.service_name,
        "service_user": cfg.service_user,
        "port": cfg.port,
    }
    if cfg.gpu_memory_utilization is not None:
        data["gpu_memory_utilization"] = cfg.gpu_memory_utilization
    if cfg.gpu_overhead_gb is not None:
        data["gpu_overhead_gb"] = cfg.gpu_overhead_gb
    if cfg.llamacpp_root is not None:
        data["llamacpp_root"] = str(cfg.llamacpp_root)
    if cfg.llamacpp_service_name != "llama-cpp":
        data["llamacpp_service_name"] = cfg.llamacpp_service_name
    if cfg.llamacpp_service_user != "llama-cpp":
        data["llamacpp_service_user"] = cfg.llamacpp_service_user
    with open(CONFIG_FILE, "w") as f:
        f.write("# vserve configuration — generated by vserve init\n")
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    return CONFIG_FILE


# --- Lazy singleton ---

_cfg: VserveConfig | None = None


def cfg() -> VserveConfig:
    """Return the global config, loading on first access."""
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


def reset_config() -> None:
    """Reset the singleton — for testing."""
    global _cfg
    _cfg = None


# --- Backward-compatible module-level constants ---
# These call cfg() lazily so imports don't trigger discovery at import time.


# Backward-compatible module-level constants via __getattr__.
# `from vserve.config import MODELS_DIR` resolves lazily on first access,
# returning a real Path from cfg(). Zero changes needed in callers.
_COMPAT = {
    "MODELS_DIR": "models_dir",
    "CONFIGS_DIR": "configs_dir",
    "BENCHMARKS_DIR": "benchmarks_dir",
    "LOGS_DIR": "logs_dir",
    "VLLM_BIN": "vllm_bin",
    "VLLM_PYTHON": "vllm_python",
    "VLLM_ROOT": "vllm_root",
}


def __getattr__(name: str):  # type: ignore[override]
    if name in _COMPAT:
        return getattr(cfg(), _COMPAT[name])
    raise AttributeError(f"module 'vserve.config' has no attribute {name!r}")


def limits_path(provider: str, model_name: str) -> Path:
    return cfg().configs_dir / f"{provider}--{model_name}.limits.json"


def profile_path(provider: str, model_name: str, profile: str) -> Path:
    return cfg().configs_dir / f"{provider}--{model_name}.{profile}.yaml"


def active_yaml_path() -> Path:
    return cfg().active_yaml


def read_limits(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def write_limits(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def timing_path(provider: str, model_name: str) -> Path:
    return cfg().configs_dir / f"{provider}--{model_name}.timing.json"


def read_timing(provider: str, model_name: str) -> dict:
    p = timing_path(provider, model_name)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def write_timing(provider: str, model_name: str, phase: str, seconds: float) -> None:
    p = timing_path(provider, model_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = read_timing(provider, model_name)
    data[phase] = round(seconds, 1)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def read_profile_yaml(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def write_profile_yaml(path: Path, data: dict, comment: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        if comment:
            for line in comment.splitlines():
                f.write(f"# {line}\n")
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
