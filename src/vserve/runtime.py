"""Runtime metadata, compatibility checks, and upgrade helpers."""

from __future__ import annotations

import json
import hashlib
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from vserve.model_files import is_weight_file_name, iter_recursive_files_with_suffix

SUPPORTED_VLLM_RANGE = ">=0.20,<0.21"
PINNED_STABLE_VLLM = "0.20.0"
DETECTOR_SCHEMA_VERSION = 2
_VLLM_MIN = Version("0.20")
_VLLM_MAX = Version("0.21")


@dataclass(frozen=True)
class RuntimeInfo:
    """Facts collected from an external inference runtime."""

    backend: str
    executable: Path | None
    python: Path | None
    vllm_version: str | None = None
    torch_version: str | None = None
    torch_cuda: str | None = None
    transformers_version: str | None = None
    huggingface_hub_version: str | None = None
    pip_check_ok: bool | None = None
    pip_check_output: str = ""
    errors: tuple[str, ...] = field(default_factory=tuple)

    def fingerprint(self) -> dict[str, str | None]:
        """Return runtime fields that should invalidate tuning caches when they drift."""
        return {
            "vllm_version": self.vllm_version,
            "torch_version": self.torch_version,
            "torch_cuda": self.torch_cuda,
            "transformers_version": self.transformers_version,
        }


@dataclass(frozen=True)
class CompatibilityCheck:
    """Result of checking an installed runtime against vserve's support policy."""

    backend: str
    range: str
    supported: bool
    messages: list[str]
    warnings: list[str]
    errors: list[str]


_METADATA_SCRIPT = (
    "import importlib.metadata as m, json\n"
    "def version(name):\n"
    "    try:\n"
    "        return m.version(name)\n"
    "    except Exception:\n"
    "        return None\n"
    "torch_version = version('torch')\n"
    "torch_cuda = None\n"
    "try:\n"
    "    import torch\n"
    "    torch_cuda = getattr(torch.version, 'cuda', None)\n"
    "except Exception:\n"
    "    pass\n"
    "print(json.dumps({\n"
    "    'vllm': version('vllm'),\n"
    "    'torch': torch_version,\n"
    "    'torch_cuda': torch_cuda,\n"
    "    'transformers': version('transformers'),\n"
    "    'huggingface_hub': version('huggingface-hub'),\n"
    "}))\n"
)


def _parse_version_text(text: str) -> str | None:
    match = re.search(r"(?i)vllm\s+([0-9][^\s]+)", text)
    if match:
        return match.group(1)
    match = re.search(r"([0-9]+(?:\.[0-9]+)+(?:[a-z0-9.+-]+)?)", text)
    return match.group(1) if match else None


def collect_vllm_runtime_info(config: Any | None = None) -> RuntimeInfo:
    """Collect vLLM, Python package, and dependency-health facts from /opt/vllm."""
    if config is None:
        from vserve.config import cfg

        config = cfg()

    vllm_bin = Path(config.vllm_bin)
    vllm_python = Path(config.vllm_python)
    errors: list[str] = []
    cli_version: str | None = None
    metadata: dict[str, Any] = {}
    pip_check_ok: bool | None = None
    pip_check_output = ""

    try:
        result = subprocess.run(
            [str(vllm_bin), "--version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout.strip() or result.stderr.strip()
        if result.returncode == 0:
            cli_version = _parse_version_text(output)
        else:
            errors.append(output or f"{vllm_bin} --version failed")
    except Exception as exc:
        errors.append(f"{vllm_bin} --version failed: {exc}")

    try:
        result = subprocess.run(
            [str(vllm_python), "-c", _METADATA_SCRIPT],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode == 0:
            metadata = json.loads(result.stdout.strip() or "{}")
        else:
            details = result.stderr.strip() or result.stdout.strip()
            errors.append(details or f"{vllm_python} metadata probe failed")
    except Exception as exc:
        errors.append(f"{vllm_python} metadata probe failed: {exc}")

    try:
        result = subprocess.run(
            [str(vllm_python), "-m", "pip", "check"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        pip_check_ok = result.returncode == 0
        pip_check_output = result.stdout.strip() or result.stderr.strip()
    except Exception as exc:
        pip_check_ok = False
        pip_check_output = str(exc)

    return RuntimeInfo(
        backend="vllm",
        executable=vllm_bin,
        python=vllm_python,
        vllm_version=metadata.get("vllm") or cli_version,
        torch_version=metadata.get("torch"),
        torch_cuda=metadata.get("torch_cuda"),
        transformers_version=metadata.get("transformers"),
        huggingface_hub_version=metadata.get("huggingface_hub"),
        pip_check_ok=pip_check_ok,
        pip_check_output=pip_check_output,
        errors=tuple(errors),
    )


def check_vllm_compatibility(info: RuntimeInfo) -> CompatibilityCheck:
    """Check vLLM runtime against vserve's pinned support policy."""
    messages: list[str] = []
    warnings: list[str] = []
    raw_errors = getattr(info, "errors", ())
    errors = list(raw_errors) if isinstance(raw_errors, (list, tuple)) else []

    if not info.vllm_version:
        errors.append("Could not determine vLLM version.")
    else:
        try:
            parsed = Version(info.vllm_version)
        except InvalidVersion:
            errors.append(f"Invalid vLLM version: {info.vllm_version}")
        else:
            if parsed.is_prerelease or parsed.is_devrelease:
                errors.append(f"vLLM pre-release/dev build is unsupported: {info.vllm_version}")
            if not (_VLLM_MIN <= parsed < _VLLM_MAX):
                errors.append(f"vLLM {info.vllm_version} is outside supported range {SUPPORTED_VLLM_RANGE}.")
            if not errors:
                messages.append(f"vLLM {info.vllm_version} is within supported range {SUPPORTED_VLLM_RANGE}.")

    if info.pip_check_ok is False:
        errors.append(f"pip check failed: {info.pip_check_output or 'dependency conflicts found'}")
    elif info.pip_check_ok is None:
        warnings.append("pip check was not run.")

    if not info.torch_version:
        warnings.append("Could not determine torch version.")
    if not info.transformers_version:
        warnings.append("Could not determine transformers version.")

    return CompatibilityCheck(
        backend="vllm",
        range=SUPPORTED_VLLM_RANGE,
        supported=not errors,
        messages=messages,
        warnings=warnings,
        errors=errors,
    )


def upgrade_vllm_stable(config: Any | None = None, *, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    """Force-reinstall the supported stable vLLM runtime into the configured venv."""
    if config is None:
        from vserve.config import cfg

        config = cfg()
    vllm_python = Path(config.vllm_python)
    if not vllm_python.exists():
        raise RuntimeError(f"vLLM Python not found at {vllm_python}")
    result = subprocess.run(
        [
            str(vllm_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            f"vllm=={PINNED_STABLE_VLLM}",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(details or f"pip install vllm=={PINNED_STABLE_VLLM} failed")
    return result


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tokenizer_template_hash(model_path: Path) -> str | None:
    path = model_path / "tokenizer_config.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict) or "chat_template" not in data:
        return None
    template = data["chat_template"]
    if isinstance(template, str):
        return _sha256_text(template)
    try:
        normalized = json.dumps(template, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return None
    return _sha256_text(normalized)


def _model_file_identity(model_path: Path) -> list[dict[str, int | str]]:
    identity: list[dict[str, int | str]] = []
    if not model_path.exists():
        return identity
    for path in sorted(model_path.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(model_path).as_posix()
        if not (is_weight_file_name(rel) or rel.lower().endswith(".index.json")):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        identity.append({
            "path": rel,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return identity


def _gguf_metadata_hash(model_path: Path) -> str | None:
    gguf_files = iter_recursive_files_with_suffix(model_path, ".gguf")
    if not gguf_files:
        return None
    digest = hashlib.sha256()
    for path in gguf_files:
        digest.update(path.relative_to(model_path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        try:
            with open(path, "rb") as f:
                digest.update(f.read(1024 * 1024))
        except OSError:
            continue
    return digest.hexdigest()


def build_tuning_fingerprint(
    *,
    model_info: Any,
    gpu: Any,
    backend: str,
    gpu_mem_util: float,
    runtime_info: Any | None = None,
) -> dict[str, Any]:
    """Return the inputs that make a cached tuning table valid."""
    fingerprint: dict[str, Any] = {
        "backend": backend,
        "model_path": str(model_info.path),
        "quant_method": model_info.quant_method,
        "architecture": model_info.architecture,
        "is_moe": model_info.is_moe,
        "max_position_embeddings": model_info.max_position_embeddings,
        "num_kv_heads": model_info.num_kv_heads,
        "head_dim": model_info.head_dim,
        "num_layers": model_info.num_layers,
        "model_size_gb": model_info.model_size_gb,
        "gpu_name": getattr(gpu, "name", None),
        "gpu_index": getattr(gpu, "index", None),
        "gpu_driver": getattr(gpu, "driver", None),
        "gpu_cuda": getattr(gpu, "cuda", None),
        "vram_total_gb": getattr(gpu, "vram_total_gb", None),
        "gpu_memory_utilization": gpu_mem_util,
        "detector_schema_version": DETECTOR_SCHEMA_VERSION,
        "tokenizer_template_hash": _tokenizer_template_hash(Path(model_info.path)),
        "gguf_metadata_hash": _gguf_metadata_hash(Path(model_info.path)),
        "model_file_identity": _model_file_identity(Path(model_info.path)),
        "vllm_version": None,
        "torch_version": None,
        "torch_cuda": None,
        "transformers_version": None,
        "runtime_executable": None,
        "llama_server_version": None,
    }
    if runtime_info is not None:
        if isinstance(runtime_info, dict):
            fingerprint["runtime_executable"] = runtime_info.get("executable")
            fingerprint["llama_server_version"] = runtime_info.get("llama_server_version")
        else:
            runtime_fingerprint = runtime_info.fingerprint()
            if isinstance(runtime_fingerprint, dict):
                fingerprint.update(runtime_fingerprint)
    return fingerprint
