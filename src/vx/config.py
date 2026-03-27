"""Paths, YAML/JSON read/write, and data model helpers."""

import json
from pathlib import Path

import yaml

VLLM_ROOT = Path("/opt/vllm")
MODELS_DIR = VLLM_ROOT / "models"
CONFIGS_DIR = VLLM_ROOT / "configs" / "models"
BENCHMARKS_DIR = VLLM_ROOT / "benchmarks"
VLLM_BIN = VLLM_ROOT / "venv" / "bin" / "vllm"
VLLM_PYTHON = VLLM_ROOT / "venv" / "bin" / "python"


def limits_path(provider: str, model_name: str) -> Path:
    return CONFIGS_DIR / f"{provider}--{model_name}.limits.json"


def profile_path(provider: str, model_name: str, profile: str) -> Path:
    return CONFIGS_DIR / f"{provider}--{model_name}.{profile}.yaml"


def active_yaml_path() -> Path:
    return VLLM_ROOT / "configs" / "active.yaml"


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
