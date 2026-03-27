# vx Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `vx`, a Python CLI that manages vLLM models — download, probe limits, benchmark, compare, and serve — with guided interactive UX.

**Architecture:** Typer CLI with 7 modules (config, gpu, models, probe, tune, serve, compare). Each module is independently testable with `tmp_path` fixtures and mock subprocesses. The probe module builds a 2D limits table (context x kv-dtype → max concurrency); all other modules query it.

**Tech Stack:** Python 3.12, Typer 0.24, Rich, pytest, uv, huggingface_hub, PyYAML

**Spec:** `docs/specs/2026-03-27-vx-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Project metadata, dependencies, `[project.scripts] vx = "vx.cli:app"` |
| `src/vx/__init__.py` | Package marker, version |
| `src/vx/config.py` | Paths (`MODELS_DIR`, `CONFIGS_DIR`, etc.), YAML/JSON read/write, `limits.json` schema, active symlink |
| `src/vx/gpu.py` | `nvidia-smi` queries: VRAM total, utilization, driver, GPU name, compute `gpu_memory_utilization` |
| `src/vx/models.py` | Scan downloaded models, parse `config.json`, fuzzy match, model size, quant flag mapping |
| `src/vx/probe.py` | Binary-search context/concurrency limits, write `limits.json` |
| `src/vx/tune.py` | Generate sweep param JSONs, invoke `vllm bench sweep`, parse results, pick winner, write profile YAML |
| `src/vx/serve.py` | Start/stop vLLM via systemd, update active symlink |
| `src/vx/compare.py` | Load limits for all models, filter by workload requirements, rank |
| `src/vx/cli.py` | Typer app with all subcommands, interactive prompts, dashboard |
| `tests/conftest.py` | Shared fixtures: `fake_model_dir`, `fake_limits`, `fake_config_json` |
| `tests/test_config.py` | Path helpers, JSON/YAML read/write |
| `tests/test_gpu.py` | GPU info parsing with mocked nvidia-smi |
| `tests/test_models.py` | Model detection, quant mapping, fuzzy matching |
| `tests/test_probe.py` | Probe logic with mocked subprocess |
| `tests/test_tune.py` | Param generation, winner selection |
| `tests/test_compare.py` | Workload filtering and ranking |
| `tests/test_serve.py` | Start/stop with mocked systemctl |
| `tests/test_cli.py` | CLI smoke tests via `CliRunner` |

---

### Task 1: Project scaffold + config module

**Files:**
- Create: `pyproject.toml`
- Create: `src/vx/__init__.py`
- Create: `src/vx/config.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "vx"
version = "0.1.0"
description = "vLLM model manager — download, probe, tune, serve, compare"
requires-python = ">=3.12"
dependencies = [
    "typer>=0.15",
    "rich>=14",
    "huggingface-hub>=0.30",
    "pyyaml>=6",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-mock>=3"]

[project.scripts]
vx = "vx.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.backends"

[tool.hatch.build.targets.wheel]
packages = ["src/vx"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package init**

```python
# src/vx/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 3: Write the failing test for config**

```python
# tests/test_config.py
from pathlib import Path
from vx.config import (
    MODELS_DIR,
    CONFIGS_DIR,
    BENCHMARKS_DIR,
    VLLM_BIN,
    limits_path,
    profile_path,
    active_yaml_path,
    read_limits,
    write_limits,
    read_profile_yaml,
    write_profile_yaml,
)


def test_paths_are_under_opt_vllm():
    assert str(MODELS_DIR).startswith("/opt/vllm")
    assert str(CONFIGS_DIR).startswith("/opt/vllm")
    assert str(BENCHMARKS_DIR).startswith("/opt/vllm")


def test_limits_path():
    p = limits_path("apolo13x", "Qwen3.5-27B-NVFP4")
    assert p == CONFIGS_DIR / "apolo13x--Qwen3.5-27B-NVFP4.limits.json"


def test_profile_path():
    p = profile_path("apolo13x", "Qwen3.5-27B-NVFP4", "interactive")
    assert p == CONFIGS_DIR / "apolo13x--Qwen3.5-27B-NVFP4.interactive.yaml"


def test_active_yaml_path():
    assert active_yaml_path() == Path("/opt/vllm/configs/active.yaml")


def test_write_and_read_limits(tmp_path):
    data = {
        "model_path": "/opt/vllm/models/test/model",
        "limits": {"4096": {"auto": 64, "fp8": 128}},
    }
    path = tmp_path / "test.limits.json"
    write_limits(path, data)
    loaded = read_limits(path)
    assert loaded["limits"]["4096"]["fp8"] == 128


def test_read_limits_missing(tmp_path):
    path = tmp_path / "nonexistent.json"
    assert read_limits(path) is None


def test_write_and_read_profile_yaml(tmp_path):
    data = {
        "model": "/opt/vllm/models/test/model",
        "host": "0.0.0.0",
        "port": 8888,
        "gpu-memory-utilization": 0.958,
    }
    path = tmp_path / "test.yaml"
    write_profile_yaml(path, data, comment="test profile")
    content = path.read_text()
    assert "# test profile" in content
    loaded = read_profile_yaml(path)
    assert loaded["port"] == 8888
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd ~/vx && uv sync && uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vx.config'`

- [ ] **Step 5: Write conftest.py with shared fixtures**

```python
# tests/conftest.py
import json
from pathlib import Path

import pytest


@pytest.fixture
def fake_model_dir(tmp_path: Path) -> Path:
    """Create a fake model directory with config.json."""
    model_dir = tmp_path / "models" / "testprovider" / "TestModel-7B-FP8"
    model_dir.mkdir(parents=True)

    config = {
        "architectures": ["TestForCausalLM"],
        "model_type": "test",
        "quantization_config": {"quant_method": "fp8"},
        "text_config": {
            "max_position_embeddings": 131072,
            "num_key_value_heads": 4,
            "hidden_size": 4096,
            "num_hidden_layers": 32,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))

    # Fake safetensors file (~1GB)
    fake_weights = model_dir / "model.safetensors"
    fake_weights.write_bytes(b"\0" * 1024)  # tiny for tests

    return model_dir


@pytest.fixture
def fake_moe_model_dir(tmp_path: Path) -> Path:
    """Create a fake MoE model directory."""
    model_dir = tmp_path / "models" / "testprovider" / "TestMoE-35B-FP8"
    model_dir.mkdir(parents=True)

    config = {
        "architectures": ["TestMoeForCausalLM"],
        "model_type": "test_moe",
        "quantization_config": {"quant_method": "fp8"},
        "text_config": {
            "max_position_embeddings": 262144,
            "num_experts": 256,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "model.safetensors").write_bytes(b"\0" * 2048)

    return model_dir


@pytest.fixture
def fake_limits() -> dict:
    """Sample limits.json data."""
    return {
        "model_path": "/opt/vllm/models/testprovider/TestModel-7B-FP8",
        "probed_at": "2026-03-27T10:00:00",
        "vram_total_gb": 47.8,
        "gpu_memory_utilization": 0.958,
        "model_size_gb": 7.2,
        "quant_method": "fp8",
        "architecture": "TestForCausalLM",
        "is_moe": False,
        "max_position_embeddings": 131072,
        "limits": {
            "4096": {"auto": 64, "fp8": 128},
            "8192": {"auto": 32, "fp8": 64},
            "16384": {"auto": 16, "fp8": 32},
            "32768": {"auto": 8, "fp8": 16},
            "65536": {"auto": None, "fp8": 8},
            "131072": {"auto": None, "fp8": None},
        },
    }


@pytest.fixture
def models_root(tmp_path: Path, fake_model_dir: Path) -> Path:
    """Return the models root containing fake_model_dir."""
    return tmp_path / "models"
```

- [ ] **Step 6: Write config.py**

```python
# src/vx/config.py
"""Paths, YAML/JSON read/write, and data model helpers."""

import json
from pathlib import Path

import yaml

# ── Paths ──────────────────────────────────────────

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


# ── JSON (limits) ──────────────────────────────────


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


# ── YAML (profiles) ───────────────────────────────


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
```

- [ ] **Step 7: Run tests**

Run: `cd ~/vx && uv run pytest tests/test_config.py -v`
Expected: all PASS

- [ ] **Step 8: Commit**

```bash
cd ~/vx && git add -A && git commit -m "feat: project scaffold + config module with paths and YAML/JSON helpers"
```

---

### Task 2: GPU module

**Files:**
- Create: `src/vx/gpu.py`
- Create: `tests/test_gpu.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gpu.py
from vx.gpu import GpuInfo, get_gpu_info, compute_gpu_memory_utilization


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/vx && uv run pytest tests/test_gpu.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vx.gpu'`

- [ ] **Step 3: Write gpu.py**

```python
# src/vx/gpu.py
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
```

- [ ] **Step 4: Run tests**

Run: `cd ~/vx && uv run pytest tests/test_gpu.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd ~/vx && git add -A && git commit -m "feat: GPU info module with nvidia-smi parsing and VRAM utilization"
```

---

### Task 3: Models module — detection + fuzzy matching

**Files:**
- Create: `src/vx/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models.py
import json
from pathlib import Path

import pytest

from vx.models import (
    ModelInfo,
    detect_model,
    scan_models,
    fuzzy_match,
    quant_flag,
)


def test_detect_model_basic(fake_model_dir):
    info = detect_model(fake_model_dir)
    assert info.architecture == "TestForCausalLM"
    assert info.quant_method == "fp8"
    assert info.max_position_embeddings == 131072
    assert info.is_moe is False
    assert info.provider == "testprovider"
    assert info.model_name == "TestModel-7B-FP8"


def test_detect_model_moe(fake_moe_model_dir):
    info = detect_model(fake_moe_model_dir)
    assert info.is_moe is True
    assert info.max_position_embeddings == 262144


def test_detect_model_missing_config(tmp_path):
    empty = tmp_path / "models" / "test" / "empty"
    empty.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        detect_model(empty)


def test_quant_flag_mapping():
    assert quant_flag("gptq") == "--quantization gptq_marlin"
    assert quant_flag("awq") == "--quantization awq_marlin"
    assert quant_flag("fp8") == "--quantization fp8"
    assert quant_flag("compressed-tensors") == ""
    assert quant_flag("none") == ""
    assert quant_flag(None) == ""


def test_scan_models(models_root):
    models = scan_models(models_root)
    assert len(models) == 1
    assert models[0].model_name == "TestModel-7B-FP8"


def test_scan_models_empty(tmp_path):
    empty_root = tmp_path / "models"
    empty_root.mkdir()
    assert scan_models(empty_root) == []


@pytest.mark.parametrize(
    "query,expected_count",
    [
        ("TestModel-7B-FP8", 1),          # exact model name
        ("testmodel", 1),                   # case-insensitive substring
        ("7b-fp8", 1),                      # partial match
        ("fp8", 1),                         # short substring
        ("nonexistent", 0),                 # no match
    ],
)
def test_fuzzy_match(models_root, query, expected_count):
    models = scan_models(models_root)
    matches = fuzzy_match(query, models)
    assert len(matches) == expected_count


def test_fuzzy_match_exact_provider_model(models_root):
    models = scan_models(models_root)
    matches = fuzzy_match("testprovider/TestModel-7B-FP8", models)
    assert len(matches) == 1


def test_fuzzy_match_full_path(models_root):
    models = scan_models(models_root)
    full_path = str(models[0].path)
    matches = fuzzy_match(full_path, models)
    assert len(matches) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/vx && uv run pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vx.models'`

- [ ] **Step 3: Write models.py**

```python
# src/vx/models.py
"""Model scanning, detection, and fuzzy matching."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelInfo:
    path: Path
    provider: str
    model_name: str
    architecture: str
    model_type: str
    quant_method: str | None
    max_position_embeddings: int
    is_moe: bool
    model_size_gb: float

    @property
    def full_name(self) -> str:
        return f"{self.provider}/{self.model_name}"


QUANT_FLAGS = {
    "gptq": "--quantization gptq_marlin",
    "awq": "--quantization awq_marlin",
    "fp8": "--quantization fp8",
    "compressed-tensors": "",
    "none": "",
}


def quant_flag(method: str | None) -> str:
    if method is None:
        return ""
    return QUANT_FLAGS.get(method, "")


def detect_model(model_dir: Path) -> ModelInfo:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {model_dir}")

    with open(config_path) as f:
        config = json.load(f)

    text_config = config.get("text_config", {})

    arch = (config.get("architectures") or ["unknown"])[0]
    model_type = config.get("model_type", "unknown")
    qmethod = (config.get("quantization_config") or {}).get("quant_method")
    max_pos = text_config.get(
        "max_position_embeddings",
        config.get("max_position_embeddings", 32768),
    )
    num_experts = text_config.get("num_experts", config.get("num_experts"))
    is_moe = num_experts is not None and num_experts > 0

    # Model size from safetensors files
    size_bytes = sum(
        f.stat().st_size for f in model_dir.glob("*.safetensors")
    )
    size_gb = round(size_bytes / (1024**3), 1)

    provider = model_dir.parent.name
    model_name = model_dir.name

    return ModelInfo(
        path=model_dir,
        provider=provider,
        model_name=model_name,
        architecture=arch,
        model_type=model_type,
        quant_method=qmethod,
        max_position_embeddings=max_pos,
        is_moe=is_moe,
        model_size_gb=size_gb,
    )


def scan_models(models_root: Path) -> list[ModelInfo]:
    models = []
    if not models_root.exists():
        return models
    for provider_dir in sorted(models_root.iterdir()):
        if not provider_dir.is_dir():
            continue
        for model_dir in sorted(provider_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if not (model_dir / "config.json").exists():
                continue
            models.append(detect_model(model_dir))
    return models


def fuzzy_match(query: str, models: list[ModelInfo]) -> list[ModelInfo]:
    # 1. Exact path match
    for m in models:
        if query == str(m.path):
            return [m]

    # 2. Exact provider/model match
    for m in models:
        if query == m.full_name:
            return [m]

    # 3. Case-insensitive substring on model name or full name
    query_lower = query.lower()
    matches = [
        m
        for m in models
        if query_lower in m.model_name.lower()
        or query_lower in m.full_name.lower()
    ]
    return matches
```

- [ ] **Step 4: Run tests**

Run: `cd ~/vx && uv run pytest tests/test_models.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd ~/vx && git add -A && git commit -m "feat: model detection, scanning, quant flag mapping, and fuzzy matching"
```

---

### Task 4: Probe module

**Files:**
- Create: `src/vx/probe.py`
- Create: `tests/test_probe.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_probe.py
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vx.models import ModelInfo
from vx.probe import (
    probe_context_limit,
    probe_concurrency_limit,
    probe_model,
    CONTEXT_STEPS,
)


@pytest.fixture
def sample_model_info(fake_model_dir) -> ModelInfo:
    from vx.models import detect_model
    return detect_model(fake_model_dir)


def test_context_steps_are_powers_of_two():
    for step in CONTEXT_STEPS:
        assert step & (step - 1) == 0, f"{step} is not a power of 2"


def test_probe_context_limit_success(mocker, sample_model_info):
    """vLLM starts successfully at 4096 → returns True."""
    mock_run = mocker.patch("vx.probe._try_start_vllm", return_value=True)
    result = probe_context_limit(
        model_info=sample_model_info,
        context_len=4096,
        kv_dtype="fp8",
        gpu_mem_util=0.958,
    )
    assert result is True


def test_probe_context_limit_oom(mocker, sample_model_info):
    """vLLM fails to start → returns False."""
    mocker.patch("vx.probe._try_start_vllm", return_value=False)
    result = probe_context_limit(
        model_info=sample_model_info,
        context_len=262144,
        kv_dtype="auto",
        gpu_mem_util=0.958,
    )
    assert result is False


def test_probe_concurrency_limit(mocker, sample_model_info):
    """Binary search finds max concurrency before OOM."""
    # Succeeds for max_num_seqs <= 32, fails at 64
    def fake_start(*, model_info, context_len, kv_dtype, gpu_mem_util, max_num_seqs):
        return max_num_seqs <= 32

    mocker.patch("vx.probe._try_start_vllm", side_effect=fake_start)
    max_seqs = probe_concurrency_limit(
        model_info=sample_model_info,
        context_len=8192,
        kv_dtype="fp8",
        gpu_mem_util=0.958,
    )
    assert max_seqs == 32


def test_probe_concurrency_limit_none(mocker, sample_model_info):
    """If even 1 slot fails, return None."""
    mocker.patch("vx.probe._try_start_vllm", return_value=False)
    max_seqs = probe_concurrency_limit(
        model_info=sample_model_info,
        context_len=262144,
        kv_dtype="auto",
        gpu_mem_util=0.958,
    )
    assert max_seqs is None


def test_probe_model_full(mocker, sample_model_info, tmp_path):
    """Full probe builds the limits dict."""
    call_log = []

    def fake_start(*, model_info, context_len, kv_dtype, gpu_mem_util, max_num_seqs=1):
        call_log.append((context_len, kv_dtype, max_num_seqs))
        # auto works up to 32k, fp8 works up to 65k
        if kv_dtype == "auto" and context_len > 32768:
            return False
        if kv_dtype == "fp8" and context_len > 65536:
            return False
        # Concurrency: fits up to 16 at any working context
        if max_num_seqs > 16:
            return False
        return True

    mocker.patch("vx.probe._try_start_vllm", side_effect=fake_start)

    limits = probe_model(
        model_info=sample_model_info,
        gpu_mem_util=0.958,
        vram_total_gb=47.8,
    )

    assert limits["limits"]["4096"]["auto"] == 16
    assert limits["limits"]["4096"]["fp8"] == 16
    assert limits["limits"]["65536"]["auto"] is None
    assert limits["limits"]["65536"]["fp8"] == 16
    assert limits["limits"]["131072"]["fp8"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/vx && uv run pytest tests/test_probe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vx.probe'`

- [ ] **Step 3: Write probe.py**

```python
# src/vx/probe.py
"""Probe context and concurrency limits for a model."""

import subprocess
import time
from datetime import datetime, timezone

import requests

from vx.config import VLLM_BIN
from vx.models import ModelInfo, quant_flag

CONTEXT_STEPS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
CONCURRENCY_STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256]

_PROBE_PORT = 18188  # unlikely to conflict
_HEALTH_TIMEOUT = 300  # seconds to wait for server startup


def _try_start_vllm(
    *,
    model_info: ModelInfo,
    context_len: int,
    kv_dtype: str,
    gpu_mem_util: float,
    max_num_seqs: int = 1,
) -> bool:
    """Try to start vLLM with given params. Returns True if server becomes ready."""
    cmd = [
        str(VLLM_BIN),
        "serve",
        str(model_info.path),
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--host", "127.0.0.1",
        "--port", str(_PROBE_PORT),
        "--max-model-len", str(context_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--kv-cache-dtype", kv_dtype,
        "--max-num-seqs", str(max_num_seqs),
    ]

    qflag = quant_flag(model_info.quant_method)
    if qflag:
        cmd.extend(qflag.split())

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        deadline = time.monotonic() + _HEALTH_TIMEOUT
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return False  # process exited (OOM or error)
            try:
                resp = requests.get(
                    f"http://127.0.0.1:{_PROBE_PORT}/health", timeout=2
                )
                if resp.status_code == 200:
                    return True
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(2)
        return False  # timeout
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def probe_context_limit(
    *,
    model_info: ModelInfo,
    context_len: int,
    kv_dtype: str,
    gpu_mem_util: float,
) -> bool:
    """Check if a model can start with the given context length."""
    return _try_start_vllm(
        model_info=model_info,
        context_len=context_len,
        kv_dtype=kv_dtype,
        gpu_mem_util=gpu_mem_util,
        max_num_seqs=1,
    )


def probe_concurrency_limit(
    *,
    model_info: ModelInfo,
    context_len: int,
    kv_dtype: str,
    gpu_mem_util: float,
) -> int | None:
    """Binary search for max max-num-seqs at a given context length."""
    best = None
    for seqs in CONCURRENCY_STEPS:
        ok = _try_start_vllm(
            model_info=model_info,
            context_len=context_len,
            kv_dtype=kv_dtype,
            gpu_mem_util=gpu_mem_util,
            max_num_seqs=seqs,
        )
        if ok:
            best = seqs
        else:
            break
    return best


def probe_model(
    *,
    model_info: ModelInfo,
    gpu_mem_util: float,
    vram_total_gb: float,
) -> dict:
    """Full probe: context x kv-dtype → max concurrency."""
    max_pos = model_info.max_position_embeddings
    steps = [s for s in CONTEXT_STEPS if s <= max_pos]

    limits: dict[str, dict[str, int | None]] = {}

    for kv_dtype in ("auto", "fp8"):
        hit_ceiling = False
        for ctx in steps:
            key = str(ctx)
            if key not in limits:
                limits[key] = {}

            if hit_ceiling:
                limits[key][kv_dtype] = None
                continue

            if not probe_context_limit(
                model_info=model_info,
                context_len=ctx,
                kv_dtype=kv_dtype,
                gpu_mem_util=gpu_mem_util,
            ):
                limits[key][kv_dtype] = None
                hit_ceiling = True
                continue

            max_seqs = probe_concurrency_limit(
                model_info=model_info,
                context_len=ctx,
                kv_dtype=kv_dtype,
                gpu_mem_util=gpu_mem_util,
            )
            limits[key][kv_dtype] = max_seqs

    return {
        "model_path": str(model_info.path),
        "probed_at": datetime.now(timezone.utc).isoformat(),
        "vram_total_gb": vram_total_gb,
        "gpu_memory_utilization": gpu_mem_util,
        "model_size_gb": model_info.model_size_gb,
        "quant_method": model_info.quant_method,
        "architecture": model_info.architecture,
        "is_moe": model_info.is_moe,
        "max_position_embeddings": model_info.max_position_embeddings,
        "limits": limits,
    }
```

- [ ] **Step 4: Run tests**

Run: `cd ~/vx && uv run pytest tests/test_probe.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd ~/vx && git add -A && git commit -m "feat: probe module — binary search context/concurrency limits"
```

---

### Task 5: Tune module — param generation + winner selection

**Files:**
- Create: `src/vx/tune.py`
- Create: `tests/test_tune.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tune.py
import json
from pathlib import Path

import pytest

from vx.tune import (
    generate_sweep_params,
    pick_winner,
    PROFILES,
)


@pytest.fixture
def sample_limits(fake_limits):
    return fake_limits


def test_profiles_exist():
    assert set(PROFILES.keys()) == {"interactive", "batch", "agentic", "creative"}


def test_generate_interactive_params(sample_limits, tmp_path):
    serve, bench, subcommand = generate_sweep_params(
        profile="interactive",
        limits=sample_limits,
        output_dir=tmp_path,
    )
    serve_data = json.loads(serve.read_text())
    bench_data = json.loads(bench.read_text())
    assert len(serve_data) > 0
    assert len(bench_data) > 0
    assert subcommand == "serve"
    # All serve entries have kv_cache_dtype
    for entry in serve_data.values():
        assert "kv_cache_dtype" in entry
    # All bench entries have num_prompts = 20
    for entry in bench_data.values():
        assert entry["num_prompts"] == 20


def test_generate_batch_params(sample_limits, tmp_path):
    serve, bench, subcommand = generate_sweep_params(
        profile="batch",
        limits=sample_limits,
        output_dir=tmp_path,
    )
    assert subcommand == "serve_workload"
    serve_data = json.loads(serve.read_text())
    # max_num_seqs values present
    seqs_values = {e["max_num_seqs"] for e in serve_data.values()}
    assert seqs_values.issubset({8, 16, 32, 64})


def test_generate_agentic_params(sample_limits, tmp_path):
    serve, bench, subcommand = generate_sweep_params(
        profile="agentic",
        limits=sample_limits,
        output_dir=tmp_path,
    )
    serve_data = json.loads(serve.read_text())
    # All entries should use the max proven context
    for entry in serve_data.values():
        assert entry["max_model_len"] == 32768  # max where fp8 has slots


def test_generate_creative_params(sample_limits, tmp_path):
    serve, bench, subcommand = generate_sweep_params(
        profile="creative",
        limits=sample_limits,
        output_dir=tmp_path,
    )
    bench_data = json.loads(bench.read_text())
    # Output lengths should be large
    for entry in bench_data.values():
        assert entry["random_output_len"] >= 1024


def test_generate_batch_moe_caps_seqs(sample_limits, tmp_path):
    sample_limits["is_moe"] = True
    serve, bench, subcommand = generate_sweep_params(
        profile="batch",
        limits=sample_limits,
        output_dir=tmp_path,
    )
    serve_data = json.loads(serve.read_text())
    for entry in serve_data.values():
        assert entry["max_num_seqs"] <= 32


# ── Winner selection ───────────────────────────────


def test_pick_winner_interactive(tmp_path):
    # Create fake summary.json files (list of runs)
    combo = tmp_path / "SERVE--fast-BENCH--b"
    combo.mkdir(parents=True)
    (combo / "summary.json").write_text(json.dumps([
        {"median_tpot_ms": 11.0, "mean_ttft_ms": 40.0, "output_throughput": 85.0, "failed": 0, "gpu_memory_utilization": 0.95, "kv_cache_dtype": "fp8", "max_model_len": 4096, "enable_prefix_caching": True},
        {"median_tpot_ms": 12.0, "mean_ttft_ms": 42.0, "output_throughput": 83.0, "failed": 0, "gpu_memory_utilization": 0.95, "kv_cache_dtype": "fp8", "max_model_len": 4096, "enable_prefix_caching": True},
    ]))

    combo2 = tmp_path / "SERVE--slow-BENCH--b"
    combo2.mkdir(parents=True)
    (combo2 / "summary.json").write_text(json.dumps([
        {"median_tpot_ms": 20.0, "mean_ttft_ms": 80.0, "output_throughput": 50.0, "failed": 0, "gpu_memory_utilization": 0.90, "kv_cache_dtype": "auto", "max_model_len": 4096, "enable_prefix_caching": True},
        {"median_tpot_ms": 22.0, "mean_ttft_ms": 85.0, "output_throughput": 48.0, "failed": 0, "gpu_memory_utilization": 0.90, "kv_cache_dtype": "auto", "max_model_len": 4096, "enable_prefix_caching": True},
    ]))

    winner, top3 = pick_winner(tmp_path, profile="interactive")
    assert winner["params"]["kv_cache_dtype"] == "fp8"  # lower TTFT


def test_pick_winner_batch(tmp_path):
    combo = tmp_path / "SERVE--high-BENCH--b"
    combo.mkdir(parents=True)
    (combo / "summary.json").write_text(json.dumps([
        {"median_tpot_ms": 8.0, "mean_ttft_ms": 30.0, "output_throughput": 120.0, "failed": 0, "max_num_seqs": 32},
    ]))

    combo2 = tmp_path / "SERVE--low-BENCH--b"
    combo2.mkdir(parents=True)
    (combo2 / "summary.json").write_text(json.dumps([
        {"median_tpot_ms": 12.0, "mean_ttft_ms": 25.0, "output_throughput": 85.0, "failed": 0, "max_num_seqs": 8},
    ]))

    winner, _ = pick_winner(tmp_path, profile="batch")
    assert winner["output_throughput"] == 120.0


def test_pick_winner_no_results(tmp_path):
    with pytest.raises(SystemExit):
        pick_winner(tmp_path, profile="interactive")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/vx && uv run pytest tests/test_tune.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vx.tune'`

- [ ] **Step 3: Write tune.py**

```python
# src/vx/tune.py
"""Benchmark orchestration — param generation and winner selection."""

import json
import sys
from collections import defaultdict
from pathlib import Path

PROFILES = {
    "interactive": {
        "subcommand": "serve",
        "metric": "mean_ttft_ms",
        "direction": "minimize",
    },
    "batch": {
        "subcommand": "serve_workload",
        "metric": "output_throughput",
        "direction": "maximize",
    },
    "agentic": {
        "subcommand": "serve",
        "metric": "mean_ttft_ms",
        "direction": "minimize",
    },
    "creative": {
        "subcommand": "serve",
        "metric": "median_tpot_ms",
        "direction": "minimize",
    },
}


def _max_proven_context(limits: dict, kv_dtype: str | None = None) -> int:
    """Find the largest context length with any working config."""
    best = 0
    for ctx_str, dtypes in limits.get("limits", {}).items():
        ctx = int(ctx_str)
        if kv_dtype:
            if dtypes.get(kv_dtype) is not None:
                best = max(best, ctx)
        else:
            if any(v is not None for v in dtypes.values()):
                best = max(best, ctx)
    return best


def _working_kv_dtypes(limits: dict, context_len: int) -> list[str]:
    """Return kv-dtypes that work at a given context length."""
    entry = limits.get("limits", {}).get(str(context_len), {})
    return [k for k, v in entry.items() if v is not None]


def generate_sweep_params(
    *,
    profile: str,
    limits: dict,
    output_dir: Path,
) -> tuple[Path, Path, str]:
    """Generate serve_params.json and bench_params.json for a profile.

    Returns (serve_params_path, bench_params_path, subcommand).
    """
    is_moe = limits.get("is_moe", False)
    gpu_mem_util = limits.get("gpu_memory_utilization", 0.958)
    subcommand = PROFILES[profile]["subcommand"]

    serve: dict[str, dict] = {}
    bench: dict[str, dict] = {}

    if profile == "interactive":
        # Sweep context lengths x kv-dtype
        for ctx_str, dtypes in limits.get("limits", {}).items():
            ctx = int(ctx_str)
            for kv, slots in dtypes.items():
                if slots is None:
                    continue
                name = f"ctx{ctx // 1024}k-kv_{kv}"
                serve[name] = {
                    "max_model_len": ctx,
                    "kv_cache_dtype": kv,
                    "enable_prefix_caching": True,
                    "gpu_memory_utilization": gpu_mem_util,
                }

        for il in [256, 512, 2048]:
            for ol in [128, 256]:
                bench[f"in{il}-out{ol}"] = {
                    "random_input_len": il,
                    "random_output_len": ol,
                    "max_concurrency": 1,
                    "num_prompts": 20,
                }

    elif profile == "batch":
        max_seqs_values = [8, 16, 32, 64]
        if is_moe:
            max_seqs_values = [s for s in max_seqs_values if s <= 32]
        bt_values = [1024, 2048, 4096]

        for seqs in max_seqs_values:
            for bt in bt_values:
                if bt < seqs:
                    continue
                for kv in ("auto", "fp8"):
                    name = f"seqs{seqs}-bt{bt}-kv_{kv}"
                    serve[name] = {
                        "max_num_seqs": seqs,
                        "max_num_batched_tokens": bt,
                        "kv_cache_dtype": kv,
                        "gpu_memory_utilization": gpu_mem_util,
                        "max_model_len": 4096,
                    }

        bench["batch_workload"] = {
            "random_input_len": 512,
            "random_output_len": 256,
            "num_prompts": 50,
        }

    elif profile == "agentic":
        max_ctx = _max_proven_context(limits)
        if max_ctx == 0:
            max_ctx = 4096  # fallback

        for kv in _working_kv_dtypes(limits, max_ctx):
            serve[f"kv_{kv}"] = {
                "max_model_len": max_ctx,
                "kv_cache_dtype": kv,
                "enable_prefix_caching": True,
                "gpu_memory_utilization": gpu_mem_util,
            }

        input_len = int(max_ctx * 0.75)
        bench["agentic"] = {
            "random_input_len": input_len,
            "random_output_len": 512,
            "max_concurrency": 1,
            "num_prompts": 10,
        }

    elif profile == "creative":
        max_ctx = _max_proven_context(limits)
        if max_ctx == 0:
            max_ctx = 4096

        for kv in _working_kv_dtypes(limits, max_ctx):
            for seqs in [1, 2, 4, 8]:
                serve[f"kv_{kv}-seqs{seqs}"] = {
                    "max_model_len": max_ctx,
                    "kv_cache_dtype": kv,
                    "max_num_seqs": seqs,
                    "gpu_memory_utilization": gpu_mem_util,
                }

        for ol in [1024, 2048, 4096]:
            bench[f"out{ol}"] = {
                "random_input_len": 256,
                "random_output_len": ol,
                "max_concurrency": 1,
                "num_prompts": 5,
            }

    serve_path = output_dir / "serve_params.json"
    bench_path = output_dir / "bench_params.json"
    serve_path.write_text(json.dumps(serve, indent=2))
    bench_path.write_text(json.dumps(bench, indent=2))

    return serve_path, bench_path, subcommand


# ── Winner selection ───────────────────────────────


def _load_summaries(experiment_dir: Path) -> list[dict]:
    rows = []
    for summary in sorted(experiment_dir.glob("**/summary.json")):
        with open(summary) as f:
            data = json.load(f)
        combo_name = summary.parent.name
        if isinstance(data, list):
            for run in data:
                run["_combo_dir"] = combo_name
                rows.append(run)
        else:
            data["_combo_dir"] = combo_name
            rows.append(data)
    return rows


def pick_winner(
    experiment_dir: Path, profile: str
) -> tuple[dict, list[dict]]:
    """Pick the best config from sweep results."""
    rows = _load_summaries(experiment_dir)
    if not rows:
        print(f"ERROR: No results in {experiment_dir}", file=sys.stderr)
        sys.exit(1)

    meta = PROFILES[profile]
    metric = meta["metric"]
    maximize = meta["direction"] == "maximize"

    combos: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        combos[r["_combo_dir"]].append(r)

    scored = []
    for name, runs in combos.items():
        valid = [r for r in runs if float(r.get("failed", 0)) == 0]
        if not valid:
            continue
        mean_val = sum(float(r[metric]) for r in valid) / len(valid)
        scored.append({
            "name": name,
            metric: round(mean_val, 1),
            "mean_ttft_ms": round(
                sum(float(r.get("mean_ttft_ms", 0)) for r in valid) / len(valid), 1
            ),
            "median_tpot_ms": round(
                sum(float(r.get("median_tpot_ms", 0)) for r in valid) / len(valid), 1
            ),
            "output_throughput": round(
                sum(float(r.get("output_throughput", 0)) for r in valid) / len(valid), 1
            ),
            "params": valid[0],
        })

    if not scored:
        print(f"ERROR: No valid results in {experiment_dir}", file=sys.stderr)
        sys.exit(1)

    scored.sort(key=lambda x: x[metric], reverse=maximize)
    return scored[0], scored[:3]
```

- [ ] **Step 4: Run tests**

Run: `cd ~/vx && uv run pytest tests/test_tune.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd ~/vx && git add -A && git commit -m "feat: tune module — sweep param generation and winner selection for 4 profiles"
```

---

### Task 6: Compare module

**Files:**
- Create: `src/vx/compare.py`
- Create: `tests/test_compare.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_compare.py
import json
from pathlib import Path

import pytest

from vx.compare import filter_models_for_workload, rank_models


@pytest.fixture
def two_models_limits(tmp_path):
    """Create limits for two models."""
    configs_dir = tmp_path / "configs" / "models"
    configs_dir.mkdir(parents=True)

    model_a = {
        "model_path": "/opt/vllm/models/prov/ModelA",
        "quant_method": "fp8",
        "architecture": "TestLM",
        "is_moe": False,
        "limits": {
            "4096": {"auto": 64, "fp8": 128},
            "8192": {"auto": 32, "fp8": 64},
            "16384": {"auto": None, "fp8": 16},
        },
    }
    model_b = {
        "model_path": "/opt/vllm/models/prov/ModelB",
        "quant_method": "fp8",
        "architecture": "TestLM",
        "is_moe": False,
        "limits": {
            "4096": {"auto": 32, "fp8": 64},
            "8192": {"auto": 8, "fp8": 16},
            "16384": {"auto": None, "fp8": None},
        },
    }
    return [model_a, model_b]


def test_filter_models_fits(two_models_limits):
    result = filter_models_for_workload(
        all_limits=two_models_limits,
        users=10,
        context=8192,
    )
    # ModelA: fp8 has 64 slots at 8k → fits 10 users
    # ModelB: fp8 has 16 slots at 8k → fits 10 users
    assert len(result) == 2


def test_filter_models_excludes(two_models_limits):
    result = filter_models_for_workload(
        all_limits=two_models_limits,
        users=50,
        context=8192,
    )
    # ModelA: 64 slots → fits
    # ModelB: 16 slots → doesn't fit
    assert len(result) == 1
    assert result[0]["model_path"].endswith("ModelA")


def test_filter_models_context_too_high(two_models_limits):
    result = filter_models_for_workload(
        all_limits=two_models_limits,
        users=1,
        context=32768,
    )
    assert len(result) == 0


def test_rank_models(two_models_limits):
    fits = filter_models_for_workload(
        all_limits=two_models_limits,
        users=5,
        context=8192,
    )
    ranked = rank_models(fits)
    # ModelA has more capacity (64 slots) → ranked first
    assert ranked[0]["model_path"].endswith("ModelA")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/vx && uv run pytest tests/test_compare.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vx.compare'`

- [ ] **Step 3: Write compare.py**

```python
# src/vx/compare.py
"""Compare models against a workload."""


def _best_slots(limits_entry: dict) -> tuple[int, str]:
    """Return (max_slots, kv_dtype) for a context level."""
    best_slots = 0
    best_dtype = "auto"
    for dtype, slots in limits_entry.items():
        if slots is not None and slots > best_slots:
            best_slots = slots
            best_dtype = dtype
    return best_slots, best_dtype


def _find_context_level(limits: dict, context: int) -> str | None:
    """Find the smallest context level >= requested context."""
    levels = sorted(int(k) for k in limits.get("limits", {}).keys())
    for level in levels:
        if level >= context:
            return str(level)
    return None


def filter_models_for_workload(
    *,
    all_limits: list[dict],
    users: int,
    context: int,
) -> list[dict]:
    """Filter models that can handle the workload."""
    result = []
    for lim in all_limits:
        ctx_key = _find_context_level(lim, context)
        if ctx_key is None:
            continue
        entry = lim.get("limits", {}).get(ctx_key, {})
        best_slots, best_dtype = _best_slots(entry)
        if best_slots >= users:
            result.append({
                **lim,
                "_matched_context": int(ctx_key),
                "_matched_slots": best_slots,
                "_matched_dtype": best_dtype,
            })
    return result


def rank_models(fits: list[dict]) -> list[dict]:
    """Rank fitting models by capacity (most slots first)."""
    return sorted(fits, key=lambda m: m["_matched_slots"], reverse=True)
```

- [ ] **Step 4: Run tests**

Run: `cd ~/vx && uv run pytest tests/test_compare.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd ~/vx && git add -A && git commit -m "feat: compare module — filter and rank models by workload requirements"
```

---

### Task 7: Serve module

**Files:**
- Create: `src/vx/serve.py`
- Create: `tests/test_serve.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_serve.py
import pytest

from vx.serve import start_vllm, stop_vllm, is_vllm_running


def test_is_vllm_running_active(mocker):
    mocker.patch("vx.serve._systemctl", return_value=(True, "active"))
    assert is_vllm_running() is True


def test_is_vllm_running_inactive(mocker):
    mocker.patch("vx.serve._systemctl", return_value=(True, "inactive"))
    assert is_vllm_running() is False


def test_stop_vllm(mocker):
    mock_run = mocker.patch("vx.serve._systemctl", return_value=(True, ""))
    stop_vllm()
    mock_run.assert_called_once_with("stop")


def test_start_vllm_with_config(mocker, tmp_path):
    config_file = tmp_path / "test.yaml"
    config_file.write_text("model: /opt/vllm/models/test\nport: 8888\n")

    mock_symlink = mocker.patch("vx.serve._update_active_symlink")
    mock_systemctl = mocker.patch("vx.serve._systemctl", return_value=(True, ""))

    start_vllm(config_path=config_file)
    mock_symlink.assert_called_once_with(config_file)
    assert mock_systemctl.call_count == 1
    mock_systemctl.assert_called_with("start")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/vx && uv run pytest tests/test_serve.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vx.serve'`

- [ ] **Step 3: Write serve.py**

```python
# src/vx/serve.py
"""Start/stop vLLM via systemd."""

import subprocess
from pathlib import Path

from vx.config import active_yaml_path


def _systemctl(action: str) -> tuple[bool, str]:
    """Run systemctl action on the vllm service."""
    result = subprocess.run(
        ["sudo", "systemctl", action, "vllm"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stdout.strip()


def _update_active_symlink(config_path: Path) -> None:
    active = active_yaml_path()
    active.unlink(missing_ok=True)
    active.symlink_to(config_path.resolve())


def is_vllm_running() -> bool:
    ok, output = _systemctl("is-active")
    return ok and output == "active"


def start_vllm(config_path: Path) -> None:
    _update_active_symlink(config_path)
    _systemctl("start")


def stop_vllm() -> None:
    _systemctl("stop")
```

- [ ] **Step 4: Run tests**

Run: `cd ~/vx && uv run pytest tests/test_serve.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd ~/vx && git add -A && git commit -m "feat: serve module — start/stop vLLM via systemd with active symlink"
```

---

### Task 8: CLI — typer app with all subcommands

**Files:**
- Create: `src/vx/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py
from typer.testing import CliRunner

from vx.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "vLLM model manager" in result.output


def test_models_command_exists():
    result = runner.invoke(app, ["models", "--help"])
    assert result.exit_code == 0
    assert "List downloaded models" in result.output


def test_download_command_exists():
    result = runner.invoke(app, ["download", "--help"])
    assert result.exit_code == 0


def test_tune_command_exists():
    result = runner.invoke(app, ["tune", "--help"])
    assert result.exit_code == 0


def test_start_command_exists():
    result = runner.invoke(app, ["start", "--help"])
    assert result.exit_code == 0


def test_stop_command_exists():
    result = runner.invoke(app, ["stop", "--help"])
    assert result.exit_code == 0


def test_compare_command_exists():
    result = runner.invoke(app, ["compare", "--help"])
    assert result.exit_code == 0


def test_models_runs(mocker):
    """vx models should call scan_models and print a table."""
    from vx.models import ModelInfo
    from pathlib import Path

    fake = ModelInfo(
        path=Path("/opt/vllm/models/test/Model-7B"),
        provider="test",
        model_name="Model-7B",
        architecture="TestLM",
        model_type="test",
        quant_method="fp8",
        max_position_embeddings=32768,
        is_moe=False,
        model_size_gb=7.2,
    )
    mocker.patch("vx.cli.scan_models", return_value=[fake])
    mocker.patch("vx.cli.read_limits", return_value=None)

    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "Model-7B" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/vx && uv run pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vx.cli'`

- [ ] **Step 3: Write cli.py**

```python
# src/vx/cli.py
"""vx — vLLM model manager CLI."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from vx.config import (
    MODELS_DIR,
    CONFIGS_DIR,
    limits_path,
    profile_path,
    read_limits,
    read_profile_yaml,
)
from vx.models import scan_models, fuzzy_match, ModelInfo

app = typer.Typer(help="vLLM model manager")
console = Console()

PROFILE_NAMES = ["interactive", "batch", "agentic", "creative"]


def _resolve_model(query: str) -> ModelInfo:
    """Resolve a fuzzy model query to exactly one ModelInfo."""
    models = scan_models(MODELS_DIR)
    if not models:
        console.print("[red]No models found.[/red] Run: vx download")
        raise typer.Exit(1)
    matches = fuzzy_match(query, models)
    if len(matches) == 0:
        console.print(f"[red]No model matching '{query}'[/red]")
        console.print("Available models:")
        for m in models:
            console.print(f"  {m.full_name}")
        raise typer.Exit(1)
    if len(matches) > 1:
        console.print(f"[yellow]'{query}' matches multiple models:[/yellow]")
        for m in matches:
            console.print(f"  {m.full_name}")
        raise typer.Exit(1)
    return matches[0]


@app.callback(invoke_without_command=True)
def dashboard(ctx: typer.Context):
    """Show status dashboard when called with no subcommand."""
    if ctx.invoked_subcommand is not None:
        return

    try:
        from vx.gpu import get_gpu_info
        gpu = get_gpu_info()
        gpu_line = f"{gpu.name} ({gpu.vram_total_gb:.0f} GB, CUDA {gpu.cuda})"
    except Exception:
        gpu_line = "[dim]unavailable[/dim]"

    models = scan_models(MODELS_DIR)
    probed = sum(
        1 for m in models if read_limits(limits_path(m.provider, m.model_name))
    )

    lines = [
        f"  [bold]GPU[/bold]       {gpu_line}",
        f"  [bold]Models[/bold]    {len(models)} downloaded, {probed} probed",
        "",
        "  [bold]Commands[/bold]",
        "    download    Search & download from HuggingFace",
        "    models      List models with limits & benchmarks",
        "    tune        Probe limits + benchmark a model",
        "    start       Start serving",
        "    stop        Stop serving",
        "    compare     Find the best model for a workload",
    ]
    console.print(Panel("\n".join(lines), title="[bold cyan]vLLM[/bold cyan]", border_style="cyan"))


@app.command()
def models(model: str = typer.Argument(None, help="Model name for detail view")):
    """List downloaded models."""
    all_models = scan_models(MODELS_DIR)
    if not all_models:
        console.print("[dim]No models found.[/dim] Run: vx download")
        return

    if model:
        m = _resolve_model(model)
        _show_model_detail(m)
        return

    table = Table(title="Downloaded Models")
    table.add_column("Model", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Probed")
    table.add_column("Tuned")
    table.add_column("Max Context", justify="right")

    for m in all_models:
        lim = read_limits(limits_path(m.provider, m.model_name))
        probed = "✓" if lim else "✗"

        tuned_profiles = [
            p for p in PROFILE_NAMES
            if profile_path(m.provider, m.model_name, p).exists()
        ]
        tuned = ", ".join(tuned_profiles) if tuned_profiles else "✗"

        max_ctx = "—"
        if lim:
            for ctx_str in sorted(lim["limits"].keys(), key=int, reverse=True):
                entry = lim["limits"][ctx_str]
                if any(v is not None for v in entry.values()):
                    max_ctx = f"{int(ctx_str) // 1024}k"
                    break

        table.add_row(m.full_name, f"{m.model_size_gb} GB", probed, tuned, max_ctx)

    console.print(table)


def _show_model_detail(m: ModelInfo):
    """Show detailed info for a single model including limits table."""
    lim = read_limits(limits_path(m.provider, m.model_name))

    console.print(f"\n[bold]{m.full_name}[/bold]")
    console.print(f"  Arch: {m.architecture}  Quant: {m.quant_method or 'none'}  MoE: {m.is_moe}")
    console.print(f"  Size: {m.model_size_gb} GB  Max positions: {m.max_position_embeddings}\n")

    if not lim:
        console.print("  [dim]Not probed yet.[/dim] Run: vx tune {}\n".format(m.model_name.lower()))
        return

    table = Table(title="Context / Concurrency Limits")
    table.add_column("Context", justify="right")
    table.add_column("kv auto", justify="right")
    table.add_column("kv fp8", justify="right")

    for ctx_str in sorted(lim["limits"].keys(), key=int):
        entry = lim["limits"][ctx_str]
        auto = str(entry.get("auto", "—")) if entry.get("auto") is not None else "OOM"
        fp8 = str(entry.get("fp8", "—")) if entry.get("fp8") is not None else "OOM"
        ctx_display = f"{int(ctx_str) // 1024}k"
        table.add_row(ctx_display, auto, fp8)

    console.print(table)

    tuned = [
        p for p in PROFILE_NAMES
        if profile_path(m.provider, m.model_name, p).exists()
    ]
    if tuned:
        console.print(f"\n  Tuned profiles: {', '.join(tuned)}")
    else:
        console.print("\n  [dim]No profiles tuned yet.[/dim]")
    console.print()


@app.command()
def download(model_id: str = typer.Argument(None, help="HuggingFace model ID")):
    """Search & download a model from HuggingFace."""
    console.print("[dim]Download not yet implemented.[/dim]")


@app.command()
def tune(
    model: str = typer.Argument(None, help="Model name (fuzzy match)"),
    profile: str = typer.Argument(None, help="Profile: interactive, batch, agentic, creative, all"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show commands without running"),
    reprobe: bool = typer.Option(False, "--reprobe", help="Force re-probe even if cached"),
    no_probe: bool = typer.Option(False, "--no-probe", help="Skip probe phase"),
):
    """Probe limits + benchmark a model."""
    console.print("[dim]Tune not yet implemented.[/dim]")


@app.command()
def start(
    model: str = typer.Argument(None, help="Model name (fuzzy match)"),
    profile: str = typer.Option("interactive", "--profile", "-p", help="Profile to use"),
):
    """Start serving a model."""
    console.print("[dim]Start not yet implemented.[/dim]")


@app.command()
def stop():
    """Stop the vLLM service."""
    console.print("[dim]Stop not yet implemented.[/dim]")


@app.command()
def compare(
    users: int = typer.Option(None, "--users", "-u", help="Number of concurrent users"),
    use_case: str = typer.Option(None, "--use-case", help="interactive, batch, agentic, creative"),
    context: int = typer.Option(None, "--context", "-c", help="Required context length in tokens"),
):
    """Find the best model for a workload."""
    console.print("[dim]Compare not yet implemented.[/dim]")
```

- [ ] **Step 4: Run tests**

Run: `cd ~/vx && uv run pytest tests/test_cli.py -v`
Expected: all PASS

- [ ] **Step 5: Verify the CLI is callable**

Run: `cd ~/vx && uv run vx --help`
Expected: shows the help text with all subcommands listed

- [ ] **Step 6: Commit**

```bash
cd ~/vx && git add -A && git commit -m "feat: typer CLI with all subcommands — models, download, tune, start, stop, compare"
```

---

### Task 9: Wire tune command end-to-end

**Files:**
- Modify: `src/vx/cli.py` (the `tune` function)

This task wires the `tune` command to actually call probe + benchmark. It uses the probe and tune modules built in Tasks 4-5.

- [ ] **Step 1: Write the integration test**

```python
# tests/test_cli_tune.py
from typer.testing import CliRunner

from vx.cli import app

runner = CliRunner()


def test_tune_dry_run(mocker, fake_model_dir):
    """Dry run should print sweep command without running."""
    mocker.patch("vx.cli.MODELS_DIR", fake_model_dir.parent.parent)
    mocker.patch("vx.cli.scan_models", return_value=[])

    # Since no models will match, test the error path
    result = runner.invoke(app, ["tune", "nonexistent", "interactive", "--dry-run"])
    assert result.exit_code == 1
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `cd ~/vx && uv run pytest tests/test_cli_tune.py -v`

- [ ] **Step 3: Implement the tune command body in cli.py**

Replace the `tune` function stub in `src/vx/cli.py`:

```python
@app.command()
def tune(
    model: str = typer.Argument(None, help="Model name (fuzzy match)"),
    profile: str = typer.Argument(None, help="Profile: interactive, batch, agentic, creative, all"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show commands without running"),
    reprobe: bool = typer.Option(False, "--reprobe", help="Force re-probe even if cached"),
    no_probe: bool = typer.Option(False, "--no-probe", help="Skip probe phase"),
    all_models: bool = typer.Option(False, "--all", help="Tune all downloaded models"),
):
    """Probe limits + benchmark a model."""
    from vx.gpu import get_gpu_info, compute_gpu_memory_utilization
    from vx.config import BENCHMARKS_DIR, VLLM_BIN, write_limits, write_profile_yaml
    from vx.probe import probe_model
    from vx.tune import generate_sweep_params, pick_winner, PROFILES
    import subprocess
    from datetime import datetime

    if all_models:
        models_to_tune = scan_models(MODELS_DIR)
    elif model:
        models_to_tune = [_resolve_model(model)]
    else:
        # Interactive: pick model
        all_m = scan_models(MODELS_DIR)
        if not all_m:
            console.print("[red]No models found.[/red]")
            raise typer.Exit(1)
        for i, m in enumerate(all_m):
            console.print(f"  {i + 1}) {m.full_name}")
        idx = typer.prompt("Which model?", type=int) - 1
        models_to_tune = [all_m[idx]]

    if not profile:
        # Interactive: pick profile
        choices = list(PROFILES.keys()) + ["all", "skip"]
        console.print("\n[bold]What's this model for?[/bold]")
        for i, c in enumerate(choices):
            labels = {
                "interactive": "Interactive (chat, coding, RAG)",
                "batch": "Batch processing",
                "agentic": "Agentic workflows",
                "creative": "Creative writing",
                "all": "All profiles",
                "skip": "Skip benchmarks (probe only)",
            }
            console.print(f"  {i + 1}) {labels.get(c, c)}")
        idx = typer.prompt("Choice", type=int) - 1
        profile = choices[idx]

    gpu = get_gpu_info()
    gpu_mem_util = compute_gpu_memory_utilization(gpu.vram_total_gb)

    profiles_to_run = list(PROFILES.keys()) if profile == "all" else ([] if profile == "skip" else [profile])

    for m in models_to_tune:
        console.print(f"\n[bold]═══ {m.full_name} ═══[/bold]\n")

        # Phase 1: Probe
        lim_path = limits_path(m.provider, m.model_name)
        existing = read_limits(lim_path)

        if existing and not reprobe and not no_probe:
            console.print("[dim]Limits already probed (use --reprobe to redo)[/dim]\n")
            limits_data = existing
        elif no_probe and existing:
            limits_data = existing
        elif no_probe:
            console.print("[red]No probe data and --no-probe specified.[/red]")
            raise typer.Exit(1)
        else:
            console.print("[bold]Phase 1: Probing limits...[/bold]\n")
            if dry_run:
                console.print("[dim]DRY RUN: would probe context/concurrency limits[/dim]\n")
                limits_data = {"limits": {}, "gpu_memory_utilization": gpu_mem_util}
            else:
                limits_data = probe_model(
                    model_info=m,
                    gpu_mem_util=gpu_mem_util,
                    vram_total_gb=gpu.vram_total_gb,
                )
                write_limits(lim_path, limits_data)
                console.print("[green]✓ Limits saved[/green]\n")

        # Phase 2: Benchmark
        for prof in profiles_to_run:
            console.print(f"[bold]Phase 2: {prof} benchmark[/bold]\n")

            timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M")
            exp_name = f"{prof}-{timestamp}"
            exp_dir = BENCHMARKS_DIR / m.model_name / exp_name
            exp_dir.mkdir(parents=True, exist_ok=True)

            serve_p, bench_p, subcommand = generate_sweep_params(
                profile=prof,
                limits=limits_data,
                output_dir=exp_dir,
            )

            base_serve = f"{VLLM_BIN} serve {m.path} --dtype bfloat16 --trust-remote-code"
            qf = __import__("vx.models", fromlist=["quant_flag"]).quant_flag(m.quant_method)
            if qf:
                base_serve += f" {qf}"

            sweep_cmd = [
                str(VLLM_BIN), "bench", "sweep", subcommand,
                "--serve-cmd", base_serve,
                "--bench-cmd", f"{VLLM_BIN} bench serve --dataset-name random",
                "--serve-params", str(serve_p),
                "--bench-params", str(bench_p),
                "--num-runs", "3",
                "--server-ready-timeout", "600",
                "--output-dir", str(BENCHMARKS_DIR / m.model_name),
                "--experiment-name", exp_name,
            ]

            if dry_run:
                console.print(f"[dim]DRY RUN: {' '.join(sweep_cmd)}[/dim]\n")
                continue

            # Run with retry
            for attempt in range(3):
                console.print(f"Running sweep (attempt {attempt + 1}/3)...")
                cmd = sweep_cmd + (["--resume"] if attempt > 0 else [])
                result = subprocess.run(cmd)
                if result.returncode == 0:
                    break
                if attempt < 2:
                    console.print("[yellow]Retrying with --resume...[/yellow]")

            # Pick winner
            try:
                winner, top3 = pick_winner(exp_dir, profile=prof)
                config_data = {
                    "model": str(m.path),
                    "host": "0.0.0.0",
                    "port": 8888,
                    "dtype": "bfloat16",
                    "trust-remote-code": True,
                    "gpu-memory-utilization": gpu_mem_util,
                }
                for key in ["kv_cache_dtype", "enable_prefix_caching", "max_model_len", "max_num_seqs", "max_num_batched_tokens"]:
                    val = winner["params"].get(key)
                    if val is not None:
                        config_data[key.replace("_", "-")] = val

                comment = f"Auto-generated by vx tune — {prof} profile\nBenchmark: {exp_dir}"
                cfg_path = profile_path(m.provider, m.model_name, prof)
                write_profile_yaml(cfg_path, config_data, comment=comment)
                console.print(f"[green]✓ Config written: {cfg_path}[/green]\n")
            except SystemExit:
                console.print(f"[yellow]No results for {prof} profile.[/yellow]\n")

    console.print("[bold green]Done.[/bold green]")
```

- [ ] **Step 4: Run all tests**

Run: `cd ~/vx && uv run pytest -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
cd ~/vx && git add -A && git commit -m "feat: wire tune command — probe + benchmark + config generation end-to-end"
```

---

### Task 10: Install and verify

**Files:** None (verification only)

- [ ] **Step 1: Install in dev mode**

```bash
cd ~/vx && uv pip install -e ".[dev]"
```

- [ ] **Step 2: Verify vx is on PATH**

```bash
which vx && vx --help
```

Expected: shows help with all subcommands

- [ ] **Step 3: Test vx against real models**

```bash
vx models
```

Expected: table showing 3 downloaded models

- [ ] **Step 4: Test model detail view**

```bash
vx models nvfp4
```

Expected: shows model info (not probed yet since limits.json is in old format)

- [ ] **Step 5: Test dry-run tune**

```bash
vx tune nvfp4 interactive --dry-run
```

Expected: shows probe and sweep commands without executing

- [ ] **Step 6: Symlink to /usr/local/bin**

```bash
sudo ln -sf $(cd ~/vx && uv run which vx) /usr/local/bin/vx
```

- [ ] **Step 7: Run full test suite**

```bash
cd ~/vx && uv run pytest -v --tb=short
```

Expected: all tests pass

- [ ] **Step 8: Commit any fixes**

```bash
cd ~/vx && git add -A && git commit -m "chore: installation verification and fixes"
```
