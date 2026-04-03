# Multi-Backend Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Backend Protocol abstraction and llama.cpp support to vserve, transforming it from vLLM-only to multi-backend.

**Architecture:** Backend Protocol defines what every inference engine must implement. VllmBackend wraps existing modules (serve.py, probe.py, tools.py). LlamaCppBackend is new. Registry auto-detects backend from model format (GGUF → llama.cpp, safetensors → vLLM). CLI delegates to backend methods.

**Tech Stack:** Python 3.12+, Typer, Rich, huggingface_hub, gguf (new optional dep), pytest, ruff, mypy

**Spec:** `docs/superpowers/specs/2026-04-03-multi-backend-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/vserve/backends/__init__.py` | Create | Registry: `register()`, `get_backend()`, `get_backend_by_name()`, `available_backends()` |
| `src/vserve/backends/protocol.py` | Create | `Backend` Protocol class |
| `src/vserve/backends/vllm.py` | Create | `VllmBackend` — wraps serve.py, probe.py, tools.py |
| `src/vserve/backends/llamacpp.py` | Create | `LlamaCppBackend` — GGUF metadata, layer offload, llama-server config |
| `src/vserve/models.py` | Modify | Add GGUF model detection, keep `QUANT_FLAGS` as re-export |
| `src/vserve/config.py` | Modify | Add `llamacpp_root`, `llamacpp_service_name`, `llamacpp_service_user` |
| `src/vserve/cli.py` | Modify | Use backend protocol for start/stop/tune/status/doctor/models/init |
| `pyproject.toml` | Modify | Add `[project.optional-dependencies] llamacpp = ["gguf>=0.6"]` |
| `tests/test_backends.py` | Create | Registry, auto-detection, VllmBackend delegation |
| `tests/test_llamacpp.py` | Create | LlamaCppBackend tune, build_config, can_serve |
| `tests/conftest.py` | Modify | Add `fake_gguf_model_dir` fixture |

---

### Task 1: Backend Protocol

**Files:**
- Create: `src/vserve/backends/__init__.py`
- Create: `src/vserve/backends/protocol.py`
- Test: `tests/test_backends.py`

- [ ] **Step 1: Write the failing test for the registry**

```python
# tests/test_backends.py
"""Tests for the backend registry and protocol."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from vserve.backends import register, get_backend, get_backend_by_name, available_backends
from vserve.backends.protocol import Backend


def _make_mock_backend(name: str, can_serve_result: bool = True, has_binary: bool = True) -> Mock:
    b = Mock(spec=Backend)
    b.name = name
    b.display_name = name
    b.can_serve.return_value = can_serve_result
    b.find_entrypoint.return_value = Path("/fake/bin") if has_binary else None
    return b


def test_register_and_get_backend_by_name():
    from vserve.backends import _BACKENDS
    _BACKENDS.clear()

    b = _make_mock_backend("test")
    register(b)
    assert get_backend_by_name("test") is b


def test_get_backend_by_name_unknown():
    from vserve.backends import _BACKENDS
    _BACKENDS.clear()

    with pytest.raises(KeyError):
        get_backend_by_name("nonexistent")


def test_get_backend_auto_detect():
    from vserve.backends import _BACKENDS
    _BACKENDS.clear()

    b1 = _make_mock_backend("a", can_serve_result=False)
    b2 = _make_mock_backend("b", can_serve_result=True)
    register(b1)
    register(b2)

    model = Mock()
    assert get_backend(model) is b2


def test_get_backend_no_match():
    from vserve.backends import _BACKENDS
    _BACKENDS.clear()

    b = _make_mock_backend("a", can_serve_result=False)
    register(b)

    model = Mock()
    with pytest.raises(ValueError, match="No backend"):
        get_backend(model)


def test_available_backends_filters_missing():
    from vserve.backends import _BACKENDS
    _BACKENDS.clear()

    b1 = _make_mock_backend("installed", has_binary=True)
    b2 = _make_mock_backend("missing", has_binary=False)
    register(b1)
    register(b2)

    result = available_backends()
    assert len(result) == 1
    assert result[0].name == "installed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backends.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vserve.backends'`

- [ ] **Step 3: Create the protocol module**

```python
# src/vserve/backends/protocol.py
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
    service_name: str
    service_user: str
    root_dir: Path

    def can_serve(self, model: ModelInfo) -> bool:
        """Can this backend serve this model? (format check)"""
        ...

    def find_entrypoint(self) -> Path | str | None:
        """Locate the server binary or launch command. None = not installed."""
        ...

    def tune(self, model: ModelInfo, gpu: GpuInfo) -> dict:
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
```

- [ ] **Step 4: Create the registry module**

```python
# src/vserve/backends/__init__.py
"""Backend registry — auto-detect and manage inference backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vserve.backends.protocol import Backend
    from vserve.models import ModelInfo

_BACKENDS: list[Backend] = []


def register(backend: Backend) -> None:
    """Register a backend with the registry."""
    _BACKENDS.append(backend)


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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_backends.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All existing tests still pass

- [ ] **Step 7: Lint**

Run: `uv run ruff check src/vserve/backends/`
Expected: No errors

- [ ] **Step 8: Commit**

```bash
git add src/vserve/backends/__init__.py src/vserve/backends/protocol.py tests/test_backends.py
git commit -m "feat: add Backend Protocol and registry"
```

---

### Task 2: VllmBackend — Wrap Existing Logic

**Files:**
- Create: `src/vserve/backends/vllm.py`
- Modify: `src/vserve/models.py` (re-export after moving QUANT_FLAGS)
- Test: `tests/test_backends.py` (add VllmBackend tests)

- [ ] **Step 1: Write failing tests for VllmBackend**

Add to `tests/test_backends.py`:

```python
from vserve.backends.vllm import VllmBackend


class TestVllmBackend:
    def test_identity(self):
        b = VllmBackend()
        assert b.name == "vllm"
        assert b.display_name == "vLLM"

    def test_can_serve_safetensors(self, fake_model_dir):
        b = VllmBackend()
        from vserve.models import detect_model
        m = detect_model(fake_model_dir)
        assert b.can_serve(m) is True

    def test_cannot_serve_gguf(self, tmp_path):
        b = VllmBackend()
        model_dir = tmp_path / "models" / "user" / "Model-GGUF"
        model_dir.mkdir(parents=True)
        (model_dir / "model-Q4_K_M.gguf").write_bytes(b"\0" * 100)

        from vserve.models import ModelInfo
        m = ModelInfo(
            path=model_dir, provider="user", model_name="Model-GGUF",
            architecture="unknown", model_type="unknown", quant_method=None,
            max_position_embeddings=4096, is_moe=False, model_size_gb=0.1,
        )
        assert b.can_serve(m) is False

    def test_health_url(self):
        b = VllmBackend()
        assert b.health_url(8888) == "http://localhost:8888/health"

    def test_quant_flag(self):
        b = VllmBackend()
        assert "gptq_marlin" in b.quant_flag("gptq")
        assert "fp8" in b.quant_flag("fp8")
        assert b.quant_flag(None) == ""
        assert b.quant_flag("compressed-tensors") == ""

    def test_tune_delegates_to_probe(self, fake_model_dir, mocker):
        b = VllmBackend()
        from vserve.models import detect_model
        m = detect_model(fake_model_dir)

        mock_gpu = Mock()
        mock_gpu.vram_total_gb = 48.0

        result = b.tune(m, mock_gpu, gpu_mem_util=0.90)
        assert "limits" in result
        assert "tool_call_parser" in result

    def test_detect_tools(self, fake_model_dir):
        b = VllmBackend()
        result = b.detect_tools(fake_model_dir)
        assert "tool_call_parser" in result
        assert "reasoning_parser" in result

    def test_start_stop_delegates(self, mocker, tmp_path):
        b = VllmBackend()
        mock_start = mocker.patch("vserve.serve.start_vllm")
        mock_stop = mocker.patch("vserve.serve.stop_vllm")
        mock_running = mocker.patch("vserve.serve.is_vllm_running", return_value=True)

        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text("model: test\n")
        b.start(cfg_path)
        mock_start.assert_called_once_with(cfg_path)

        b.stop()
        mock_stop.assert_called_once()

        assert b.is_running() is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backends.py::TestVllmBackend -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vserve.backends.vllm'`

- [ ] **Step 3: Implement VllmBackend**

```python
# src/vserve/backends/vllm.py
"""vLLM backend — wraps existing serve/probe/tools modules."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from vserve.gpu import GpuInfo
    from vserve.models import ModelInfo


# Quantization flags for vLLM CLI
_QUANT_FLAGS = {
    "gptq": "--quantization gptq_marlin",
    "awq": "--quantization awq_marlin",
    "fp8": "--quantization fp8",
    "compressed-tensors": "",
    "none": "",
}


class VllmBackend:
    name = "vllm"
    display_name = "vLLM"
    service_name = "vllm"
    service_user = "vllm"

    @property
    def root_dir(self) -> Path:
        from vserve.config import cfg
        return cfg().vllm_root

    def can_serve(self, model: ModelInfo) -> bool:
        """True if model has safetensors or .bin weights (not GGUF-only)."""
        p = model.path
        has_safetensors = any(p.glob("*.safetensors"))
        has_bin = any(p.glob("*.bin"))
        has_gguf = any(p.glob("*.gguf"))
        # Serve if it has safetensors/bin, OR has config.json but no GGUF
        return (has_safetensors or has_bin) or ((p / "config.json").exists() and not has_gguf)

    def find_entrypoint(self) -> Path | None:
        from vserve.config import cfg
        vllm_bin = cfg().vllm_bin
        return vllm_bin if vllm_bin.exists() else None

    def tune(self, model: ModelInfo, gpu: GpuInfo, *, gpu_mem_util: float = 0.90) -> dict:
        from vserve.probe import calculate_limits
        return calculate_limits(model_info=model, vram_total_gb=gpu.vram_total_gb, gpu_mem_util=gpu_mem_util)

    def build_config(self, model: ModelInfo, choices: dict) -> dict:
        """Build vLLM serving config dict.

        Expected choices keys:
            context: int, kv_dtype: str, slots: int, batched_tokens: int | None,
            gpu_mem_util: float, tools: bool, tool_parser: str | None,
            reasoning_parser: str | None
        """
        cfg: dict = {
            "model": str(model.path),
            "host": "0.0.0.0",
            "port": choices.get("port", 8888),
            "dtype": "bfloat16",
            "trust-remote-code": True,
            "gpu-memory-utilization": choices["gpu_mem_util"],
            "max-model-len": choices["context"],
            "max-num-seqs": choices["slots"],
            "kv-cache-dtype": choices["kv_dtype"],
            "enable-prefix-caching": True,
        }
        bt = choices.get("batched_tokens")
        if bt is not None:
            cfg["max-num-batched-tokens"] = bt

        qf = self.quant_flag(model.quant_method)
        if qf:
            cfg["quantization"] = qf.split()[-1]

        if choices.get("tools") and choices.get("tool_parser"):
            cfg["enable-auto-tool-choice"] = True
            cfg["tool-call-parser"] = choices["tool_parser"]
            rp = choices.get("reasoning_parser")
            if rp:
                cfg["reasoning-parser"] = rp

        return cfg

    def quant_flag(self, method: str | None) -> str:
        if method is None:
            return ""
        return _QUANT_FLAGS.get(method, "")

    def start(self, config_path: Path) -> None:
        from vserve.serve import start_vllm
        start_vllm(config_path)

    def stop(self) -> None:
        from vserve.serve import stop_vllm
        stop_vllm()

    def is_running(self) -> bool:
        from vserve.serve import is_vllm_running
        return is_vllm_running()

    def health_url(self, port: int) -> str:
        return f"http://localhost:{port}/health"

    def detect_tools(self, model_path: Path) -> dict:
        from vserve.tools import detect_tool_parser, detect_reasoning_parser
        return {
            "tool_call_parser": detect_tool_parser(model_path),
            "reasoning_parser": detect_reasoning_parser(model_path),
        }

    def doctor_checks(self) -> list[tuple[str, Callable[[], bool]]]:
        def check_binary() -> bool:
            return self.find_entrypoint() is not None

        def check_service() -> bool:
            from pathlib import Path as _P
            return _P(f"/etc/systemd/system/{self.service_name}.service").exists()

        return [
            (f"{self.display_name} binary", check_binary),
            (f"{self.service_name}.service unit", check_service),
        ]
```

- [ ] **Step 4: Update models.py to re-export QUANT_FLAGS from vllm backend**

In `src/vserve/models.py`, keep `QUANT_FLAGS` and `quant_flag` in place (they still work, tests import them). No change needed — the vllm backend has its own copy in `_QUANT_FLAGS`. This avoids breaking existing imports.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_backends.py -v`
Expected: All tests PASS

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All existing tests still pass (no imports changed)

- [ ] **Step 7: Lint**

Run: `uv run ruff check src/vserve/backends/vllm.py`
Expected: No errors

- [ ] **Step 8: Commit**

```bash
git add src/vserve/backends/vllm.py tests/test_backends.py
git commit -m "feat: VllmBackend wrapping existing serve/probe/tools"
```

---

### Task 3: Config Changes for Multi-Backend

**Files:**
- Modify: `src/vserve/config.py`
- Test: `tests/test_config.py` (add tests for new fields)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_config.py`:

```python
def test_config_llamacpp_defaults():
    """New llamacpp fields have sensible defaults."""
    from vserve.config import _build_config
    from pathlib import Path

    cfg = _build_config(
        Path("/opt/vllm"), Path("/usr/local/cuda"), "vllm", "vllm", 8888,
    )
    assert cfg.llamacpp_root is None
    assert cfg.llamacpp_service_name == "llama-cpp"
    assert cfg.llamacpp_service_user == "llama-cpp"


def test_config_llamacpp_roundtrip(tmp_path, monkeypatch):
    """llamacpp fields survive save/load cycle."""
    from vserve.config import _build_config, save_config, load_config, CONFIG_FILE
    from pathlib import Path

    config_file = tmp_path / "config.yaml"
    monkeypatch.setattr("vserve.config.CONFIG_FILE", config_file)

    cfg = _build_config(
        Path("/opt/vllm"), Path("/usr/local/cuda"), "vllm", "vllm", 8888,
    )
    cfg.llamacpp_root = Path("/opt/llama-cpp")
    save_config(cfg)

    loaded = load_config()
    assert loaded.llamacpp_root == Path("/opt/llama-cpp")
    assert loaded.llamacpp_service_name == "llama-cpp"
    assert loaded.llamacpp_service_user == "llama-cpp"


def test_config_without_llamacpp_fields(tmp_path, monkeypatch):
    """Existing config files without llamacpp fields load fine."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("vllm_root: /opt/vllm\ncuda_home: /usr/local/cuda\n")
    monkeypatch.setattr("vserve.config.CONFIG_FILE", config_file)

    from vserve.config import load_config
    cfg = load_config()
    assert cfg.llamacpp_root is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py::test_config_llamacpp_defaults -v`
Expected: FAIL — `TypeError: _build_config() ... unexpected keyword argument` or missing attribute

- [ ] **Step 3: Add llamacpp fields to VserveConfig and update _build_config, load_config, save_config**

In `src/vserve/config.py`:

Add to `VserveConfig` dataclass (after `gpu_overhead_gb`):
```python
    llamacpp_root: Path | None = None
    llamacpp_service_name: str = "llama-cpp"
    llamacpp_service_user: str = "llama-cpp"
```

Update `_build_config` signature to accept optional llamacpp params:
```python
def _build_config(vllm_root: Path, cuda_home: Path, service_name: str,
                  service_user: str, port: int, *,
                  gpu_memory_utilization: float | None = None,
                  gpu_overhead_gb: float | None = None,
                  llamacpp_root: Path | None = None,
                  llamacpp_service_name: str = "llama-cpp",
                  llamacpp_service_user: str = "llama-cpp") -> VserveConfig:
    return VserveConfig(
        # ... existing fields ...
        llamacpp_root=llamacpp_root,
        llamacpp_service_name=llamacpp_service_name,
        llamacpp_service_user=llamacpp_service_user,
    )
```

Update `load_config()` to read new fields:
```python
        llamacpp_root_raw = data.get("llamacpp_root")
        return _build_config(
            # ... existing args ...
            llamacpp_root=Path(str(llamacpp_root_raw)) if llamacpp_root_raw else None,
            llamacpp_service_name=str(data.get("llamacpp_service_name", "llama-cpp")),
            llamacpp_service_user=str(data.get("llamacpp_service_user", "llama-cpp")),
        )
```

Update `save_config()` to write new fields:
```python
    if cfg.llamacpp_root is not None:
        data["llamacpp_root"] = str(cfg.llamacpp_root)
    if cfg.llamacpp_service_name != "llama-cpp":
        data["llamacpp_service_name"] = cfg.llamacpp_service_name
    if cfg.llamacpp_service_user != "llama-cpp":
        data["llamacpp_service_user"] = cfg.llamacpp_service_user
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: All PASS

- [ ] **Step 5: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/vserve/config.py tests/test_config.py
git commit -m "feat: add llamacpp config fields to VserveConfig"
```

---

### Task 4: GGUF Model Detection in models.py

**Files:**
- Modify: `src/vserve/models.py`
- Modify: `tests/conftest.py` (add `fake_gguf_model_dir` fixture)
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/conftest.py`:

```python
@pytest.fixture
def fake_gguf_model_dir(tmp_path: Path) -> Path:
    """Create a fake GGUF model directory."""
    model_dir = tmp_path / "models" / "bartowski" / "TestModel-8B-GGUF"
    model_dir.mkdir(parents=True)

    # Fake GGUF file (just needs to exist for detection; metadata tests mock GGUFReader)
    (model_dir / "TestModel-8B-Q4_K_M.gguf").write_bytes(b"\0" * 4096)

    # tokenizer_config.json for tool detection
    import json
    tok_config = {"chat_template": "{% if tools %}tools{% endif %}{{ messages }}"}
    (model_dir / "tokenizer_config.json").write_text(json.dumps(tok_config))

    return model_dir
```

Add to `tests/test_models.py`:

```python
def test_detect_gguf_model(fake_gguf_model_dir):
    from vserve.models import detect_model
    m = detect_model(fake_gguf_model_dir)
    assert m.provider == "bartowski"
    assert m.model_name == "TestModel-8B-GGUF"
    assert m.model_size_gb > 0
    assert m.is_gguf is True


def test_scan_models_finds_gguf(tmp_path, fake_gguf_model_dir):
    from vserve.models import scan_models
    models = scan_models(tmp_path / "models")
    gguf_models = [m for m in models if m.is_gguf]
    assert len(gguf_models) == 1


def test_safetensors_model_not_gguf(fake_model_dir):
    from vserve.models import detect_model
    m = detect_model(fake_model_dir)
    assert m.is_gguf is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py::test_detect_gguf_model -v`
Expected: FAIL — `AttributeError: 'ModelInfo' has no attribute 'is_gguf'` or detection fails

- [ ] **Step 3: Add is_gguf field and GGUF detection to models.py**

Add `is_gguf: bool = False` to `ModelInfo` dataclass.

Update `detect_model()` to handle GGUF models:

```python
def detect_model(model_dir: Path) -> ModelInfo:
    config_path = model_dir / "config.json"
    gguf_files = list(model_dir.glob("*.gguf"))

    if gguf_files and not config_path.exists():
        # GGUF-only model — no config.json
        size_bytes = sum(f.stat().st_size for f in gguf_files)
        size_gb = round(size_bytes / (1024**3), 1)
        return ModelInfo(
            path=model_dir,
            provider=model_dir.parent.name,
            model_name=model_dir.name,
            architecture="gguf",
            model_type="gguf",
            quant_method=None,
            max_position_embeddings=0,  # read from GGUF metadata at tune time
            is_moe=False,
            model_size_gb=size_gb,
            is_gguf=True,
        )

    # Existing config.json logic...
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {model_dir}")
    # ... rest unchanged, but set is_gguf=False in the return
```

Update `scan_models()` to also find GGUF-only directories (no `config.json` check when `.gguf` files present):

```python
def scan_models(models_root: Path) -> list[ModelInfo]:
    models: list[ModelInfo] = []
    if not models_root.exists():
        return models
    for provider_dir in sorted(models_root.iterdir()):
        if not provider_dir.is_dir():
            continue
        for model_dir in sorted(provider_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            has_config = (model_dir / "config.json").exists()
            has_gguf = any(model_dir.glob("*.gguf"))
            if not has_config and not has_gguf:
                continue
            try:
                models.append(detect_model(model_dir))
            except Exception:
                continue
    return models
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_models.py -v`
Expected: All PASS

- [ ] **Step 5: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/vserve/models.py tests/conftest.py tests/test_models.py
git commit -m "feat: detect GGUF models in scan_models and detect_model"
```

---

### Task 5: LlamaCppBackend Core

**Files:**
- Create: `src/vserve/backends/llamacpp.py`
- Create: `tests/test_llamacpp.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_llamacpp.py
"""Tests for the llama.cpp backend."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from vserve.backends.llamacpp import LlamaCppBackend


class TestLlamaCppCanServe:
    def test_can_serve_gguf(self, fake_gguf_model_dir):
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)
        assert b.can_serve(m) is True

    def test_cannot_serve_safetensors(self, fake_model_dir):
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_model_dir)
        assert b.can_serve(m) is False


class TestLlamaCppIdentity:
    def test_name(self):
        b = LlamaCppBackend()
        assert b.name == "llamacpp"
        assert b.display_name == "llama.cpp"
        assert b.service_name == "llama-cpp"
        assert b.service_user == "llama-cpp"


class TestLlamaCppHealthUrl:
    def test_health_url(self):
        b = LlamaCppBackend()
        assert b.health_url(8888) == "http://localhost:8888/health"


class TestLlamaCppBuildConfig:
    def test_basic_config(self, fake_gguf_model_dir):
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)

        choices = {
            "context": 8192,
            "n_gpu_layers": 35,
            "parallel": 4,
            "port": 8888,
            "tools": False,
        }
        cfg = b.build_config(m, choices)
        assert cfg["ctx-size"] == 8192
        assert cfg["n-gpu-layers"] == 35
        assert cfg["parallel"] == 4
        assert cfg["flash-attn"] is True
        assert cfg["cont-batching"] is True
        assert "jinja" not in cfg

    def test_config_with_tools(self, fake_gguf_model_dir):
        b = LlamaCppBackend()
        from vserve.models import detect_model
        m = detect_model(fake_gguf_model_dir)

        choices = {
            "context": 8192,
            "n_gpu_layers": 35,
            "parallel": 4,
            "port": 8888,
            "tools": True,
        }
        cfg = b.build_config(m, choices)
        assert cfg["jinja"] is True


class TestLlamaCppQuant:
    def test_quant_flag_always_empty(self):
        b = LlamaCppBackend()
        assert b.quant_flag("gptq") == ""
        assert b.quant_flag(None) == ""


class TestLlamaCppDetectTools:
    def test_detect_tools_with_template(self, fake_gguf_model_dir):
        b = LlamaCppBackend()
        result = b.detect_tools(fake_gguf_model_dir)
        assert "supports_tools" in result
        assert result["supports_tools"] is True

    def test_detect_tools_no_template(self, tmp_path):
        b = LlamaCppBackend()
        model_dir = tmp_path / "models" / "u" / "m"
        model_dir.mkdir(parents=True)
        (model_dir / "model.gguf").write_bytes(b"\0" * 100)
        result = b.detect_tools(model_dir)
        assert result["supports_tools"] is False


class TestLlamaCppLifecycle:
    def test_start_calls_systemctl(self, mocker, tmp_path):
        b = LlamaCppBackend()
        mock_run = mocker.patch("subprocess.run", return_value=Mock(returncode=0, stdout="", stderr=""))

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text('{"model": "test"}')

        # Mock the active config path
        active = tmp_path / "active.json"
        mocker.patch.object(b, "_active_config_path", return_value=active)

        b.start(cfg_path)
        # Verify systemctl start was called
        calls = [c for c in mock_run.call_args_list if "systemctl" in str(c)]
        assert len(calls) >= 1

    def test_is_running(self, mocker):
        b = LlamaCppBackend()
        mocker.patch("subprocess.run", return_value=Mock(returncode=0, stdout="active", stderr=""))
        assert b.is_running() is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llamacpp.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vserve.backends.llamacpp'`

- [ ] **Step 3: Implement LlamaCppBackend**

```python
# src/vserve/backends/llamacpp.py
"""llama.cpp backend — serves GGUF models via llama-server."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from vserve.gpu import GpuInfo
    from vserve.models import ModelInfo


class LlamaCppBackend:
    name = "llamacpp"
    display_name = "llama.cpp"
    service_name = "llama-cpp"
    service_user = "llama-cpp"

    @property
    def root_dir(self) -> Path:
        from vserve.config import cfg
        root = cfg().llamacpp_root
        return root if root is not None else Path("/opt/llama-cpp")

    def can_serve(self, model: ModelInfo) -> bool:
        """True if model directory contains GGUF files."""
        return getattr(model, "is_gguf", False) or any(model.path.glob("*.gguf"))

    def find_entrypoint(self) -> Path | None:
        # Check configured root first
        candidate = self.root_dir / "bin" / "llama-server"
        if candidate.exists():
            return candidate
        # Fall back to PATH
        found = shutil.which("llama-server")
        return Path(found) if found else None

    def tune(self, model: ModelInfo, gpu: GpuInfo, *, gpu_mem_util: float = 0.90) -> dict:
        """Calculate n-gpu-layers, context sizes, and parallel slots."""
        gguf_files = sorted(model.path.glob("*.gguf"))
        if not gguf_files:
            raise ValueError(f"No GGUF files in {model.path}")

        model_size_bytes = sum(f.stat().st_size for f in gguf_files)
        model_size_gb = model_size_bytes / (1024**3)

        # Try to read metadata from GGUF
        metadata = self._read_gguf_metadata(gguf_files[0])
        num_layers = metadata.get("num_layers", 32)
        max_context = metadata.get("max_context", 4096)
        num_kv_heads = metadata.get("num_kv_heads", 8)
        head_dim = metadata.get("head_dim", 128)

        usable_gb = gpu.vram_total_gb * gpu_mem_util
        layer_size_gb = model_size_gb / num_layers

        # Calculate n-gpu-layers (with 10% buffer for embeddings/scratch)
        available_for_layers = usable_gb * 0.9
        n_gpu_layers = min(num_layers, int(available_for_layers / layer_size_gb))
        full_offload = n_gpu_layers >= num_layers

        # KV cache: FP16 by default
        # kv_per_token = 2 (K+V) * num_kv_heads * head_dim * num_layers * 2 (FP16)
        kv_per_token_bytes = 2 * num_kv_heads * head_dim * num_layers * 2

        # Available VRAM after model layers
        gpu_model_gb = n_gpu_layers * layer_size_gb
        remaining_gb = usable_gb - gpu_model_gb
        remaining_bytes = max(0, remaining_gb) * (1024**3)

        # Calculate limits per context step
        from vserve.probe import _context_steps
        steps = _context_steps(max_context)
        limits: dict[str, int | None] = {}
        for ctx in steps:
            total_kv_tokens = int(remaining_bytes // kv_per_token_bytes) if kv_per_token_bytes > 0 else 0
            parallel = total_kv_tokens // ctx if ctx > 0 else 0
            limits[str(ctx)] = parallel if parallel >= 1 else None

        # Tool detection
        supports_tools = self.detect_tools(model.path).get("supports_tools", False)

        from datetime import datetime, timezone
        return {
            "backend": "llamacpp",
            "model_path": str(model.path),
            "calculated_at": datetime.now(timezone.utc).isoformat(),
            "vram_total_gb": gpu.vram_total_gb,
            "gpu_memory_utilization": gpu_mem_util,
            "model_size_gb": round(model_size_gb, 1),
            "n_gpu_layers": n_gpu_layers,
            "num_layers": num_layers,
            "full_offload": full_offload,
            "max_context": max_context,
            "supports_tools": supports_tools,
            "limits": limits,
        }

    def _read_gguf_metadata(self, gguf_path: Path) -> dict:
        """Read model metadata from GGUF file header."""
        try:
            from gguf import GGUFReader  # type: ignore[import-untyped]
            reader = GGUFReader(str(gguf_path))

            # Find architecture name
            arch = "llama"
            arch_field = reader.fields.get("general.architecture")
            if arch_field is not None:
                arch = str(arch_field.parts[-1], "utf-8") if isinstance(arch_field.parts[-1], bytes) else str(arch_field.parts[-1])

            def _get_int(key: str, default: int) -> int:
                field = reader.fields.get(key)
                if field is not None:
                    return int(field.parts[-1])
                return default

            num_layers = _get_int(f"{arch}.block_count", 32)
            max_context = _get_int(f"{arch}.context_length", 4096)
            num_kv_heads = _get_int(f"{arch}.attention.head_count_kv", 8)
            num_attn_heads = _get_int(f"{arch}.attention.head_count", 32)
            embedding_length = _get_int(f"{arch}.embedding_length", 4096)
            head_dim = embedding_length // num_attn_heads if num_attn_heads else 128

            return {
                "arch": arch,
                "num_layers": num_layers,
                "max_context": max_context,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
            }
        except ImportError:
            # gguf package not installed — use defaults
            return {}
        except Exception:
            return {}

    def build_config(self, model: ModelInfo, choices: dict) -> dict:
        """Build llama-server JSON config."""
        gguf_files = sorted(model.path.glob("*.gguf"))
        model_file = str(gguf_files[0]) if gguf_files else str(model.path)

        cfg: dict = {
            "model": model_file,
            "host": "0.0.0.0",
            "port": choices.get("port", 8888),
            "ctx-size": choices["context"],
            "n-gpu-layers": choices["n_gpu_layers"],
            "parallel": choices.get("parallel", 1),
            "flash-attn": True,
            "cont-batching": True,
        }
        if choices.get("tools"):
            cfg["jinja"] = True
        return cfg

    def quant_flag(self, method: str | None) -> str:
        """llama.cpp quant is baked into GGUF — no CLI flag needed."""
        return ""

    def start(self, config_path: Path) -> None:
        """Update active config symlink/copy and start systemd service."""
        active = self._active_config_path()
        active.parent.mkdir(parents=True, exist_ok=True)
        # Copy JSON config (not symlink — llama-server reads JSON directly)
        import shutil
        shutil.copy2(config_path, active)

        result = subprocess.run(
            ["sudo", "systemctl", "start", self.service_name],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"systemctl start {self.service_name} failed: {result.stderr}")

    def stop(self) -> None:
        result = subprocess.run(
            ["sudo", "systemctl", "stop", self.service_name],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"systemctl stop {self.service_name} failed: {result.stderr}")

    def is_running(self) -> bool:
        result = subprocess.run(
            ["sudo", "systemctl", "is-active", self.service_name],
            capture_output=True, text=True,
        )
        return result.returncode == 0 and result.stdout.strip() == "active"

    def health_url(self, port: int) -> str:
        return f"http://localhost:{port}/health"

    def detect_tools(self, model_path: Path) -> dict:
        """Check if model supports tool calling via chat template."""
        from vserve.tools import supports_tools
        return {"supports_tools": supports_tools(model_path)}

    def doctor_checks(self) -> list[tuple[str, Callable[[], bool]]]:
        def check_binary() -> bool:
            return self.find_entrypoint() is not None

        def check_service() -> bool:
            return Path(f"/etc/systemd/system/{self.service_name}.service").exists()

        return [
            (f"{self.display_name} binary (llama-server)", check_binary),
            (f"{self.service_name}.service unit", check_service),
        ]

    def _active_config_path(self) -> Path:
        return self.root_dir / "configs" / "active.json"
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_llamacpp.py -v`
Expected: All PASS

- [ ] **Step 5: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 6: Lint**

Run: `uv run ruff check src/vserve/backends/llamacpp.py`
Expected: No errors

- [ ] **Step 7: Commit**

```bash
git add src/vserve/backends/llamacpp.py tests/test_llamacpp.py
git commit -m "feat: LlamaCppBackend — GGUF serving via llama-server"
```

---

### Task 6: Register Backends and Add Optional Dependency

**Files:**
- Modify: `src/vserve/backends/__init__.py` (auto-register on import)
- Modify: `pyproject.toml` (optional dependency)

- [ ] **Step 1: Write failing test**

Add to `tests/test_backends.py`:

```python
def test_default_backends_registered():
    """Importing backends registers vllm and llamacpp."""
    import importlib
    import vserve.backends
    importlib.reload(vserve.backends)

    from vserve.backends import _BACKENDS
    names = [b.name for b in _BACKENDS]
    assert "vllm" in names
    assert "llamacpp" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backends.py::test_default_backends_registered -v`
Expected: FAIL — backends not auto-registered

- [ ] **Step 3: Add auto-registration to backends/__init__.py**

Add to the end of `src/vserve/backends/__init__.py`:

```python
# Auto-register built-in backends on import
def _register_defaults() -> None:
    from vserve.backends.vllm import VllmBackend
    from vserve.backends.llamacpp import LlamaCppBackend
    register(VllmBackend())
    register(LlamaCppBackend())

_register_defaults()
```

- [ ] **Step 4: Fix test isolation**

The existing registry tests mutate `_BACKENDS`. Update them to save/restore:

```python
@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry before each test."""
    from vserve.backends import _BACKENDS
    saved = _BACKENDS.copy()
    _BACKENDS.clear()
    yield
    _BACKENDS.clear()
    _BACKENDS.extend(saved)
```

Add this fixture at the top of `tests/test_backends.py`. The `test_default_backends_registered` test should reload the module to trigger re-registration.

- [ ] **Step 5: Add optional dependency to pyproject.toml**

In `pyproject.toml`, add after the `[dependency-groups]` section:

```toml
[project.optional-dependencies]
llamacpp = ["gguf>=0.6"]
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_backends.py -v`
Expected: All PASS

- [ ] **Step 7: Run full suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 8: Commit**

```bash
git add src/vserve/backends/__init__.py pyproject.toml
git commit -m "feat: auto-register backends, add gguf optional dependency"
```

---

### Task 7: CLI Integration — start/stop with Backend Protocol

**Files:**
- Modify: `src/vserve/cli.py`

This is the largest task — refactoring start, stop, and the launch/health loop to use the backend protocol.

- [ ] **Step 1: Refactor _launch_vllm into generic _launch_backend**

Replace `_launch_vllm(cfg_path, label)` with `_launch_backend(backend, cfg_path, label)`:

```python
def _launch_backend(backend, cfg_path: "pathlib.Path", label: str) -> None:
    """Stop any running backend, start with given config, wait for health."""
    from vserve.config import read_profile_yaml
    import time
    import json

    lock = _lock_or_exit("gpu", f"starting {backend.display_name} ({label})")

    # Read config (YAML for vLLM, JSON for llama.cpp)
    if str(cfg_path).endswith(".json"):
        cfg = json.loads(cfg_path.read_text())
    else:
        cfg = read_profile_yaml(cfg_path) or {}

    try:
        _session_or_exit()

        if backend.is_running():
            console.print(f"[yellow]{backend.display_name} is already running.[/yellow]")
            if not typer.confirm("Stop and restart?", default=False):
                lock.release()
                return
            clear_session()
            try:
                backend.stop()
            except RuntimeError as e:
                console.print(f"[red]{e}[/red]")
            time.sleep(2)

        ctx = cfg.get("max-model-len") or cfg.get("ctx-size", "?")
        ctx_d = f"{ctx // 1024}k" if isinstance(ctx, int) else ctx
        console.print(f"\n[bold]Starting[/bold] with [cyan]{label}[/cyan] ({backend.display_name})")
        console.print(f"  context: {ctx_d}")

        try:
            backend.start(cfg_path)
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            console.print(f"  Check: sudo journalctl -u {backend.service_name} --no-pager -n 20")
            raise typer.Exit(1)

        port = cfg.get("port", 8888)
        health = backend.health_url(port)

        # Backend-specific first-run messages
        if backend.name == "vllm":
            from vserve.config import cfg as _cfg
            fi_cache = _cfg().vllm_root / ".cache" / "flashinfer"
            if not fi_cache.is_dir() or not list(fi_cache.glob("**/*.so")):
                console.print("  [yellow]First run — compiling kernels (~5-10 min).[/yellow]")

        console.print(f"  [dim]Waiting for {health} ...[/dim]")
        from urllib.request import urlopen
        for i in range(150):
            time.sleep(2)
            try:
                resp = urlopen(health, timeout=2)
                if resp.status == 200:
                    write_session(label)
                    console.print(f"\n[bold green]{backend.display_name} is running[/bold green] at http://localhost:{port}/v1")
                    console.print(f"  Config: {cfg_path}")
                    console.print(f"  Logs:   sudo journalctl -u {backend.service_name} -f\n")
                    return
            except Exception:
                pass
            if i > 5 and not backend.is_running():
                console.print(f"[red]Service stopped unexpectedly.[/red] Check: sudo journalctl -u {backend.service_name} --no-pager -n 50")
                raise typer.Exit(1)
            if i > 0 and i % 15 == 0:
                console.print(f"  [dim]still starting... ({i * 2}s)[/dim]")
        console.print(f"[red]Timed out waiting for health endpoint.[/red] Check: sudo journalctl -u {backend.service_name} --no-pager -n 50")
        raise typer.Exit(1)
    finally:
        lock.release()
```

- [ ] **Step 2: Update start command to use backend**

```python
@app.command()
def start(
    model: str = typer.Argument(None, help="Model name (fuzzy match)"),
    tools: bool = typer.Option(False, "--tools", help="Enable tool/function calling"),
    tool_parser: str | None = typer.Option(None, "--tool-parser", help="Override tool-call parser"),
    backend_name: str | None = typer.Option(None, "--backend", help="Force backend (vllm, llamacpp)"),
):
    """Start serving a model — interactive config picker."""
    _session_or_exit()

    if model is None:
        # Scan all backend model directories
        from vserve.backends import available_backends, _BACKENDS
        all_models = _scan_all_models()
        if not all_models:
            console.print("[red]No models found.[/red] Run: vserve download")
            raise typer.Exit(1)

        while True:
            console.print("\n[bold]Select a model:[/bold]")
            for i, (m, b) in enumerate(all_models, 1):
                has_limits = read_limits(limits_path(m.provider, m.model_name)) is not None
                status = "[green]tuned[/green]" if has_limits else "[dim]not tuned[/dim]"
                console.print(f"  {i}) {m.full_name}  ({m.model_size_gb} GB, {b.display_name})  {status}")

            choice = _pick_number("\nModel number", len(all_models))
            m, backend = all_models[choice - 1]
            break
    else:
        m = _resolve_model(model)
        if backend_name:
            from vserve.backends import get_backend_by_name
            backend = get_backend_by_name(backend_name)
        else:
            from vserve.backends import get_backend
            backend = get_backend(m)

    if tool_parser and not tools:
        tools = True
    cfg_path = _custom_config(m, backend=backend, tools=tools, tool_parser=tool_parser)
    _launch_backend(backend, cfg_path, m.model_name)
```

- [ ] **Step 3: Add _scan_all_models helper**

```python
def _scan_all_models() -> list[tuple[ModelInfo, "Backend"]]:
    """Scan all backend model directories, return (model, backend) pairs."""
    from vserve.backends import _BACKENDS
    results = []
    seen_paths: set[Path] = set()

    for b in _BACKENDS:
        models_dir = b.root_dir / "models"
        if not models_dir.exists():
            continue
        for m in scan_models(models_dir):
            if m.path not in seen_paths and b.can_serve(m):
                results.append((m, b))
                seen_paths.add(m.path)
    return results
```

- [ ] **Step 4: Update _resolve_model to search all backends**

```python
def _resolve_model(query: str) -> ModelInfo:
    all_models = [m for m, _ in _scan_all_models()]
    if not all_models:
        # Fall back to legacy MODELS_DIR
        all_models = scan_models(MODELS_DIR)
    if not all_models:
        console.print("[red]No models found.[/red] Run: vserve download")
        raise typer.Exit(1)
    matches = fuzzy_match(query, all_models)
    if len(matches) == 0:
        console.print(f"[red]No model matching '{query}'[/red]")
        for m in all_models:
            console.print(f"  {m.full_name}")
        raise typer.Exit(1)
    if len(matches) > 1:
        console.print(f"[yellow]'{query}' matches multiple models:[/yellow]")
        for m in matches:
            console.print(f"  {m.full_name}")
        raise typer.Exit(1)
    return matches[0]
```

- [ ] **Step 5: Update stop command**

```python
@app.command()
def stop():
    """Stop the inference server."""
    from vserve.backends import _BACKENDS

    _session_or_exit()

    # Find which backend is running
    running_backend = None
    for b in _BACKENDS:
        try:
            if b.is_running():
                running_backend = b
                break
        except Exception:
            continue

    if running_backend is None:
        clear_session()
        console.print("[dim]No server is running.[/dim]")
        return

    lock = _lock_or_exit("gpu", f"stopping {running_backend.display_name}")
    try:
        _session_or_exit()

        try:
            running_backend.stop()
        except RuntimeError as e:
            clear_session()
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
        clear_session()
        console.print(f"[green]{running_backend.display_name} stopped.[/green]")
    finally:
        lock.release()
```

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass. Some CLI tests may need minor adjustments if they mock `_launch_vllm` — update those to mock `_launch_backend` instead.

- [ ] **Step 7: Lint**

Run: `uv run ruff check src/vserve/cli.py`
Expected: No errors

- [ ] **Step 8: Commit**

```bash
git add src/vserve/cli.py
git commit -m "feat: CLI start/stop uses Backend Protocol"
```

---

### Task 8: CLI Integration — tune, models, status, doctor, init

**Files:**
- Modify: `src/vserve/cli.py`

- [ ] **Step 1: Update tune command**

The tune command needs to detect the backend and delegate. Replace the `calculate_limits` call with `backend.tune()`:

```python
@app.command()
def tune(
    model: str = typer.Argument(None, help="Model name (fuzzy match)"),
    all_models: bool = typer.Option(False, "--all", help="Tune all models"),
    recalc: bool = typer.Option(False, "--recalc", help="Force recalculation"),
    gpu_util: float = typer.Option(None, "--gpu-util", help="GPU memory utilization"),
):
    """Calculate serving limits for a model."""
    # ... existing gpu_util resolution logic stays unchanged ...

    for m in models_to_tune:
        from vserve.backends import get_backend
        try:
            backend = get_backend(m)
        except ValueError:
            console.print(f"[bold]{m.full_name}[/bold]  [red]No backend available[/red]")
            continue

        lim_path = limits_path(m.provider, m.model_name)
        if not recalc:
            existing = read_limits(lim_path)
            if existing and existing.get("gpu_memory_utilization") == gpu_util and existing.get("vram_total_gb") == gpu.vram_total_gb:
                console.print(f"[bold]{m.full_name}[/bold]  [dim](cached)[/dim]")
                _print_limits_table(existing, m, backend)
                continue

        limits_data = backend.tune(m, gpu, gpu_mem_util=gpu_util)
        write_limits(lim_path, limits_data)
        console.print(f"[bold]{m.full_name}[/bold]  [dim]({m.model_size_gb} GB, {backend.display_name})[/dim]")
        _print_limits_table(limits_data, m, backend)
```

- [ ] **Step 2: Update models command to show Backend column**

```python
@app.command()
def models(model: str = typer.Argument(None, help="Model name for detail view")):
    """List downloaded models."""
    all_entries = _scan_all_models()
    if not all_entries:
        console.print("[dim]No models found.[/dim] Run: vserve download")
        return

    if model:
        m = _resolve_model(model)
        _show_model_detail(m)
        return

    table = Table(title="Downloaded Models")
    table.add_column("Model", style="bold")
    table.add_column("Backend", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Limits")
    table.add_column("Max Context", justify="right")
    table.add_column("Tools", style="green")
    table.add_column("Reasoning", style="green")

    for m, backend in all_entries:
        lim = read_limits(limits_path(m.provider, m.model_name))

        max_ctx = "\u2014"
        if lim:
            for ctx_str in sorted(lim.get("limits", {}).keys(), key=lambda x: int(x), reverse=True):
                entry = lim["limits"][ctx_str]
                if isinstance(entry, dict):
                    if any(v is not None for v in entry.values()):
                        max_ctx = f"{int(ctx_str) // 1024}k"
                        break
                elif entry is not None:
                    max_ctx = f"{int(ctx_str) // 1024}k"
                    break

        tool_info = backend.detect_tools(m.path)
        tp = tool_info.get("tool_call_parser") or ("\u2713" if tool_info.get("supports_tools") else "\u2014")
        rp = tool_info.get("reasoning_parser", "\u2014") or "\u2014"

        table.add_row(
            m.full_name, backend.display_name, f"{m.model_size_gb} GB",
            "\u2713" if lim else "\u2717", max_ctx, tp, rp,
        )

    console.print(table)
```

- [ ] **Step 3: Update status command**

```python
@app.command()
def status():
    """Show current serving status."""
    from vserve.backends import _BACKENDS

    running_backend = None
    for b in _BACKENDS:
        try:
            if b.is_running():
                running_backend = b
                break
        except Exception:
            continue

    if running_backend is None:
        console.print("[dim]No server is running.[/dim] Start with: vserve start")
        return

    # Show status based on backend
    console.print(f"\n[bold green]{running_backend.display_name} is running[/bold green]")
    # ... read active config (YAML for vLLM, JSON for llama.cpp) ...
```

- [ ] **Step 4: Update doctor command**

Add iteration over backends in the doctor command:

```python
    # -- Backends --
    from vserve.backends import _BACKENDS
    console.print("\n  [bold]Backends[/bold]")
    for b in _BACKENDS:
        for desc, check_fn in b.doctor_checks():
            try:
                if check_fn():
                    _ok(f"{desc}")
                else:
                    _fail(f"{desc}")
            except Exception:
                _fail(f"{desc} (check raised error)")
```

- [ ] **Step 5: Update init command**

Add llama.cpp discovery to the init command:

```python
    # ── llama.cpp ──
    llamacpp_root = None
    llamacpp_bin = shutil.which("llama-server")
    candidate = _Path("/opt/llama-cpp")
    if (candidate / "bin" / "llama-server").exists():
        llamacpp_root = candidate
        _ok(f"llama.cpp   found at {candidate}")
    elif llamacpp_bin:
        _ok(f"llama.cpp   {llamacpp_bin}")
        llamacpp_root = _Path(llamacpp_bin).parent.parent
    else:
        _warn("llama.cpp   not found (optional — for GGUF models)")

    # Update config build to include llamacpp_root
    config = _build_config(root, cuda, svc_name, svc_user, port,
                           llamacpp_root=llamacpp_root)
```

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 7: Lint**

Run: `uv run ruff check src/vserve/cli.py`
Expected: No errors

- [ ] **Step 8: Commit**

```bash
git add src/vserve/cli.py
git commit -m "feat: tune/models/status/doctor/init use Backend Protocol"
```

---

### Task 9: Download Flow — GGUF Support

**Files:**
- Modify: `src/vserve/cli.py` (download command)

- [ ] **Step 1: Update download to route GGUF to hf_hub_download**

In the download command, after the user picks a variant, check if it's GGUF:

```python
    # Determine destination based on variant type
    is_gguf_variant = any(f.endswith(".gguf") for f in allow_files) if allow_files else False

    if is_gguf_variant:
        # GGUF: download individual files to llama-cpp models dir
        from vserve.config import cfg as _cfg
        _c = _cfg()
        llamacpp_root = _c.llamacpp_root or Path("/opt/llama-cpp")
        local_dir = llamacpp_root / "models" / provider / model_name
    else:
        # Safetensors: snapshot_download to vllm models dir
        local_dir = MODELS_DIR / provider / model_name

    local_dir.mkdir(parents=True, exist_ok=True)

    # Download
    if is_gguf_variant:
        from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]
        for filename in allow_files:
            hf_hub_download(repo_id=model_id, filename=filename, local_dir=local_dir)
        # Also grab tokenizer_config.json for tool detection
        try:
            hf_hub_download(repo_id=model_id, filename="tokenizer_config.json", local_dir=local_dir)
        except Exception:
            pass  # Not all GGUF repos have it
    else:
        snapshot_download(repo_id=model_id, local_dir=local_dir, allow_patterns=allow_files)
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 3: Lint**

Run: `uv run ruff check src/vserve/cli.py`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add src/vserve/cli.py
git commit -m "feat: download routes GGUF to llama-cpp models dir"
```

---

### Task 10: Final Integration, Version Bump, Full Test

**Files:**
- Modify: `pyproject.toml` (version bump)
- Modify: `src/vserve/__init__.py` (version bump)
- Modify: `README.md` (document multi-backend)

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All pass

- [ ] **Step 2: Run linter and type checker**

Run: `uv run ruff check src/ tests/ && uv run mypy src/vserve/`
Expected: No errors

- [ ] **Step 3: Manual smoke test**

Run: `uv run vserve models` — should show Backend column
Run: `uv run vserve doctor` — should show backend checks

- [ ] **Step 4: Version bump to 0.4.0**

In `pyproject.toml`: `version = "0.4.0"`
In `src/vserve/__init__.py`: `__version__ = "0.4.0"`

- [ ] **Step 5: Update README**

Add llama.cpp to the Quick Start, Commands table, and Prerequisites sections.

- [ ] **Step 6: Commit and tag**

```bash
git add -A
git commit -m "feat: v0.4.0 — multi-backend support (vLLM + llama.cpp)"
git tag v0.4.0
```
