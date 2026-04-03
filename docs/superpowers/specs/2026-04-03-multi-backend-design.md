# Multi-Backend Architecture for vserve

**Date:** 2026-04-03
**Status:** Approved
**Version target:** v0.4.0

## Goal

Transform vserve from a vLLM-only CLI into a multi-backend inference manager. First addition: llama.cpp (GGUF models). Future: SGLang. Ultimate vision: any LLM/VLM inference engine.

## Constraints

- No breaking changes to existing vLLM workflows
- Existing 205+ tests must pass without modification (re-exports where needed)
- Separate install directories per backend (`/opt/vllm`, `/opt/llama-cpp`) — unified `/opt/llm` deferred to a future version
- Backend auto-detected from model format (no user decision needed for unambiguous cases)
- Python — HuggingFace Hub SDK, NVML bindings, and ML ecosystem require it

---

## Architecture

### Backend Protocol

Every backend implements this Protocol (structural subtyping — no inheritance required):

```python
class Backend(Protocol):
    # Identity
    name: str                              # "vllm", "llamacpp"
    display_name: str                      # "vLLM", "llama.cpp"

    # System integration
    service_name: str                      # systemd unit: "vllm", "llama-cpp"
    service_user: str                      # OS user: "vllm", "llama-cpp"
    root_dir: Path                         # /opt/vllm, /opt/llama-cpp

    # Detection
    def can_serve(self, model: ModelInfo) -> bool: ...
    def find_entrypoint(self) -> Path | str | None: ...

    # Tuning
    def tune(self, model: ModelInfo, gpu: GpuInfo) -> dict: ...

    # Configuration
    def build_config(self, model: ModelInfo, choices: dict) -> dict: ...
    def quant_flag(self, method: str | None) -> str: ...  # empty string if N/A (e.g. llama.cpp)

    # Serving lifecycle
    def start(self, config_path: Path) -> None: ...
    def stop(self) -> None: ...
    def is_running(self) -> bool: ...
    def health_url(self, port: int) -> str: ...

    # Capabilities
    def detect_tools(self, model_path: Path) -> dict: ...

    # Diagnostics
    def doctor_checks(self) -> list[tuple[str, Callable[[], bool]]]: ...
```

Design decisions:
- **Protocol over ABC** — backends may live in optional-dependency packages; structural subtyping avoids import-time coupling.
- **`find_entrypoint()` not `find_binary()`** — SGLang launches via `python -m sglang.launch_server`, not a standalone binary.
- **`download()` excluded** — downloading is a model/format concern handled by `variants.py` and HuggingFace Hub, not a backend concern.
- **`detect_tools()` returns dict** — vLLM returns `{"tool_call_parser": str, "reasoning_parser": str}`, llama.cpp returns `{"supports_tools": bool}`. Flexible per backend.

### Registry

```python
# backends/__init__.py

_BACKENDS: list[Backend] = []

def register(backend: Backend) -> None: ...
def get_backend(model: ModelInfo) -> Backend: ...      # auto-detect from format
def get_backend_by_name(name: str) -> Backend: ...     # explicit --backend flag
def available_backends() -> list[Backend]: ...          # installed backends only
```

Auto-detection rules (no format overlap today):
- Model directory contains `.gguf` files -> `llamacpp`
- Model directory contains `config.json` with safetensors -> `vllm`
- Future: multiple candidates for safetensors (vLLM vs SGLang) -> prompt user or use config preference

### File Structure

```
src/vserve/
  backends/
    __init__.py        # registry, get_backend(), auto-detect
    protocol.py        # Backend Protocol
    vllm.py            # VllmBackend — wraps existing serve/probe/tools
    llamacpp.py        # LlamaCppBackend — new
  cli.py               # thinner — delegates to backend methods
  config.py            # gains llamacpp_* fields
  serve.py             # kept as-is, called by VllmBackend
  probe.py             # kept as-is, called by VllmBackend
  tools.py             # kept as-is, called by VllmBackend
  gpu.py               # unchanged (backend-agnostic)
  fan.py               # unchanged (backend-agnostic)
  compare.py           # unchanged (backend-agnostic)
  variants.py          # unchanged (backend-agnostic)
  models.py            # loses QUANT_FLAGS (re-exported for compat)
  lock.py              # unchanged (backend-agnostic)
  version.py           # unchanged
```

---

## VllmBackend

Wraps existing modules — no rewrite, delegation only:

| Protocol method | Delegates to |
|----------------|-------------|
| `can_serve(model)` | Check for `config.json` + safetensors/bin weights |
| `find_entrypoint()` | `config._discover_vllm_root()` + `venv/bin/vllm` |
| `tune(model, gpu)` | `probe.calculate_limits()` |
| `build_config(model, choices)` | New — extracts config dict builder from `cli._custom_config()` |
| `quant_flag(method)` | `models.QUANT_FLAGS` (moved here, re-exported from models.py) |
| `start(config_path)` | `serve.start_vllm()` |
| `stop()` | `serve.stop_vllm()` |
| `is_running()` | `serve.is_vllm_running()` |
| `health_url(port)` | `f"http://localhost:{port}/health"` |
| `detect_tools(model_path)` | `tools.detect_tool_parser()` + `tools.detect_reasoning_parser()` |
| `doctor_checks()` | Extract checks from current `cli.doctor()` |

---

## LlamaCppBackend

### Identity

```python
name = "llamacpp"
display_name = "llama.cpp"
service_name = "llama-cpp"
service_user = "llama-cpp"
root_dir = Path("/opt/llama-cpp")
```

User/group setup: `llama-cpp` user under `llm` group, mirroring `vllm` user under `llm` group.

### can_serve(model)

Returns True if model directory contains `.gguf` files.

### find_entrypoint()

Searches: `/opt/llama-cpp/bin/llama-server`, then `$PATH`.

### tune(model, gpu)

GGUF-specific tuning using the `gguf` Python package to read metadata from GGUF headers:

```python
from gguf import GGUFReader
reader = GGUFReader(gguf_path)  # first shard for split files
num_layers = reader.fields[f"{arch}.block_count"]
context_length = reader.fields[f"{arch}.context_length"]
num_kv_heads = reader.fields[f"{arch}.attention.head_count_kv"]
```

Calculation:
1. **Layer offload**: `layer_size = model_file_size / num_layers`. How many fit in VRAM (with 10% buffer for embeddings/scratch) -> `--n-gpu-layers`.
2. **Context size**: If all layers on GPU, remaining VRAM -> max `--ctx-size`. KV cache is FP16 by default (2 bytes per element), same formula as vLLM but fixed dtype.
3. **Parallel slots**: `available_kv_memory / (kv_per_token * ctx_size)` -> `--parallel`.
4. **Partial offload**: If model doesn't fully fit, calculate how many layers fit and warn about performance impact.

Output format:
```json
{
  "model_path": "/opt/llama-cpp/models/.../model.gguf",
  "n_gpu_layers": 35,
  "limits": {"4096": 8, "8192": 4, "16384": 2},
  "supports_tools": true,
  "max_context": 131072
}
```

### build_config(model, choices)

Produces llama-server JSON config (llama-server uses `--config` with JSON, not YAML):
```json
{
  "model": "/opt/llama-cpp/models/.../model-Q4_K_M.gguf",
  "host": "0.0.0.0",
  "port": 8888,
  "ctx-size": 8192,
  "n-gpu-layers": 35,
  "parallel": 4,
  "flash-attn": true,
  "cont-batching": true,
  "jinja": true
}
```

### detect_tools(model_path)

Reads `tokenizer_config.json` (downloaded alongside GGUF), checks chat template for tool markers (`{% if tools %}`, `<tool_call>`, etc.). Returns `{"supports_tools": True/False}`. No parser name needed — llama.cpp uses `--jinja` universally.

### health_url(port)

`http://localhost:{port}/health` — same endpoint as vLLM.

### Download flow

Not a backend method. The existing `vserve download` flow handles this:
- `variants.py` already detects GGUF files and groups them by quant type
- User picks a variant (Q4_K_M, Q8_0, etc.)
- Download uses `hf_hub_download(repo_id, filename)` for the selected GGUF file
- Also downloads `tokenizer_config.json` for tool detection
- Destination: `/opt/llama-cpp/models/{provider}/{repo_name}/{filename}.gguf`

---

## CLI Changes

### vserve start

```
1. Pick model (unchanged)
2. backend = get_backend(model)           # auto from format
3. --backend flag overrides               # explicit
4. _session_or_exit()                     # unchanged
5. limits = backend.tune(model, gpu)      # if not cached
6. choices = _interactive_wizard(model, limits, backend)
7. config = backend.build_config(model, choices)
8. backend.start(config_path)
9. _wait_for_health(backend, port)        # generic health-poll loop
```

`_launch_vllm()` decomposed into:
- `backend.start(config_path)` — write config, systemctl start
- `_wait_for_health(backend, port)` — generic: poll `backend.health_url(port)`, timeout, UI feedback

Interactive wizard adapts per backend:
- **vLLM**: KV dtype (auto/fp8), max-model-len, max-num-seqs, batched tokens, tool parser selection
- **llama.cpp**: ctx-size, n-gpu-layers (if partial offload needed), parallel slots, --tools toggle

### vserve stop

Reads which backend's service is running (check each backend's `is_running()`), calls `backend.stop()`.

### vserve models

New "Backend" column showing which engine(s) can serve each model.

### vserve status

Shows active backend name alongside model info.

### vserve init

Discovers both `/opt/vllm` and `/opt/llama-cpp`, reports what's found.

### vserve doctor

Iterates `available_backends()`, runs each backend's `doctor_checks()`.

### vserve download

Mostly unchanged. Variant picker already shows GGUF. Download dispatches by format:
- safetensors -> `snapshot_download()` to `{vllm_root}/models/`
- GGUF -> `hf_hub_download()` to `{llamacpp_root}/models/`

### vserve tune

Calls `backend.tune(model, gpu)`. Interactive GPU usage menu stays in CLI (backend-agnostic).

---

## Config Changes

`~/.config/vserve/config.yaml`:
```yaml
# Shared
port: 8888

# vLLM
vllm_root: /opt/vllm
cuda_home: /usr/local/cuda
service_name: vllm
service_user: vllm
gpu_memory_utilization: 0.92

# llama.cpp
llamacpp_root: /opt/llama-cpp
llamacpp_service_name: llama-cpp
llamacpp_service_user: llama-cpp
```

`VserveConfig` dataclass gains:
- `llamacpp_root: Path | None = None`
- `llamacpp_service_name: str = "llama-cpp"`
- `llamacpp_service_user: str = "llama-cpp"`

Backend instances receive the shared config and read their own fields.

---

## Dependencies

New optional dependency for llama.cpp support:
```toml
[project.optional-dependencies]
llamacpp = ["gguf>=0.6"]
```

The `gguf` package is needed to read GGUF file metadata for tuning. It is a lightweight pure-Python package from the llama.cpp project.

---

## Systemd

New service unit for llama.cpp (parallel to existing vllm.service):

```ini
# /etc/systemd/system/llama-cpp.service
[Unit]
Description=llama.cpp inference server
After=network.target

[Service]
Type=simple
User=llama-cpp
Group=llm
ExecStart=/opt/llama-cpp/bin/llama-server --config /opt/llama-cpp/configs/active.json
Restart=on-failure
Environment=CUDA_VISIBLE_DEVICES=0

[Install]
WantedBy=multi-user.target
```

`vserve init` can generate this unit file, or document it for manual setup.

---

## Test Strategy

- Existing 205+ tests pass unchanged (re-export `quant_flag` from `models.py`)
- New unit tests for: `LlamaCppBackend.tune()`, `LlamaCppBackend.build_config()`, registry auto-detection
- Integration test: mock GGUF file with known metadata, verify tune output
- VllmBackend tests: verify delegation to existing modules produces same results

---

## Migration Notes

- No breaking changes for existing users
- `vserve init` on a machine without llama.cpp simply doesn't register that backend
- Config file is additive — existing configs without `llamacpp_*` fields work (defaults to None/not-installed)
- Future: `vserve init --migrate` to move to unified `/opt/llm` structure
