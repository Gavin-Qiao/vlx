<div align="center">

# vserve

**A CLI for managing LLM inference on GPU workstations.**

Download models. Auto-tune limits. Serve with one command. Multiple backends.

![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776ab?style=flat-square&logo=python&logoColor=white)
![vLLM 0.20.x](https://img.shields.io/badge/vLLM-0.20.x-ff6f00?style=flat-square)
![llama.cpp](https://img.shields.io/badge/llama.cpp-GGUF-purple?style=flat-square)
![Tests](https://img.shields.io/badge/tests-428%20passed-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

</div>

---

## Beta Release: 0.5.2b1

`vserve` is now in beta.

Highlights in `0.5.2b1`:

- lifecycle coordination is now backend-wide: `run` stops conflicting backends before launch, and `stop` drains all detected active backends
- startup handling is stricter for automation and friendlier for operators: interactive `run` can return while a backend is still warming, while non-interactive `run` still requires a healthy API
- `vserve run` now polls health every 3 seconds and shows the latest 5 `journalctl` lines while waiting, which makes JIT/kernel warmup much easier to follow
- config, profile, limits, and local state parsing is hardened against malformed files
- doctor/update/runtime paths are more defensive around probe failures, unreadable files, and backend/service uncertainty
- vLLM runtime support is pinned to stable `>=0.20,<0.21`; `vserve runtime check vllm` reports installed vLLM, torch, CUDA, Transformers, and dependency health
- tuning caches are versioned and tied to model, GPU, backend, and runtime fingerprints, so stale limits are recalculated after runtime drift

Current beta caveats:

- non-interactive startup remains intentionally strict: if the backend never reaches a healthy API state within the timeout window, `run` exits nonzero even if the service is still warming
- multi-user coordination is best-effort operational safety, not a security boundary

---

## Install

Beta/pre-release channel (primary while `vserve` is in beta):

```bash
uv tool install --prerelease allow vserve
pip install --pre vserve
vserve update --nightly
```

For llama.cpp GGUF tuning support:

```bash
pip install 'vserve[llamacpp]'
```

---

## Quick Start

```bash
vserve init                        # scan GPU, backends, CUDA, systemd — write config
vserve runtime check vllm          # verify the external vLLM runtime
vserve add                         # search HuggingFace, pick variant, download
vserve run <model>                 # auto-tune + interactive config + serve
vserve run <model> --tools         # enable tool calling (auto-detected)
vserve run <model> --backend llamacpp  # force a specific backend
```

Scriptable serving:

```bash
vserve run qwen fp8 --yes --context 32768 --slots 4 --kv-cache-dtype fp8 --port 8888
vserve run qwen fp8 --yes --replace              # safe non-interactive restart
vserve run qwen fp8 --save-profile fast --yes
vserve run qwen fp8 --profile fast
vserve run --profile /opt/vllm/configs/models/provider--Model.fast.yaml --yes
```

Runtime repair and GGUF-only setup:

```bash
vserve runtime check vllm
vserve runtime upgrade vllm --stable
vserve add TheBloke/some-model-GGUF
vserve run some model q4 --backend llamacpp --yes --gpu-layers 999
```

Automation:

```bash
vserve run qwen fp8 --profile fast --yes
vserve status
vserve stop
```

---

## Backends

vserve auto-detects the right backend from the model format:

| Format | Backend | Engine |
|:-------|:--------|:-------|
| safetensors, GPTQ, AWQ, FP8 | **vLLM** | PagedAttention, continuous batching |
| GGUF | **llama.cpp** | CPU/GPU offload, quantized inference |

No configuration needed — download a model and `vserve run` picks the right engine.

### vLLM

The default for transformer models in safetensors format. Optimized for high-throughput serving with PagedAttention, KV cache management, and automatic batching.

- Auto-tunes `--max-model-len`, `--max-num-seqs`, `--kv-cache-dtype` based on your GPU
- Tool calling with parser auto-detection (Qwen, Llama, Mistral, DeepSeek, Gemma, GPT-OSS)
- Systemd service management via `vllm.service`

### llama.cpp

For GGUF quantized models. Serves via `llama-server` with an OpenAI-compatible API.

- Auto-calculates `--n-gpu-layers`, `--ctx-size`, `--parallel` based on VRAM
- Partial GPU offload — serve models that don't fully fit in VRAM
- Tool calling via `--jinja` (no parser configuration needed)
- Systemd service management via `llama-cpp.service`

---

## What It Does

**vserve** manages the full lifecycle of serving LLMs on a GPU workstation:

- **Download** — search HuggingFace, see available weight variants (FP8, NVFP4, BF16, GGUF) with sizes, download only what you need
- **Auto-tune** — calculate exactly what context lengths and concurrency your GPU can handle, based on model architecture and available VRAM
- **Tool calling** — auto-detects the correct parser from the model's chat template (vLLM) or uses `--jinja` (llama.cpp)
- **Run/Stop** — interactive config wizard, systemd service management, health check with timeout
- **Fan control** — temperature-based curve daemon with quiet hours, or hold a fixed speed
- **Multi-user** — best-effort session coordination warns other `vserve` users before they disrupt your running model
- **Doctor** — diagnose GPU, CUDA, backend, systemd issues with actionable fix suggestions

---

## Commands

| Command | Description |
|:--------|:------------|
| `vserve` | Dashboard — GPU, models, status |
| `vserve init` | Auto-discover backends and write config |
| `vserve list [name]` | List models with backend, tools, and limits |
| `vserve add [model]` | Search and download from HuggingFace with variant picker |
| `vserve rm <name>` | Remove a downloaded model |
| `vserve tune [model]` | Calculate context/concurrency limits |
| `vserve run [model]` | Configure and start serving (auto-tunes if needed) |
| `vserve run MODEL... --yes --context N --slots N` | Non-interactive serving from flags |
| `vserve run MODEL... --yes --replace` | Non-interactive restart; without `--replace`, running backends are refused |
| `vserve run MODEL... --profile NAME_OR_PATH` | Serve a saved profile by name or explicit path |
| `vserve run MODEL... --tools --tool-parser hermes --reasoning-parser qwen3` | Start with explicit parsers |
| `vserve run MODEL... --trust-remote-code` | Opt in to vLLM remote model code execution |
| `vserve run MODEL... --backend llamacpp --gpu-layers 999` | Force llama.cpp for GGUF |
| `vserve profile list\|show\|rm` | Manage saved serving profiles |
| `vserve stop` | Stop the running server |
| `vserve status [--json]` | Show current serving config and probe uncertainty |
| `vserve fan [auto\|off\|30-100]` | GPU fan control with temp-based curve |
| `vserve doctor [--json] [--strict]` | Check system readiness; strict exits nonzero on failures |
| `vserve cache clean [--dry-run] [--all] [--yes]` | Preview or clean stale sockets and JIT caches |
| `vserve runtime check vllm` | Check vLLM version/dependency compatibility |
| `vserve runtime upgrade vllm --stable` | Reinstall vserve's pinned stable vLLM runtime |
| `vserve version` | Show current version and check for updates |
| `vserve update [--nightly]` | Update vserve, optionally allowing pre-releases |

Model-taking commands support **fuzzy matching** — `vserve run qwen fp8` finds the right model.

Profile rules: profile names resolve inside configured vserve profile roots; explicit `--profile` paths are accepted for `run` and infer backend from YAML/JSON when possible. `profile rm` never deletes arbitrary external paths, even with `--force`.

---

## Tool Calling

### vLLM

Auto-detects the correct vLLM parser by reading the model's chat template:

| Model Family | Tool Parser | Reasoning Parser |
|:-------------|:------------|:-----------------|
| Qwen 2.5 | `hermes` | — |
| Qwen 3 | `hermes` | `qwen3` |
| Qwen 3.5 | `qwen3_coder` | `qwen3` |
| Llama 3.1 / 3.2 / 3.3 | `llama3_json` | — |
| Llama 4 | `llama4_pythonic` | — |
| Mistral / Mixtral | `mistral` | `mistral` |
| DeepSeek V3 / R1 | `deepseek_v3` | `deepseek_r1` |
| Gemma 4 | `gemma4` | `gemma4` |
| GPT-OSS | `openai` | `openai_gptoss` |

Detection is template-based (not model-name regex), so it works for fine-tunes and community uploads.

Remote model code is disabled by default. Use `--trust-remote-code` only for repositories you trust; generated profiles include `trust-remote-code` only when that flag is explicitly set.

### llama.cpp

Uses `--jinja` to read the model's chat template directly. No parser selection needed — one flag covers all model families.

---

## Prerequisites

| Requirement | Check | Install |
|:------------|:------|:--------|
| NVIDIA GPU + drivers | `nvidia-smi` | [nvidia.com/drivers](https://www.nvidia.com/drivers) |
| CUDA toolkit | `nvcc --version` | `sudo apt install nvidia-cuda-toolkit` |
| systemd | (most Linux servers) | See [troubleshooting](docs/troubleshooting.md) |
| sudo access | for systemctl, fan control | |

**For vLLM backend:**

| Requirement | Check | Install |
|:------------|:------|:--------|
| stable vLLM 0.20.x | `vserve runtime check vllm` | `vserve runtime upgrade vllm --stable` or [docs.vllm.ai](https://docs.vllm.ai/en/latest/getting_started/installation.html) |

**For llama.cpp backend:**

| Requirement | Check | Install |
|:------------|:------|:--------|
| llama-server | `llama-server --version` | [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |

---

## Configuration

Auto-discovered on first run. Override at `~/.config/vserve/config.yaml`:

```yaml
schema_version: 2
cuda_home: /usr/local/cuda
gpu:
  index: 0
  memory_utilization: 0.91
backends:
  vllm:
    root: /opt/vllm
    service_name: vllm
    service_user: vllm
    port: 8888
  llamacpp:
    root: /opt/llama-cpp
    service_name: llama-cpp
    service_user: llama-cpp
```

Legacy top-level `vllm_root`, `service_name`, `llamacpp_root`, and GPU memory keys still load, but newly saved config uses the backend-indexed schema above.

---

## Directory Layout

```
/opt/vllm/                     # vLLM backend
├── venv/bin/vllm              # Python venv
├── .venv/bin/vllm             # alternate Python venv location
├── models/                    # safetensors models
├── configs/
│   ├── .env                   # service environment
│   ├── active.yaml            # active profile symlink
│   └── models/                # limits + YAML profiles
├── tmp/                       # RPC sockets / runtime temp files
├── .cache/
│   ├── flashinfer/            # FlashInfer JIT cache
│   ├── torch_extensions/      # torch extension cache
│   └── vllm/                  # vLLM/torch.compile cache
├── run/
│   └── active-manifest.json   # active backend state
└── logs/

/opt/llama-cpp/                # llama.cpp backend
├── bin/llama-server           # compiled binary
├── models/                    # GGUF models
├── configs/
│   ├── active.sh              # active launch script symlink
│   ├── active.json            # active config symlink
│   └── models/                # JSON profiles
├── run/
│   └── active-manifest.json   # active backend state
└── logs/
```

GGUF downloads create one runnable model root per selected quant/subdirectory, so `Q4_K_M` and `Q8_0` variants do not share a directory. Source roots left only for materialization are ignored by model scanning.

---

## Fan Control

```bash
vserve fan              # show status, interactive menu
vserve fan auto         # temp-based curve with quiet hours
vserve fan 80           # hold at 80% (persistent daemon)
vserve fan off          # stop daemon, restore NVIDIA auto
```

The auto curve ramps with temperature and caps fan speed during quiet hours (configurable). Emergency override at 88C ignores quiet hours.

---

## Architecture

vserve uses a **Backend Protocol** pattern. Each inference engine implements the same interface:

```
Backend Protocol
├── VllmBackend        — safetensors, AWQ, FP8, GPTQ
├── LlamaCppBackend    — GGUF
└── (future: SGLang, etc.)
```

The registry auto-detects the right backend from the model format. Runtime checks, tuning fingerprints, profile/config generation, service lifecycle, active manifests, and status summaries live behind the backend protocol so the command layer can stay focused on user workflows.

---

## Development

```bash
git clone https://github.com/Gavin-Qiao/vserve.git
cd vserve
uv sync --dev
uv run pytest tests/              # 428 tests
uv run ruff check src/ tests/     # lint
uv run mypy src/vserve/ --ignore-missing-imports --check-untyped-defs
```

---

## License

[MIT](LICENSE)
