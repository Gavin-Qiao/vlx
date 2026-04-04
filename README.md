<div align="center">

# vserve

**A CLI for managing LLM inference on GPU workstations.**

Download models. Auto-tune limits. Serve with one command. Multiple backends.

![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776ab?style=flat-square&logo=python&logoColor=white)
![vLLM 0.19+](https://img.shields.io/badge/vLLM-0.19%2B-ff6f00?style=flat-square)
![llama.cpp](https://img.shields.io/badge/llama.cpp-GGUF-purple?style=flat-square)
![Tests](https://img.shields.io/badge/tests-318%20passed-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

</div>

---

## Install

```bash
uv tool install vserve
```

Or with pip:

```bash
pip install vserve
```

For llama.cpp GGUF tuning support:

```bash
pip install 'vserve[llamacpp]'
```

---

## Quick Start

```bash
vserve init                        # scan GPU, backends, CUDA, systemd — write config
vserve add                         # search HuggingFace, pick variant, download
vserve run <model>                 # auto-tune + interactive config + serve
vserve run <model> --tools         # enable tool calling (auto-detected)
vserve run <model> --backend llamacpp  # force a specific backend
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
- **Multi-user** — session-based GPU ownership prevents other users from disrupting your running model
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
| `vserve run <model> --tools` | Start with tool calling enabled |
| `vserve run <model> --backend llamacpp` | Force a specific backend |
| `vserve stop` | Stop the running server |
| `vserve status` | Show current serving config |
| `vserve fan [auto\|off\|30-100]` | GPU fan control with temp-based curve |
| `vserve doctor` | Check system readiness |
| `vserve cache clean [--all]` | Clean stale sockets and JIT caches |
| `vserve version` | Show current version and check for updates |
| `vserve update` | Update vserve to the latest version |

All commands support **fuzzy matching** — `vserve run qwen fp8` finds the right model.

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
| vLLM 0.19+ | `vllm --version` | [docs.vllm.ai](https://docs.vllm.ai/en/latest/getting_started/installation.html) |

**For llama.cpp backend:**

| Requirement | Check | Install |
|:------------|:------|:--------|
| llama-server | `llama-server --version` | [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |

---

## Configuration

Auto-discovered on first run. Override at `~/.config/vserve/config.yaml`:

```yaml
# Shared
port: 8888

# vLLM
vllm_root: /opt/vllm
cuda_home: /usr/local/cuda
service_name: vllm
service_user: vllm

# llama.cpp (optional)
llamacpp_root: /opt/llama-cpp
llamacpp_service_name: llama-cpp
llamacpp_service_user: llama-cpp
```

---

## Directory Layout

```
/opt/vllm/                     # vLLM backend
├── venv/bin/vllm              # Python venv
├── models/                    # safetensors models
├── configs/                   # limits + profiles
└── logs/

/opt/llama-cpp/                # llama.cpp backend
├── bin/llama-server           # compiled binary
├── models/                    # GGUF models
├── configs/                   # JSON configs
└── logs/
```

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

The registry auto-detects the right backend from the model format. All CLI commands work through the protocol — no backend-specific code in the command layer.

---

## Development

```bash
git clone https://github.com/Gavin-Qiao/vserve.git
cd vserve
uv sync --dev
uv run pytest tests/              # 318 tests
uv run ruff check src/ tests/     # lint
uv run mypy src/vserve/           # type check
```

---

## License

[MIT](LICENSE)
