<div align="center">

# vserve

**A CLI for managing vLLM inference on GPU workstations.**

Download models. Auto-tune limits. Serve with one command. Tool calling built in.

![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776ab?style=flat-square&logo=python&logoColor=white)
![vLLM 0.18+](https://img.shields.io/badge/vLLM-0.18+-ff6f00?style=flat-square)
![Tests](https://img.shields.io/badge/tests-205%20passed-brightgreen?style=flat-square)
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

---

## Quick Start

```bash
vserve init                        # scan GPU, vLLM, CUDA, systemd — write config
vserve download                    # search HuggingFace, pick variant, download
vserve start <model>               # auto-tune + interactive config + serve
vserve start <model> --tools       # enable tool calling (parser auto-detected)
```

---

## What It Does

**vserve** manages the full lifecycle of serving LLMs with vLLM on a GPU workstation:

- **Download** — search HuggingFace, see available weight variants (FP8, NVFP4, BF16, GGUF) with sizes, download only what you need
- **Auto-tune** — calculate exactly what context lengths and concurrency your GPU can handle, based on model architecture and available VRAM. Runs automatically on first start.
- **Tool calling** — auto-detects the correct `--tool-call-parser` and `--reasoning-parser` from the model's chat template. Supports Qwen, Llama, Mistral, DeepSeek, Gemma 4, GPT-OSS, and more.
- **Start/Stop** — interactive config wizard, systemd service management, health check with timeout
- **Fan control** — temperature-based curve daemon with quiet hours, or hold a fixed speed
- **Multi-user** — session-based GPU ownership prevents other users from disrupting your running model. File-based locking with terminal notifications.
- **Doctor** — diagnose GPU, CUDA, vLLM, systemd issues with actionable fix suggestions

---

## Commands

| Command | Description |
|:--------|:------------|
| `vserve` | Dashboard — GPU, models, status |
| `vserve init` | Auto-discover vLLM and write config |
| `vserve download [model]` | Search and download from HuggingFace with variant picker |
| `vserve models [name]` | List models or show detail (fuzzy match) |
| `vserve tune [model]` | Calculate context/concurrency limits and detect tool capabilities |
| `vserve start [model]` | Configure and start serving (auto-tunes if needed) |
| `vserve start <model> --tools` | Start with tool calling enabled (parser auto-detected) |
| `vserve start <model> --tool-parser <p>` | Override tool-call parser manually |
| `vserve stop` | Stop the vLLM service |
| `vserve status` | Show current serving config |
| `vserve fan [auto\|off\|30-100]` | GPU fan control with temp-based curve |
| `vserve doctor` | Check system readiness |

All commands support **fuzzy matching** — `vserve start qwen fp8` finds the right model.

---

## Tool Calling

vserve auto-detects the correct vLLM parser by reading the model's chat template:

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

Detection is template-based (not model-name regex), so it works for fine-tunes and community uploads. Use `--tool-parser` to override when auto-detection can't determine the parser.

---

## Prerequisites

| Requirement | Check | Install |
|:------------|:------|:--------|
| NVIDIA GPU + drivers | `nvidia-smi` | [nvidia.com/drivers](https://www.nvidia.com/drivers) |
| CUDA toolkit | `nvcc --version` | `sudo apt install nvidia-cuda-toolkit` |
| vLLM 0.18+ | `vllm --version` | [docs.vllm.ai](https://docs.vllm.ai/en/latest/getting_started/installation.html) |
| systemd | (most Linux servers) | See [troubleshooting](docs/troubleshooting.md) |
| sudo access | for systemctl, fan control | |

---

## Configuration

Auto-discovered on first run. Override at `~/.config/vserve/config.yaml`:

```yaml
vllm_root: /opt/vllm
cuda_home: /usr/local/cuda
service_name: vllm
service_user: vllm
port: 8888
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

## Development

```bash
git clone https://github.com/Gavin-Qiao/vserve.git
cd vserve
uv sync --dev
uv run pytest tests/              # 205 tests
uv run ruff check src/ tests/     # lint
uv run mypy src/vserve/           # type check
```

---

## License

[MIT](LICENSE)
