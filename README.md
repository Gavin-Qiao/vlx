<div align="center">

# vserve

**A CLI for managing vLLM inference on GPU workstations.**

Download models. Calculate limits. Serve with one command. Control fans.

![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776ab?style=flat-square&logo=python&logoColor=white)
![vLLM 0.18+](https://img.shields.io/badge/vLLM-0.18+-ff6f00?style=flat-square)
![Tests](https://img.shields.io/badge/tests-175%20passed-brightgreen?style=flat-square)
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
vserve init                  # scan GPU, vLLM, CUDA, systemd — write config
vserve download              # search HuggingFace, pick variant, download
vserve tune <model>          # calculate context/concurrency limits for your GPU
vserve start <model>         # interactive config → serve via systemd
```

---

## What It Does

**vserve** manages the full lifecycle of serving LLMs with vLLM on a GPU workstation:

- **Download** — search HuggingFace, see available weight variants (MXFP4, BF16, GGUF) with sizes, download only what you need
- **Tune** — calculate exactly what context lengths and concurrency your GPU can handle, based on model architecture and available VRAM
- **Start/Stop** — interactive config wizard, systemd service management, health check with timeout
- **Fan control** — temperature-based curve daemon with quiet hours, or hold a fixed speed
- **Multi-user** — file-based locking, terminal notifications when another user holds the GPU
- **Doctor** — diagnose GPU, CUDA, vLLM, systemd issues with actionable fix suggestions

---

## Commands

| Command | Description |
|:--------|:------------|
| `vserve` | Dashboard — GPU, models, status |
| `vserve init` | Auto-discover vLLM and write config |
| `vserve download [model]` | Search and download from HuggingFace with variant picker |
| `vserve models [name]` | List models or show detail (fuzzy match) |
| `vserve tune <model>` | Calculate context/concurrency limits for your GPU |
| `vserve start [model]` | Configure and start serving |
| `vserve stop` | Stop the vLLM service |
| `vserve status` | Show current serving config |
| `vserve fan [auto\|off\|30-100]` | GPU fan control with temp-based curve |
| `vserve doctor` | Check system readiness |

All commands support **fuzzy matching** — `vserve start qwen fp8` finds the right model.

---

## Prerequisites

| Requirement | Check | Install |
|------------|-------|---------|
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
uv run pytest tests/              # 175 tests
uv run ruff check src/ tests/     # lint
uv run mypy src/vserve/           # type check
```

---

## License

[MIT](LICENSE)
