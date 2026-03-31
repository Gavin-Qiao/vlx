<div align="center">

# vlx

**A modern CLI for managing vLLM inference on GPU workstations.**

Probe limits. Benchmark profiles. Tune configs. Serve models. Control fans.

![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776ab?style=flat-square&logo=python&logoColor=white)
![vLLM 0.18+](https://img.shields.io/badge/vLLM-0.18+-ff6f00?style=flat-square)
![Ruff](https://img.shields.io/badge/code%20style-ruff-d4aa00?style=flat-square&logo=ruff&logoColor=white)
![mypy](https://img.shields.io/badge/type%20checked-mypy-blue?style=flat-square)
![Tests](https://img.shields.io/badge/tests-126%20passed-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

</div>

---

## Why

Running vLLM on a workstation means juggling model downloads, context limits, concurrency tuning, benchmark configs, systemd services, and fan curves — all through fragile bash scripts.

**vlx** replaces all of that with a single CLI. It probes your GPU's actual limits, benchmarks across workload profiles, generates optimized serving configs, and keeps everything running.

---

## Quick Start

```bash
pip install vlx
vlx init                  # auto-discover vLLM installation
vlx                       # dashboard
vlx download <model>      # download from HuggingFace
vlx tune <model>          # probe limits + benchmark
vlx start <model>         # configure & serve
```

---

## Commands

| Command | Description |
|:--------|:------------|
| `vlx` | Dashboard — GPU info, models, available commands |
| `vlx init` | Auto-discover vLLM installation and write config |
| `vlx download [model]` | Search and download a model from HuggingFace |
| `vlx models [name]` | List models or show detail (fuzzy match) |
| `vlx tune <model> [profile]` | Probe context/concurrency limits, then benchmark |
| `vlx start [model]` | Configure and start serving — interactive picker if no args |
| `vlx stop` | Stop the vLLM service |
| `vlx status` | Show current serving config |
| `vlx fan [auto\|off\|30-100]` | GPU fan control with temp-based curve |
| `vlx doctor` | Check system readiness |

All commands support **fuzzy matching** — `vlx start qwen fp8` finds the right model.

---

## Configuration

vlx auto-discovers your vLLM installation on first run. Override with `~/.config/vx/config.yaml`:

```yaml
vllm_root: /opt/vllm
cuda_home: /usr/local/cuda
service_name: vllm
service_user: vllm
port: 8888
```

Run `vlx init` to regenerate from auto-discovery.

---

## Fan Control

`vlx fan auto` runs a temperature-based fan curve daemon with quiet-hours scheduling:

```
Temp (C)   Quiet hours (09-18)    Off hours (18-09)
<=35       30%                    30%
  |        |  linear ramp          |  linear ramp
 80        60% (cap)              100%
 88+       100% (emergency)       100% (emergency)
```

The emergency override at 88C ignores quiet hours — thermal safety always wins.

```bash
vlx fan              # interactive — view status, configure
vlx fan auto         # start daemon with default quiet hours
vlx fan off          # stop daemon, restore NVIDIA auto
vlx fan 80           # fixed 80%
```

Requires [Coolbits](docs/troubleshooting.md#coolbits-setup) enabled in X11 config for manual fan control.

---

## How It Works

```
vlx tune <model>           # calculate limits (instant, no GPU needed)
   |
   +-- Read model architecture (config.json: kv_heads, head_dim, layers)
   +-- Read GPU info (nvidia-smi: total VRAM)
   +-- KV formula: available_kv = VRAM × util − model_weights − overhead
   +-- Output: context × concurrency table for auto and fp8 KV dtypes

vlx start <model>          # interactive config picker → serve
   |
   +-- Pick context window, KV dtype, concurrent slots from limits
   +-- Write config YAML, start vLLM via systemd
```

---

## Prerequisites

You need these before installing vlx:

**1. NVIDIA GPU + drivers**

```bash
nvidia-smi                        # should show your GPU
```

No output? Install drivers from [nvidia.com/drivers](https://www.nvidia.com/drivers).

**2. CUDA toolkit** (needed for first-run JIT compilation)

```bash
nvcc --version                    # should show CUDA version
# If missing:
sudo apt install nvidia-cuda-toolkit
```

**3. vLLM 0.18+**

```bash
pip install vllm                  # or: conda install -c conda-forge vllm
vllm --version
```

See [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation.html) for detailed options.

**4. systemd** (for service management)

vlx uses systemd to manage the vLLM service. Most Linux servers have it. You'll need a service unit — `vlx init` will detect an existing one, or see [troubleshooting](docs/troubleshooting.md) for setup.

**5. sudo access** (for systemctl, fan control)

---

## Setup

```bash
pip install vlx
vlx init                          # scans GPU, vLLM, CUDA, systemd — writes config
vlx doctor                        # verify everything is ready
```

Or from source:

```bash
git clone https://github.com/Gavin-Qiao/vlx.git ~/vlx
cd ~/vlx
uv sync && uv pip install -e .
sudo ln -sf ~/vlx/.venv/bin/vlx /usr/local/bin/vlx
vlx init
```

**Multi-user:** Install to a shared venv so all users can run vlx:

```bash
sudo uv venv /opt/vlx-app --python 3.12
sudo uv pip install --python /opt/vlx-app/bin/python vlx
sudo ln -sf /opt/vlx-app/bin/vlx /usr/local/bin/vlx
# Each user runs: vlx init
```

**Development:**

```bash
uv run pytest tests/              # 126 tests
uv run ruff check src/ tests/     # linting
uv run mypy src/vlx/              # type checking
```

---

## Project Structure

```
src/vlx/
  cli.py        Typer CLI — commands, interactive pickers, dashboard
  config.py     Auto-discovery, VxConfig, YAML/JSON I/O
  models.py     Model detection, fuzzy matching, metadata
  gpu.py        GPU info, VRAM calculations, fan speed control
  fan.py        Fan curve daemon — temp polling, quiet hours, emergency override
  probe.py      Binary search for context/concurrency/kv-dtype limits
  tune.py       Benchmark orchestration, profile parameter generation
  serve.py      systemd service management (start/stop/symlink)
```

---

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md) for GPU crash recovery, JIT compilation, driver management, and fan control setup.

---

## License

[MIT](LICENSE)
