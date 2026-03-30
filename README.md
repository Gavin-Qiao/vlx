<div align="center">

# vx

**A modern CLI for managing vLLM inference on GPU workstations.**

Probe limits. Benchmark profiles. Tune configs. Serve models. Control fans.

![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776ab?style=flat-square&logo=python&logoColor=white)
![vLLM 0.18+](https://img.shields.io/badge/vLLM-0.18+-ff6f00?style=flat-square)
![Ruff](https://img.shields.io/badge/code%20style-ruff-d4aa00?style=flat-square&logo=ruff&logoColor=white)
![mypy](https://img.shields.io/badge/type%20checked-mypy-blue?style=flat-square)
![Tests](https://img.shields.io/badge/tests-86%20passed-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

</div>

---

## Why

Running vLLM on a workstation means juggling model downloads, context limits, concurrency tuning, benchmark configs, systemd services, and fan curves — all through fragile bash scripts.

**vx** replaces all of that with a single CLI. It probes your GPU's actual limits, benchmarks across workload profiles, generates optimized serving configs, and keeps everything running.

---

## Quick Start

```bash
git clone https://github.com/Gavin-Qiao/vx.git ~/vx
cd ~/vx && uv sync && uv pip install -e .

vx                        # dashboard
vx models                 # list downloaded models
vx tune <model>           # probe limits + benchmark
vx start <model>          # serve with a tuned profile
```

---

## Commands

| Command | Description |
|:--------|:------------|
| `vx` | Dashboard — GPU info, models, available commands |
| `vx models [name]` | List models or show detail (fuzzy match) |
| `vx tune <model> [profile]` | Probe context/concurrency limits, then benchmark |
| `vx start [model] [profile]` | Start serving — interactive picker if no args |
| `vx stop` | Stop the vLLM service |
| `vx status` | Show current serving config |
| `vx compare` | Compare models for a workload |
| `vx fan [auto\|off\|30-100]` | GPU fan control with temp-based curve |
| `vx doctor` | Check system readiness |
| `vx download` | Download a model from HuggingFace |

All commands support **fuzzy matching** — `vx start qwen fp8` finds the right model.

---

## Benchmark Profiles

vx benchmarks each model across four workload profiles using `vllm bench sweep`:

| Profile | Optimizes | Use Case |
|:--------|:----------|:---------|
| **interactive** | TTFT | Single-user chat — fastest first token |
| **batch** | Throughput | Many concurrent requests — max tokens/sec |
| **agentic** | TTFT @ max context | Tool-use and RAG — fast response with large inputs |
| **creative** | TPOT | Long-form generation — smooth token-by-token output |

---

## Fan Control

`vx fan auto` runs a temperature-based fan curve daemon with quiet-hours scheduling:

```
Temp (C)   Quiet hours (09-18)    Off hours (18-09)
<=35       30%                    30%
  |        |  linear ramp          |  linear ramp
 80        60% (cap)              100%
 88+       100% (emergency)       100% (emergency)
```

The emergency override at 88C ignores quiet hours — thermal safety always wins.

```bash
vx fan              # interactive — view status, configure
vx fan auto         # start daemon with default quiet hours
vx fan off          # stop daemon, restore NVIDIA auto
vx fan 80           # fixed 80%
```

---

## How It Works

```
vx tune <model>
   |
   +-- 1. Probe: binary search for max context x concurrency x kv-dtype
   |        (tries each combination against the real GPU)
   |
   +-- 2. Benchmark: sweep across the viable parameter space
   |        (vllm bench sweep, measures TTFT/TPOT/throughput)
   |
   +-- 3. Config: pick the best params per profile, write YAML
            (ready for vx start)
```

Configs are stored at `/opt/vllm/configs/models/` as per-model per-profile YAMLs. The active config is a symlink at `/opt/vllm/configs/active.yaml` consumed by the systemd service.

---

## Hardware

Built for and tested on:

| Component | Spec |
|:----------|:-----|
| GPU | NVIDIA RTX PRO 5000 Blackwell, 48 GB |
| Driver | 595.58.03 (production branch) |
| CUDA | 13.2 |
| vLLM | 0.18.0 |
| OS | Ubuntu 24.04, kernel 6.17 |

---

## Setup

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), vLLM installed at `/opt/vllm/`, NVIDIA GPU.

```bash
git clone https://github.com/Gavin-Qiao/vx.git ~/vx
cd ~/vx
uv sync
uv pip install -e .
sudo ln -sf ~/vx/.venv/bin/vx /usr/local/bin/vx
```

**Development:**

```bash
uv run pytest tests/              # 86 tests
uv run ruff check src/ tests/     # linting
uv run mypy src/vx/               # type checking
```

---

## Project Structure

```
src/vx/
  cli.py        Typer CLI — commands, interactive pickers, dashboard
  config.py     Config paths, YAML/JSON I/O, symlink management
  models.py     Model detection, fuzzy matching, metadata
  gpu.py        GPU info, VRAM calculations, fan speed control
  fan.py        Fan curve daemon — temp polling, quiet hours, emergency override
  probe.py      Binary search for context/concurrency/kv-dtype limits
  tune.py       Benchmark orchestration, profile parameter generation
  compare.py    Model comparison and workload filtering
  serve.py      systemd service management (start/stop/symlink)
```

---

## License

[MIT](LICENSE)
