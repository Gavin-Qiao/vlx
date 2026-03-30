# Troubleshooting

Hard-won lessons from running vLLM on NVIDIA workstation GPUs.

## GPU Crashes (Xid Errors)

### Xid 8 — GPU Stopped Processing

**Symptom:** vLLM dies after hours of stable inference. Kernel log shows:
```
NVRM: krcWatchdog_IMPL: RC watchdog: GPU is probably locked!
NVRM: Xid (PCI:0000:02:00): 8, pid=..., name=VLLM::EngineCor
```

**Root cause:** The GPU's recovery counter watchdog detected a hang. On Blackwell (SM120), this correlates with CUDA graphs under sustained FP8 load (see [vllm-project/vllm#35659](https://github.com/vllm-project/vllm/issues/35659)).

**Recovery:**
1. The GPU usually recovers automatically after the faulting process exits — no reset needed.
2. `nvidia-smi --gpu-reset` is deprecated as of driver 570+.
3. If the GPU is unresponsive, reload kernel modules: `sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia`
4. Last resort: reboot.

**Prevention:**
- Use the production driver branch (595.x), not the new-feature branch (590.x).
- `vlx fan auto` keeps temps below throttle threshold.
- systemd `Restart=always` with exponential backoff recovers automatically.
- If crashes persist, try `--enforce-eager` (disables CUDA graphs, ~2.3x throughput cost).

### vLLM exits cleanly but systemd doesn't restart

**Cause:** vLLM's APIServer shuts down cleanly (exit 0) after an EngineCore crash. `Restart=on-failure` ignores exit 0.

**Fix:** Use `Restart=always` in the systemd unit.

## Fan Control

### Why override NVIDIA's auto curve?

The default auto curve on workstation GPUs (RTX PRO series) caps fan speed conservatively for noise. Under sustained 300W inference, this can allow temps to reach 85-90°C, causing:
- Thermal throttling (reduced inference throughput)
- Accelerated component aging (electromigration, capacitor degradation)
- Increased risk of Xid errors

### Coolbits setup

Fan control via `nvidia-settings` requires Coolbits enabled in X11 config:

```bash
# /etc/X11/xorg.conf
Section "Device"
    Identifier     "GPU0"
    Driver         "nvidia"
    BusID          "PCI:2:0:0"    # check with: nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader
    Option         "Coolbits" "4"  # bit 2 = manual fan control
EndSection
```

On headless systems, `vlx fan` uses a temporary Xvfb virtual display — no persistent X server needed.

### Hardware thermal failsafe

Even if the fan daemon crashes and the fan is stuck at 30%, the GPU has hardware protection:
- GPU Boost reduces clocks starting at ~83°C
- Aggressive power limiting at ~90°C
- Hardware shutdown at ~100-105°C (cannot be overridden by software)

The `vlx fan` emergency override (100% fan at 88°C regardless of quiet hours) is software-level defense. The hardware failsafe is always present underneath.

## JIT Compilation

### FlashInfer JIT cache

vLLM 0.18+ uses FlashInfer with just-in-time compiled CUDA kernels. First run for each model/KV-dtype combination triggers JIT compilation that takes 2-5 minutes.

**Where the cache lives:** `$VLLM_ROOT/.cache/flashinfer/`

**Problem:** If vLLM runs as a different user (e.g., systemd `User=vllm`), the JIT cache must be writable by that user. First-time startup will be slow.

**The preheat solution:** `vlx tune` runs a preheat step that starts vLLM briefly as the service user to build the JIT cache before benchmarking. This uses `sudo -u <service_user>` with the correct `CUDA_HOME`, `PATH`, and `HF_HOME` environment.

### torch.compile cache

vLLM also uses `torch.compile` which has its own cache at `$VLLM_ROOT/.cache/vllm/torch_compile_cache/`. Same ownership considerations apply.

### CUDA_HOME and PATH

The JIT compiler needs:
- `nvcc` accessible (CUDA toolkit bin directory in PATH)
- `CUDA_HOME` set to the CUDA toolkit root
- Matching CUDA version between the toolkit and the PyTorch build

If JIT compilation fails with `nvcc not found` or architecture mismatches, check:
```bash
which nvcc
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

### ProtectSystem=strict breaks JIT

Do NOT use `ProtectSystem=strict` in the vLLM systemd unit — it makes `/usr` read-only, preventing nvcc from writing temporary files during JIT compilation. The `gpu-fan.service` can use it (fan control doesn't need JIT), but the main `vllm.service` cannot.

## Driver Management

### Which driver branch to use

| Branch | Status | Use |
|--------|--------|-----|
| 570.x | Old production | Pre-Blackwell only |
| 575.x | Early Blackwell | Known issues, avoid |
| 590.x | New-feature branch | Experimental, not for production |
| 595.x | **Current production** | Recommended for Blackwell |

Check your branch: `nvidia-smi` shows the driver version in the header.

### Open vs. proprietary kernel modules

Blackwell GPUs (RTX 50-series, RTX PRO 5000/6000) **require open kernel modules**. The proprietary modules do not support Blackwell at all. Always use packages with `-open` suffix (e.g., `nvidia-driver-595-server-open` or `nvidia-open` from the CUDA repo).

### nvidia-settings version mismatch

If `nvidia-settings --version` shows a different version than `nvidia-smi`, you have a partial driver upgrade. This can cause subtle issues. Fix by upgrading the full driver stack to match.

## Headless Operation

### Disabling the display manager

For 24/7 inference, disable the display manager to free GPU memory and simplify driver reloads:

```bash
sudo systemctl disable gdm    # or lightdm/sddm
sudo systemctl set-default multi-user.target
```

GUI is still available on demand: `sudo systemctl start gdm`

### GPU memory with no display

With no display manager: ~2 MiB GPU memory used (driver overhead only).
With Xorg/GDM running: ~15 MiB (Xorg frame buffer).

This is negligible for 48 GB, but matters for driver reload — fewer processes holding the GPU open means cleaner `rmmod`.
