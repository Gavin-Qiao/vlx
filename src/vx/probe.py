"""Probe context and concurrency limits for a model."""
from __future__ import annotations
from typing import Callable

import subprocess
import time
from datetime import datetime, timezone

from vx.config import VLLM_BIN
from vx.models import ModelInfo, quant_flag

CONTEXT_STEPS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
CONCURRENCY_STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256]

_PROBE_PORT = 18188
_HEALTH_TIMEOUT = 300


def _try_start_vllm(
    *,
    model_info: ModelInfo,
    context_len: int,
    kv_dtype: str,
    gpu_mem_util: float,
    max_num_seqs: int = 1,
) -> bool:
    cmd = [
        str(VLLM_BIN), "serve", str(model_info.path),
        "--dtype", "bfloat16", "--trust-remote-code",
        "--host", "127.0.0.1", "--port", str(_PROBE_PORT),
        "--max-model-len", str(context_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--kv-cache-dtype", kv_dtype,
        "--max-num-seqs", str(max_num_seqs),
    ]
    qf = quant_flag(model_info.quant_method)
    if qf:
        cmd.extend(qf.split())

    import os as _os
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, preexec_fn=_os.setsid)
    try:
        deadline = time.monotonic() + _HEALTH_TIMEOUT
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                if proc.stderr:
                    err = proc.stderr.read().decode(errors="replace")[-500:]
                    if err.strip():
                        import logging
                        logging.debug("vLLM probe failed: %s", err.strip())
                return False
            try:
                import requests
                resp = requests.get(f"http://127.0.0.1:{_PROBE_PORT}/health", timeout=2)
                if resp.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(2)
        return False
    finally:
        # Kill the entire process group (catches EngineCore child processes)
        import os
        import signal
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.kill()
            proc.wait()

        # Wait until port is actually free before next probe
        import socket
        for _ in range(30):  # up to 15s
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", _PROBE_PORT)) != 0:
                    break
            time.sleep(0.5)


def probe_context_limit(
    *, model_info: ModelInfo, context_len: int, kv_dtype: str, gpu_mem_util: float,
) -> bool:
    return _try_start_vllm(
        model_info=model_info, context_len=context_len,
        kv_dtype=kv_dtype, gpu_mem_util=gpu_mem_util, max_num_seqs=1,
    )


def probe_concurrency_limit(
    *, model_info: ModelInfo, context_len: int, kv_dtype: str, gpu_mem_util: float,
) -> int | None:
    """Binary search for max max-num-seqs that doesn't OOM."""
    # First check if even 1 slot works
    if not _try_start_vllm(
        model_info=model_info, context_len=context_len,
        kv_dtype=kv_dtype, gpu_mem_util=gpu_mem_util, max_num_seqs=1,
    ):
        return None

    # Binary search through CONCURRENCY_STEPS
    lo, hi = 0, len(CONCURRENCY_STEPS) - 1
    best = CONCURRENCY_STEPS[0]  # 1 is known to work

    while lo <= hi:
        mid = (lo + hi) // 2
        seqs = CONCURRENCY_STEPS[mid]
        ok = _try_start_vllm(
            model_info=model_info, context_len=context_len,
            kv_dtype=kv_dtype, gpu_mem_util=gpu_mem_util, max_num_seqs=seqs,
        )
        if ok:
            best = seqs
            lo = mid + 1
        else:
            hi = mid - 1

    return best


def probe_model(
    *, model_info: ModelInfo, gpu_mem_util: float, vram_total_gb: float,
    on_step: Callable | None = None,
) -> dict:
    """Full probe. on_step(ctx, kv_dtype, phase, result) is called for live updates.

    phase: "context" (testing if ctx starts) or "concurrency" (binary search slots)
    result: True/False for context, int|None for concurrency
    """
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
                if on_step:
                    on_step(ctx, kv_dtype, "context", False)
                continue

            if on_step:
                on_step(ctx, kv_dtype, "concurrency", None)

            # Single pass: concurrency probe starts at max_num_seqs=1,
            # so it doubles as the context viability check.
            max_seqs = probe_concurrency_limit(
                model_info=model_info, context_len=ctx,
                kv_dtype=kv_dtype, gpu_mem_util=gpu_mem_util,
            )
            limits[key][kv_dtype] = max_seqs

            if max_seqs is None:
                hit_ceiling = True
                if on_step:
                    on_step(ctx, kv_dtype, "context", False)
            else:
                if on_step:
                    on_step(ctx, kv_dtype, "concurrency", max_seqs)

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
