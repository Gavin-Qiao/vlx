"""Probe context and concurrency limits for a model."""
from __future__ import annotations
from typing import Callable

import logging
import subprocess
import time
from datetime import datetime, timezone

from vlx.config import LOGS_DIR, VLLM_BIN, cfg
from vlx.models import ModelInfo, quant_flag

_log = logging.getLogger("vlx.probe")

CONTEXT_STEPS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
CONCURRENCY_STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256]

_PROBE_PORT = 18188
_HEALTH_TIMEOUT = 300
_CLEANUP_TIMEOUT = 30  # seconds to wait for GPU memory release


def setup_probe_logging() -> str:
    """Set up file logging for probe runs. Call once before probe_model."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"probe-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    ))
    _log.addHandler(handler)
    _log.setLevel(logging.DEBUG)
    _log.info("Probe log started: %s", log_file)
    return str(log_file)


def _gpu_memory_used_mib() -> int:
    """Return current GPU memory usage in MiB via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5,
        )
        return int(out.decode().strip().split("\n")[0])
    except Exception:
        return 0


def _kill_probe_processes(proc: subprocess.Popen[bytes]) -> None:
    """Kill vLLM process group and wait for GPU memory release."""
    import os
    import signal
    import socket

    pgid: int | None = None
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        pass

    # SIGTERM the process group
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    # Wait for parent to exit
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass

    # SIGKILL the entire group if anything survived
    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass

    # Snapshot GPU memory right after kill — wait until it drops back
    # (EngineCore may take seconds to release VRAM even after SIGKILL)
    baseline = _gpu_memory_used_mib()
    _log.debug("cleanup: baseline gpu=%dMiB, waiting for release...", baseline)
    waited = 0.0
    for _ in range(_CLEANUP_TIMEOUT * 2):  # poll every 0.5s
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            port_free = s.connect_ex(("127.0.0.1", _PROBE_PORT)) != 0
        mem_now = _gpu_memory_used_mib()
        mem_released = mem_now < 100 or mem_now < baseline - 100
        if port_free and mem_released:
            _log.debug("cleanup: done in %.1fs gpu=%dMiB port_free=%s", waited, mem_now, port_free)
            break
        time.sleep(0.5)
        waited += 0.5
    else:
        mem_now = _gpu_memory_used_mib()
        _log.warning("cleanup: TIMEOUT after %.1fs gpu=%dMiB — memory may not be released!", waited, mem_now)


def _read_stderr_file(path: str) -> str:
    """Read stderr temp file, return content as string."""
    try:
        with open(path, "rb") as f:
            return f.read().decode(errors="replace")
    except OSError:
        return ""


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

    gpu_before = _gpu_memory_used_mib()
    _log.info(
        "START ctx=%d kv=%s seqs=%d gpu_before=%dMiB | %s",
        context_len, kv_dtype, max_num_seqs, gpu_before, " ".join(cmd),
    )

    import os as _os
    import tempfile
    stderr_file = tempfile.NamedTemporaryFile(prefix="vx-probe-", suffix=".stderr", delete=False)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=stderr_file,
        preexec_fn=_os.setsid,
    )
    t0 = time.monotonic()
    try:
        deadline = t0 + _HEALTH_TIMEOUT
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                elapsed = time.monotonic() - t0
                stderr_text = _read_stderr_file(stderr_file.name)
                gpu_after = _gpu_memory_used_mib()
                _log.warning(
                    "EXITED rc=%d after %.1fs gpu_after=%dMiB ctx=%d kv=%s seqs=%d\n"
                    "--- stderr (last 2000 chars) ---\n%s\n--- end stderr ---",
                    proc.returncode, elapsed, gpu_after,
                    context_len, kv_dtype, max_num_seqs,
                    stderr_text[-2000:],
                )
                return False
            try:
                from urllib.request import urlopen
                resp = urlopen(f"http://127.0.0.1:{_PROBE_PORT}/health", timeout=2)
                if resp.status == 200:
                    elapsed = time.monotonic() - t0
                    gpu_after = _gpu_memory_used_mib()
                    _log.info(
                        "HEALTHY after %.1fs gpu_after=%dMiB ctx=%d kv=%s seqs=%d",
                        elapsed, gpu_after, context_len, kv_dtype, max_num_seqs,
                    )
                    return True
            except Exception:
                pass
            time.sleep(2)
        elapsed = time.monotonic() - t0
        stderr_text = _read_stderr_file(stderr_file.name)
        _log.warning(
            "TIMEOUT after %.1fs ctx=%d kv=%s seqs=%d\n"
            "--- stderr (last 2000 chars) ---\n%s\n--- end stderr ---",
            elapsed, context_len, kv_dtype, max_num_seqs,
            stderr_text[-2000:],
        )
        return False
    finally:
        _kill_probe_processes(proc)
        gpu_final = _gpu_memory_used_mib()
        _log.info("CLEANUP done gpu_final=%dMiB", gpu_final)
        try:
            _os.unlink(stderr_file.name)
        except OSError:
            pass


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
    setup_probe_logging()
    _log.info(
        "probe_model start: %s size=%.1fGB quant=%s max_pos=%d gpu_util=%.3f vram=%.1fGB",
        model_info.path, model_info.model_size_gb, model_info.quant_method,
        model_info.max_position_embeddings, gpu_mem_util, vram_total_gb,
    )

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


def preheat_jit(
    *, model_info: ModelInfo, gpu_mem_util: float, kv_dtype: str = "auto",
) -> bool:
    """Start vLLM as the service user to build flashinfer JIT cache.

    Returns True if the server reached healthy state.
    """
    import os
    import tempfile

    c = cfg()
    cmd = [
        "sudo", "-u", c.service_user,
        "env",
        f"PATH={c.vllm_bin.parent}:{c.cuda_home / 'bin'}:/usr/bin:/bin",
        f"CUDA_HOME={c.cuda_home}",
        f"HF_HOME={c.models_dir / '.hf_cache'}",
        str(VLLM_BIN), "serve", str(model_info.path),
        "--dtype", "bfloat16", "--trust-remote-code",
        "--host", "127.0.0.1", "--port", str(_PROBE_PORT),
        "--max-model-len", "4096",
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--kv-cache-dtype", kv_dtype,
        "--max-num-seqs", "1",
    ]
    qf = quant_flag(model_info.quant_method)
    if qf:
        cmd.extend(qf.split())

    _log.info("PREHEAT JIT kv=%s | %s", kv_dtype, " ".join(cmd))

    stderr_file = tempfile.NamedTemporaryFile(prefix="vx-preheat-", suffix=".stderr", delete=False)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=stderr_file, preexec_fn=os.setsid)
    t0 = time.monotonic()
    healthy = False
    try:
        # JIT compilation can take 2-3 minutes on first run
        deadline = t0 + 600
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                _log.warning("PREHEAT exited rc=%d after %.0fs", proc.returncode, time.monotonic() - t0)
                break
            try:
                from urllib.request import urlopen
                resp = urlopen(f"http://127.0.0.1:{_PROBE_PORT}/health", timeout=2)
                if resp.status == 200:
                    _log.info("PREHEAT healthy after %.0fs", time.monotonic() - t0)
                    healthy = True
                    break
            except Exception:
                pass
            time.sleep(2)
    finally:
        _kill_probe_processes(proc)
        try:
            os.unlink(stderr_file.name)
        except OSError:
            pass

    return healthy
