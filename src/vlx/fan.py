"""GPU fan curve daemon — temperature-based with quiet hours."""

import logging
import os
import signal
import subprocess
import threading
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from vlx.config import cfg

LOG_PATH: Path  # resolved lazily
PID_PATH: Path
STATE_PATH: Path
_paths_resolved = False


def _resolve_paths() -> None:
    global LOG_PATH, PID_PATH, STATE_PATH, _paths_resolved
    if _paths_resolved:
        return
    c = cfg()
    LOG_PATH = c.logs_dir / "vx-fan.log"
    PID_PATH = c.run_dir / "vx-fan.pid"
    STATE_PATH = c.run_dir / "vx-fan.json"
    _paths_resolved = True

MIN_FAN = 30
POLL_INTERVAL = 5
SUBPROCESS_TIMEOUT = 10
MAX_CONSECUTIVE_FAILURES = 30  # 30 × 5s = 2.5 min → exit for systemd restart
FAILURE_RESET_THRESHOLD = 5   # force re-apply after 5 failures
HYSTERESIS = 4  # °C deadband — ramp-down requires temp to drop this much below ramp-up threshold
MAX_RAMP_PER_CYCLE = 5  # max fan % change per poll cycle to avoid audible jumps
SMOOTHING_SAMPLES = 3   # moving average window for temperature readings

_log = logging.getLogger("vlx.fan")

# Piecewise-linear curve: (temp_C, fan_%).
# Shallow at low temps, steep near throttle zone.
# Used for non-quiet hours (cap=100%).
_CURVE = [
    (35, 30),
    (55, 35),
    (65, 50),
    (75, 75),
    (80, 100),
]

EMERGENCY_TEMP = 88  # ignore quiet hours above this


def read_state() -> dict | None:
    """Read the running daemon's config. Returns None if not running."""
    _resolve_paths()
    if not STATE_PATH.exists():
        return None
    import json
    try:
        return json.loads(STATE_PATH.read_text())
    except (ValueError, OSError):
        return None


def _interpolate_curve(temp: int, cap: int) -> int:
    """Evaluate the piecewise-linear curve, scaling to the given cap."""
    if temp <= _CURVE[0][0]:
        return MIN_FAN
    if temp >= _CURVE[-1][0]:
        return cap

    # Find the segment
    for i in range(len(_CURVE) - 1):
        t0, s0 = _CURVE[i]
        t1, s1 = _CURVE[i + 1]
        if t0 <= temp <= t1:
            frac = (temp - t0) / (t1 - t0)
            raw = s0 + frac * (s1 - s0)
            return min(int(raw), cap)
    return cap  # shouldn't reach here


def compute_fan_speed(
    temp: int,
    hour: int,
    quiet_start: int = 9,
    quiet_end: int = 18,
    quiet_max: int = 60,
) -> int:
    """Map GPU temperature to fan speed using piecewise-linear curve.

    During quiet hours, the curve is scaled so the max output is quiet_max
    instead of 100%, with a smooth ramp (no hard cap cliff).
    Emergency override at EMERGENCY_TEMP always returns 100%.
    """
    if temp >= EMERGENCY_TEMP:
        return 100

    if quiet_start < quiet_end:
        in_quiet = quiet_start <= hour < quiet_end
    else:  # wraps midnight, e.g. 22:00-06:00
        in_quiet = hour >= quiet_start or hour < quiet_end
    cap = quiet_max if in_quiet else 100

    return _interpolate_curve(temp, cap)


def _get_gpu_temp() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=SUBPROCESS_TIMEOUT,
    )
    return int(result.stdout.strip())


def _apply_fan(percent: int, env: dict[str, str]) -> None:
    """Set fan speed via nvidia-settings."""
    subprocess.run(
        ["sudo", "nvidia-settings", "-a", f"[fan:0]/GPUTargetFanSpeed={percent}"],
        capture_output=True,
        env=env,
        check=True,
        timeout=SUBPROCESS_TIMEOUT,
    )


def _start_xvfb() -> subprocess.Popen[bytes]:
    """Start Xvfb with security hardening."""
    proc = subprocess.Popen(
        ["sudo", "Xvfb", ":99", "-screen", "0", "1024x768x24", "-nolisten", "tcp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)
    return proc


def _kill_xvfb(xvfb: subprocess.Popen[bytes]) -> None:
    """Terminate Xvfb, escalate to SIGKILL if needed."""
    xvfb.terminate()
    try:
        xvfb.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _log.warning("Xvfb did not exit on SIGTERM, sending SIGKILL")
        xvfb.kill()
        try:
            xvfb.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


def _setup_logging() -> None:
    if _log.handlers:
        return
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(LOG_PATH, maxBytes=2 * 1024 * 1024, backupCount=2)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _log.addHandler(handler)
    _log.setLevel(logging.INFO)


def run_daemon(quiet_start: int = 9, quiet_end: int = 18, quiet_max: int = 60) -> None:
    """Main fan curve loop. Blocks until SIGTERM."""
    import json
    from vlx.lock import VlxLock, LockHeld

    _resolve_paths()
    _setup_logging()

    try:
        _fan_lock = VlxLock("fan", "fan daemon")
        _fan_lock.acquire()
    except LockHeld as exc:
        _log.error("Another fan daemon is running: %s", exc.message())
        return

    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()))
    STATE_PATH.write_text(json.dumps({
        "quiet_start": quiet_start,
        "quiet_end": quiet_end,
        "quiet_max": quiet_max,
    }))

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    env = {**os.environ, "DISPLAY": ":99"}
    xvfb = _start_xvfb()

    try:
        # Enable manual fan control — fail fast if this doesn't work
        subprocess.run(
            ["sudo", "nvidia-settings", "-a", "[gpu:0]/GPUFanControlState=1"],
            capture_output=True,
            env=env,
            check=True,
            timeout=SUBPROCESS_TIMEOUT,
        )

        current_speed = -1
        consecutive_failures = 0
        temp_history: list[int] = []  # rolling window for smoothing
        _log.info(
            "Fan daemon started (quiet %02d:00-%02d:00 cap %d%%)",
            quiet_start, quiet_end, quiet_max,
        )

        while not stop.is_set():
            try:
                # Restart Xvfb if it died
                if xvfb.poll() is not None:
                    _log.warning("Xvfb died (rc=%s), restarting", xvfb.returncode)
                    xvfb = _start_xvfb()
                    subprocess.run(
                        ["sudo", "nvidia-settings", "-a", "[gpu:0]/GPUFanControlState=1"],
                        capture_output=True, env=env, check=True,
                        timeout=SUBPROCESS_TIMEOUT,
                    )

                raw_temp = _get_gpu_temp()

                # Temperature smoothing — moving average
                temp_history.append(raw_temp)
                if len(temp_history) > SMOOTHING_SAMPLES:
                    temp_history.pop(0)
                temp = sum(temp_history) // len(temp_history)

                hour = datetime.now().hour
                target = compute_fan_speed(temp, hour, quiet_start, quiet_end, quiet_max)

                # Hysteresis — only ramp down if temp dropped enough
                if target < current_speed and current_speed != -1:
                    ramp_down_temp = compute_fan_speed(
                        temp + HYSTERESIS, hour, quiet_start, quiet_end, quiet_max,
                    )
                    if ramp_down_temp >= current_speed:
                        target = current_speed  # hold — not cool enough yet

                # Rate limiting — max change per cycle
                if current_speed != -1:
                    delta = target - current_speed
                    if abs(delta) > MAX_RAMP_PER_CYCLE:
                        target = current_speed + (MAX_RAMP_PER_CYCLE if delta > 0 else -MAX_RAMP_PER_CYCLE)

                target = max(MIN_FAN, min(100, target))

                if target != current_speed:
                    _apply_fan(target, env)
                    _log.info("temp=%d°C (raw=%d) hour=%02d:00 fan=%d%%", temp, raw_temp, hour, target)
                    current_speed = target

                consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    _log.critical(
                        "Too many consecutive failures (%d), exiting for restart",
                        consecutive_failures,
                    )
                    break
                if consecutive_failures >= FAILURE_RESET_THRESHOLD:
                    current_speed = -1  # force re-apply on next success
                if consecutive_failures <= 3 or consecutive_failures % 10 == 0:
                    _log.exception("Fan update failed (%d consecutive)", consecutive_failures)

            stop.wait(timeout=POLL_INTERVAL)
    finally:
        # Restore auto fan control — best effort
        restored = False
        try:
            subprocess.run(
                ["sudo", "nvidia-settings", "-a", "[gpu:0]/GPUFanControlState=0"],
                capture_output=True, env=env, timeout=SUBPROCESS_TIMEOUT,
            )
            restored = True
        except Exception:
            _log.exception("Failed to restore auto fan control")
        _kill_xvfb(xvfb)
        PID_PATH.unlink(missing_ok=True)
        STATE_PATH.unlink(missing_ok=True)
        _fan_lock.release()
        if restored:
            _log.info("Fan daemon stopped, auto control restored")
        else:
            _log.warning("Fan daemon stopped, auto control NOT restored — run: vlx fan off")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--quiet-start", type=int, default=9)
    p.add_argument("--quiet-end", type=int, default=18)
    p.add_argument("--quiet-max", type=int, default=60)
    args = p.parse_args()
    run_daemon(args.quiet_start, args.quiet_end, args.quiet_max)
