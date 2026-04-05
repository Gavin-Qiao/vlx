"""Start/stop vLLM via systemd."""

import subprocess
from pathlib import Path

from vserve.config import cfg


def _systemctl(action: str, timeout: int = 30) -> tuple[bool, str, str]:
    try:
        result = subprocess.run(
            ["sudo", "systemctl", action, cfg().service_name],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "", f"systemctl {action} timed out after {timeout}s"
    except Exception as exc:
        return False, "", str(exc)
    return result.returncode == 0, result.stdout.strip(), result.stderr.strip()


def _update_active_symlink(config_path: Path) -> None:
    active = cfg().active_yaml
    active.parent.mkdir(parents=True, exist_ok=True)
    try:
        active.unlink(missing_ok=True)
        active.symlink_to(config_path.resolve())
    except OSError as exc:
        raise RuntimeError(f"failed to update active config link {active}: {exc}") from None


def is_vllm_running() -> bool:
    ok, output, err = _systemctl("is-active", timeout=5)
    status = output.strip().lower()
    if ok and status == "active":
        return True
    if status in {"inactive", "failed"}:
        return False
    if status in {"activating", "deactivating", "reloading"}:
        raise RuntimeError(f"systemctl is-active {cfg().service_name} is transitional: {status}")
    if "could not be found" in err.lower():
        return False
    if err:
        raise RuntimeError(f"systemctl is-active {cfg().service_name} failed: {err}")
    return False


def start_vllm(config_path: Path) -> None:
    _update_active_symlink(config_path)
    ok, _out, err = _systemctl("start")
    if not ok:
        raise RuntimeError(f"systemctl start failed: {err}")


def stop_vllm() -> None:
    ok, _out, err = _systemctl("stop")
    if not ok:
        raise RuntimeError(f"systemctl stop failed: {err}")
