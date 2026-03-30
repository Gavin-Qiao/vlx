"""Start/stop vLLM via systemd."""

import subprocess
from pathlib import Path

from vlx.config import cfg


def _systemctl(action: str) -> tuple[bool, str]:
    result = subprocess.run(
        ["sudo", "systemctl", action, cfg().service_name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stdout.strip()


def _update_active_symlink(config_path: Path) -> None:
    active = cfg().active_yaml
    active.unlink(missing_ok=True)
    active.symlink_to(config_path.resolve())


def is_vllm_running() -> bool:
    ok, output = _systemctl("is-active")
    return ok and output == "active"


def start_vllm(config_path: Path) -> None:
    _update_active_symlink(config_path)
    _systemctl("start")


def stop_vllm() -> None:
    _systemctl("stop")
