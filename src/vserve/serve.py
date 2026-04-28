"""Start/stop vLLM via systemd."""

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from vserve.config import cfg, write_active_manifest


def _systemctl(action: str, timeout: int = 30) -> tuple[bool, str, str]:
    command = ["systemctl", action, cfg().service_name]
    if action in {"start", "stop", "restart", "reload"}:
        command.insert(0, "sudo")
    try:
        result = subprocess.run(
            command,
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


def _snapshot_active_config() -> dict:
    active = cfg().active_yaml
    if active.is_symlink():
        return {"kind": "symlink", "target": active.resolve(strict=False)}
    try:
        if active.exists():
            return {
                "kind": "file",
                "content": active.read_bytes(),
                "mode": active.stat().st_mode,
            }
    except OSError:
        return {"kind": "missing"}
    return {"kind": "missing"}


def _restore_active_config(snapshot: dict) -> None:
    active = cfg().active_yaml
    active.parent.mkdir(parents=True, exist_ok=True)
    try:
        active.unlink(missing_ok=True)
        if snapshot.get("kind") == "symlink":
            active.symlink_to(Path(snapshot["target"]))
        elif snapshot.get("kind") == "file":
            active.write_bytes(snapshot["content"])
            active.chmod(snapshot["mode"] & 0o777)
    except OSError as exc:
        raise RuntimeError(f"failed to restore active config link {active}: {exc}") from None


def _write_vllm_manifest(config_path: Path, *, status: str, error: str | None = None) -> None:
    manifest = {
        "backend": "vllm",
        "service_name": cfg().service_name,
        "config_path": str(config_path.resolve()),
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if error:
        manifest["error"] = error
    write_active_manifest(manifest, cfg().run_dir / "active-manifest.json")


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
    snapshot = _snapshot_active_config()
    _update_active_symlink(config_path)
    _write_vllm_manifest(config_path, status="starting")
    ok, _out, err = _systemctl("start")
    if not ok:
        _restore_active_config(snapshot)
        _write_vllm_manifest(config_path, status="failed", error=err)
        raise RuntimeError(f"systemctl start failed: {err}")


def stop_vllm() -> None:
    ok, _out, err = _systemctl("stop")
    if not ok:
        raise RuntimeError(f"systemctl stop failed: {err}")
