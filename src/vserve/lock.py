"""File-based locking for multi-user concurrency.

Uses fcntl.flock() on files in /run/lock/vserve/ — advisory locks that auto-release
on process exit or crash. Lock metadata (pid, user, timestamp) is written for
diagnostics when a lock is held by another process.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

LOCK_DIR = Path("/run/lock/vserve")


@dataclass
class LockInfo:
    pid: int
    user: str
    since: str
    command: str


class LockHeld(Exception):
    """Raised when a lock is already held by another process."""

    def __init__(self, name: str, info: LockInfo | None) -> None:
        self.name = name
        self.info = info
        super().__init__(self.message())

    def message(self) -> str:
        if self.info:
            elapsed = _format_elapsed(self.info.since)
            return (
                f"Locked by {self.info.user} (pid {self.info.pid})"
                f" since {self.info.since} ({elapsed} ago)"
                f"\n  {self.info.command}"
            )
        return f"Lock '{self.name}' is held by another process"


class VserveLock:
    """Advisory file lock via fcntl.flock().

    Usage::

        with VserveLock("fan", "fan daemon"):
            ...  # exclusive access

    The lock is held as long as the file descriptor is open.
    On process crash the OS releases it automatically.
    """

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self._path = LOCK_DIR / f"vserve-{name}.lock"
        self._fd: int | None = None

    def acquire(self) -> None:
        LOCK_DIR.mkdir(mode=0o1777, parents=True, exist_ok=True)
        # mkdir mode is masked by umask, and exist_ok skips existing dirs.
        # Ensure sticky + world-writable so any user can create lock files.
        try:
            os.chmod(str(LOCK_DIR), 0o1777)
        except PermissionError:
            pass
        # A prior sudo run may have left a root-owned lock file.
        # Try to fix permissions, or delete and recreate if stale.
        if self._path.exists():
            try:
                os.chmod(str(self._path), 0o666)
            except PermissionError:
                # Can't chmod — check if the lock is actually held
                info = _read_lock_info(self._path)
                if info and _pid_alive(info.pid):
                    raise LockHeld(self.name, info)
                # Stale lock from another user — try to remove
                try:
                    self._path.unlink()
                except PermissionError:
                    raise PermissionError(
                        f"Cannot acquire lock: {self._path} is owned by another user. "
                        f"Run: sudo rm {self._path}"
                    )
        try:
            self._fd = os.open(str(self._path), os.O_WRONLY | os.O_CREAT, 0o666)
        except PermissionError:
            raise PermissionError(
                f"Cannot create lock file: {self._path}\n"
                f"Run: sudo chmod 1777 {LOCK_DIR}"
            ) from None
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            info = _read_lock_info(self._path)
            os.close(self._fd)
            self._fd = None
            raise LockHeld(self.name, info)

        # Write metadata for other processes to read
        info = LockInfo(
            pid=os.getpid(),
            user=os.environ.get("USER", "?"),
            since=time.strftime("%Y-%m-%d %H:%M:%S"),
            command=self.description,
        )
        os.ftruncate(self._fd, 0)
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.write(self._fd, json.dumps(asdict(info)).encode() + b"\n")

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
            try:
                self._path.unlink(missing_ok=True)
            except OSError:
                pass  # another user's sticky-bit protection

    def __enter__(self) -> VserveLock:
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_lock_info(path: Path) -> LockInfo | None:
    try:
        data = json.loads(path.read_text().strip())
        return LockInfo(**data)
    except Exception:
        return None


def wait_for_release(name: str, timeout: float = 5.0) -> bool:
    """Block until the named lock is free, or timeout. Returns True if free."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            lock = VserveLock(name, "wait-probe")
            lock.acquire()
            lock.release()
            return True
        except (LockHeld, PermissionError):
            time.sleep(0.2)
    return False


def notify_user(username: str, message: str) -> bool:
    """Send a message to a user's terminal(s) via /dev/pts. Requires tty group.

    Returns True if at least one terminal was written to.
    """
    import subprocess
    sent = False
    try:
        result = subprocess.run(
            ["who"], capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == username:
                tty = f"/dev/{parts[1]}"
                try:
                    with open(tty, "w") as f:
                        f.write(f"\r\n\033[1;36m[vserve]\033[0m {message}\r\n")
                    sent = True
                except OSError:
                    continue
    except Exception:
        pass
    return sent


SESSION_PATH = LOCK_DIR / "vserve-session.json"


@dataclass
class SessionInfo:
    user: str
    since: str
    model: str


class SessionHeld(Exception):
    """Raised when another user owns the active session."""

    def __init__(self, info: SessionInfo) -> None:
        self.info = info
        super().__init__(self.message())

    def message(self) -> str:
        elapsed = _format_elapsed(self.info.since)
        return (
            f"GPU session owned by {self.info.user}"
            f" since {self.info.since} ({elapsed} ago)"
            f"\n  Model: {self.info.model}"
        )


def write_session(model: str) -> None:
    """Record current user as the GPU session owner.

    Uses open+write on a world-writable file (not atomic rename) so that
    any user can overwrite the session in a sticky-bit directory.
    """
    LOCK_DIR.mkdir(mode=0o1777, parents=True, exist_ok=True)
    try:
        os.chmod(str(LOCK_DIR), 0o1777)
    except PermissionError:
        pass
    info = SessionInfo(
        user=os.environ.get("USER", "?"),
        since=time.strftime("%Y-%m-%d %H:%M:%S"),
        model=model,
    )
    payload = json.dumps(asdict(info)).encode() + b"\n"
    # Create or overwrite with world-writable permissions so any user
    # can update the session in a sticky-bit directory.
    fd = os.open(str(SESSION_PATH), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)
    try:
        os.chmod(str(SESSION_PATH), 0o666)
    except PermissionError:
        pass


def read_session() -> SessionInfo | None:
    try:
        text = SESSION_PATH.read_text().strip()
        if not text:
            return None  # truncated (cleared by another user)
        data = json.loads(text)
        return SessionInfo(**data)
    except FileNotFoundError:
        return None
    except Exception:
        # Corrupt or partially-written session file
        import sys
        print(f"[vserve] warning: corrupt session file {SESSION_PATH}", file=sys.stderr)
        return None


def clear_session() -> None:
    try:
        SESSION_PATH.unlink(missing_ok=True)
    except PermissionError:
        # Sticky-bit dir: can't unlink another user's file. Truncate instead.
        try:
            fd = os.open(str(SESSION_PATH), os.O_WRONLY | os.O_TRUNC)
            os.close(fd)
        except OSError:
            pass
    except OSError:
        pass


def check_session() -> None:
    """Raise SessionHeld if another user owns an active session.

    Clears stale sessions (vLLM not running).  No-op if the current
    user owns the session (they can restart their own model).
    """
    from vserve.backends import any_backend_running

    info = read_session()
    if info is None:
        return
    if not any_backend_running():
        clear_session()
        return
    current_user = os.environ.get("USER", "?")
    if info.user == current_user:
        return
    raise SessionHeld(info)


def _format_elapsed(since: str) -> str:
    try:
        t = time.mktime(time.strptime(since, "%Y-%m-%d %H:%M:%S"))
        delta = int(time.time() - t)
        if delta < 60:
            return f"{delta}s"
        if delta < 3600:
            return f"{delta // 60}m"
        return f"{delta // 3600}h {(delta % 3600) // 60}m"
    except Exception:
        return "?"
