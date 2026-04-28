"""Tests for vserve.lock — file-based multi-user locking."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from vserve.lock import (
    LockHeld, LockInfo, VserveLock, _format_elapsed, wait_for_release,
    check_session, write_session, read_session, SessionHeld, SessionUnknown,
)


# ── Helpers ──────────────────────────────────────────

@pytest.fixture(autouse=True)
def _use_tmp_lock_dir(tmp_path, monkeypatch):
    """Redirect LOCK_DIR to a temp dir for all tests."""
    monkeypatch.setattr("vserve.lock.LOCK_DIR", tmp_path)


# ── Basic acquire / release ──────────────────────────

def test_acquire_and_release(tmp_path):
    lock = VserveLock("test", "test op")
    lock.acquire()
    assert lock._fd is not None
    lock_file = tmp_path / "vserve-test.lock"
    assert lock_file.exists()
    # Metadata written
    data = json.loads(lock_file.read_text().strip())
    assert data["pid"] == os.getpid()
    assert data["command"] == "test op"
    lock.release()
    assert lock._fd is None


def test_context_manager(tmp_path):
    with VserveLock("ctx", "ctx test") as lock:
        assert lock._fd is not None
        assert (tmp_path / "vserve-ctx.lock").exists()
    assert lock._fd is None


def test_release_idempotent():
    lock = VserveLock("idem", "test")
    lock.acquire()
    lock.release()
    lock.release()  # should not raise


def test_release_keeps_lock_file_stable(tmp_path):
    lock = VserveLock("stable", "test")
    lock.acquire()
    lock_file = tmp_path / "vserve-stable.lock"
    first_inode = lock_file.stat().st_ino
    lock.release()

    assert lock_file.exists()

    next_lock = VserveLock("stable", "next")
    next_lock.acquire()
    try:
        assert lock_file.stat().st_ino == first_inode
        data = json.loads(lock_file.read_text().strip())
        assert data["command"] == "next"
    finally:
        next_lock.release()


def test_lock_refuses_symlink_file_without_clobbering_target(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me\n")
    lock_file = tmp_path / "vserve-unsafe.lock"
    lock_file.symlink_to(victim)

    with pytest.raises(PermissionError, match="Refusing"):
        VserveLock("unsafe", "test").acquire()

    assert victim.read_text() == "keep me\n"


def test_session_write_refuses_symlink_without_clobbering_target(tmp_path, monkeypatch):
    monkeypatch.setattr("vserve.lock.SESSION_PATH", tmp_path / "vserve-session.json")
    monkeypatch.setenv("USER", "alice")
    victim = tmp_path / "victim-session.txt"
    victim.write_text("do not touch\n")
    session_path = tmp_path / "vserve-session-alice.json"
    session_path.symlink_to(victim)

    with pytest.raises(PermissionError, match="Refusing"):
        write_session("model")

    assert victim.read_text() == "do not touch\n"


# ── Contention ───────────────────────────────────────

def test_second_acquire_raises(tmp_path):
    """A subprocess holds the lock; parent's acquire raises LockHeld."""
    holder_script = textwrap.dedent(f"""\
        import sys, os, fcntl, time
        sys.path.insert(0, "{Path(__file__).resolve().parent.parent / 'src'}")
        from vserve import lock
        lock.LOCK_DIR = __import__("pathlib").Path("{tmp_path}")
        lk = lock.VserveLock("contend", "holder process")
        lk.acquire()
        sys.stdout.write("locked\\n")
        sys.stdout.flush()
        time.sleep(10)
    """)
    proc = subprocess.Popen(
        [sys.executable, "-c", holder_script],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        # Wait for child to acquire
        line = proc.stdout.readline()
        assert line.strip() == "locked"

        # Parent should fail
        with pytest.raises(LockHeld) as exc_info:
            VserveLock("contend", "parent attempt").acquire()
        assert exc_info.value.info is not None
        assert exc_info.value.info.pid == proc.pid
        assert "holder process" in exc_info.value.info.command
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_lock_released_on_process_death(tmp_path):
    """After holder process dies, lock can be reacquired."""
    holder_script = textwrap.dedent(f"""\
        import sys
        sys.path.insert(0, "{Path(__file__).resolve().parent.parent / 'src'}")
        from vserve import lock
        lock.LOCK_DIR = __import__("pathlib").Path("{tmp_path}")
        lk = lock.VserveLock("die", "dying process")
        lk.acquire()
        sys.stdout.write("locked\\n")
        sys.stdout.flush()
        import time; time.sleep(10)
    """)
    proc = subprocess.Popen(
        [sys.executable, "-c", holder_script],
        stdout=subprocess.PIPE, text=True,
    )
    proc.stdout.readline()  # wait for lock
    proc.terminate()
    proc.wait(timeout=5)

    # Lock file may still exist, but flock is released
    lock = VserveLock("die", "reacquire")
    lock.acquire()  # should not raise
    lock.release()


# ── Per-model download lock names ────────────────────

def test_different_model_locks_independent(tmp_path):
    lock_a = VserveLock("download-Qwen--Qwen3-27B", "dl model A")
    lock_b = VserveLock("download-Meta--Llama-8B", "dl model B")
    lock_a.acquire()
    lock_b.acquire()  # should not raise — different lock names
    lock_a.release()
    lock_b.release()


def test_same_model_lock_contends(tmp_path):
    """Same lock name from subprocess → parent blocked."""
    holder_script = textwrap.dedent(f"""\
        import sys, time
        sys.path.insert(0, "{Path(__file__).resolve().parent.parent / 'src'}")
        from vserve import lock
        lock.LOCK_DIR = __import__("pathlib").Path("{tmp_path}")
        lk = lock.VserveLock("download-Qwen--Model", "downloading Qwen/Model")
        lk.acquire()
        sys.stdout.write("locked\\n")
        sys.stdout.flush()
        time.sleep(10)
    """)
    proc = subprocess.Popen(
        [sys.executable, "-c", holder_script],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        proc.stdout.readline()
        with pytest.raises(LockHeld):
            VserveLock("download-Qwen--Model", "parent dl").acquire()
    finally:
        proc.terminate()
        proc.wait(timeout=5)


# ── LockInfo and error messages ──────────────────────

def test_lock_held_message():
    info = LockInfo(pid=999, user="zhuo", since="2026-03-30 14:00:00", command="downloading model")
    exc = LockHeld("download", info)
    msg = exc.message()
    assert "zhuo" in msg
    assert "999" in msg
    assert "downloading model" in msg


def test_lock_held_no_info():
    exc = LockHeld("test", None)
    assert "held by another process" in exc.message()


# ── Elapsed time formatting ──────────────────────────

def test_format_elapsed_seconds():
    import time
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    result = _format_elapsed(now)
    assert result in ("0s", "1s")


def test_format_elapsed_invalid():
    assert _format_elapsed("garbage") == "?"


# ── Lock dir creation ────────────────────────────────

def test_lock_dir_created(tmp_path, monkeypatch):
    subdir = tmp_path / "nested" / "lock"
    monkeypatch.setattr("vserve.lock.LOCK_DIR", subdir)
    lock = VserveLock("mkdir", "test")
    lock.acquire()
    assert subdir.exists()
    lock.release()


# ── wait_for_release ─────────────────────────────────

def test_wait_for_release_free(tmp_path):
    """Returns True immediately when no one holds the lock."""
    assert wait_for_release("free", timeout=1.0) is True


def test_wait_for_release_held_then_freed(tmp_path):
    """Returns True after holder releases."""
    holder_script = textwrap.dedent(f"""\
        import sys, time
        sys.path.insert(0, "{Path(__file__).resolve().parent.parent / 'src'}")
        from vserve import lock
        lock.LOCK_DIR = __import__("pathlib").Path("{tmp_path}")
        lk = lock.VserveLock("handoff", "holder")
        lk.acquire()
        sys.stdout.write("locked\\n")
        sys.stdout.flush()
        time.sleep(1)
        lk.release()
    """)
    proc = subprocess.Popen(
        [sys.executable, "-c", holder_script],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        proc.stdout.readline()  # wait for lock
        assert wait_for_release("handoff", timeout=5.0) is True
    finally:
        proc.wait(timeout=5)


def test_wait_for_release_timeout(tmp_path):
    """Returns False when lock is held past timeout."""
    holder_script = textwrap.dedent(f"""\
        import sys, time
        sys.path.insert(0, "{Path(__file__).resolve().parent.parent / 'src'}")
        from vserve import lock
        lock.LOCK_DIR = __import__("pathlib").Path("{tmp_path}")
        lk = lock.VserveLock("stuck", "holder")
        lk.acquire()
        sys.stdout.write("locked\\n")
        sys.stdout.flush()
        time.sleep(10)
    """)
    proc = subprocess.Popen(
        [sys.executable, "-c", holder_script],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        proc.stdout.readline()
        assert wait_for_release("stuck", timeout=0.5) is False
    finally:
        proc.terminate()
        proc.wait(timeout=5)


# ── check_session with backend-agnostic check ──────


def test_check_session_clears_stale_when_no_backend_running(tmp_path, monkeypatch):
    """Stale session markers are ignored when no backend is running."""
    monkeypatch.setattr("vserve.lock.LOCK_DIR", tmp_path)
    monkeypatch.setattr("vserve.lock.SESSION_PATH", tmp_path / "vserve-session.json")
    monkeypatch.setenv("USER", "other_user")

    write_session("old-model")
    assert read_session() is not None

    monkeypatch.setenv("USER", "current_user")
    monkeypatch.setattr("vserve.backends.probe_running_backend", lambda: (None, False))

    check_session()  # should ignore stale marker
    assert read_session() is None


def test_check_session_raises_when_backend_running(tmp_path, monkeypatch):
    """Active session with running backend raises SessionHeld."""
    monkeypatch.setattr("vserve.lock.LOCK_DIR", tmp_path)
    monkeypatch.setattr("vserve.lock.SESSION_PATH", tmp_path / "vserve-session.json")
    monkeypatch.setenv("USER", "alice")

    write_session("running-model")

    monkeypatch.setenv("USER", "bob")
    monkeypatch.setattr("vserve.backends.probe_running_backend", lambda: (object(), False))

    with pytest.raises(SessionHeld):
        check_session()


def test_check_session_same_user_allowed(tmp_path, monkeypatch):
    """Same user can restart their own model."""
    monkeypatch.setattr("vserve.lock.LOCK_DIR", tmp_path)
    monkeypatch.setattr("vserve.lock.SESSION_PATH", tmp_path / "vserve-session.json")
    monkeypatch.setenv("USER", "alice")

    write_session("my-model")
    monkeypatch.setattr("vserve.backends.probe_running_backend", lambda: (object(), False))

    check_session()  # should NOT raise — same user


def test_check_session_probe_failure_degrades_safely(tmp_path, monkeypatch):
    """Probe failures should not block advisory session checks."""
    monkeypatch.setattr("vserve.lock.LOCK_DIR", tmp_path)
    monkeypatch.setattr("vserve.lock.SESSION_PATH", tmp_path / "vserve-session.json")
    monkeypatch.setenv("USER", "alice")

    write_session("running-model")

    monkeypatch.setenv("USER", "bob")
    monkeypatch.setattr("vserve.backends.probe_running_backend", lambda: (None, True))

    check_session()


def test_check_session_probe_failure_strict_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("vserve.lock.LOCK_DIR", tmp_path)
    monkeypatch.setattr("vserve.lock.SESSION_PATH", tmp_path / "vserve-session.json")
    monkeypatch.setenv("USER", "alice")

    write_session("running-model")

    monkeypatch.setenv("USER", "bob")
    monkeypatch.setattr("vserve.backends.probe_running_backend", lambda: (None, True))

    with pytest.raises(SessionHeld):
        check_session(fail_on_probe_uncertainty=True)


def test_check_session_probe_failure_strict_allows_same_user_marker(tmp_path, monkeypatch):
    monkeypatch.setattr("vserve.lock.LOCK_DIR", tmp_path)
    monkeypatch.setattr("vserve.lock.SESSION_PATH", tmp_path / "vserve-session.json")
    monkeypatch.setenv("USER", "alice")

    write_session("running-model")
    monkeypatch.setattr("vserve.backends.probe_running_backend", lambda: (None, True))

    check_session(fail_on_probe_uncertainty=True)


def test_check_session_probe_failure_strict_without_marker_allows(tmp_path, monkeypatch):
    monkeypatch.setattr("vserve.lock.LOCK_DIR", tmp_path)
    monkeypatch.setattr("vserve.lock.SESSION_PATH", tmp_path / "vserve-session.json")
    monkeypatch.setenv("USER", "alice")
    monkeypatch.setattr("vserve.backends.probe_running_backend", lambda: (None, True))

    check_session(fail_on_probe_uncertainty=True)


def test_check_session_unknown_owner_when_backend_running_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("vserve.lock.LOCK_DIR", tmp_path)
    monkeypatch.setattr("vserve.lock.SESSION_PATH", tmp_path / "vserve-session.json")
    monkeypatch.setenv("USER", "bob")
    monkeypatch.setattr("vserve.backends.probe_running_backend", lambda: (object(), False))

    with pytest.raises(SessionUnknown):
        check_session()


def test_read_session_prefers_newest_marker(tmp_path, monkeypatch):
    monkeypatch.setattr("vserve.lock.LOCK_DIR", tmp_path)
    monkeypatch.setattr("vserve.lock.SESSION_PATH", tmp_path / "vserve-session.json")

    monkeypatch.setenv("USER", "alice")
    write_session("older-model")

    monkeypatch.setenv("USER", "bob")
    write_session("newer-model")

    info = read_session()
    assert info is not None
    assert info.user == "bob"
    assert info.model == "newer-model"
