"""Tests for vlx.lock — file-based multi-user locking."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from vlx.lock import LockHeld, LockInfo, VlxLock, _format_elapsed, wait_for_release


# ── Helpers ──────────────────────────────────────────

@pytest.fixture(autouse=True)
def _use_tmp_lock_dir(tmp_path, monkeypatch):
    """Redirect LOCK_DIR to a temp dir for all tests."""
    monkeypatch.setattr("vlx.lock.LOCK_DIR", tmp_path)


# ── Basic acquire / release ──────────────────────────

def test_acquire_and_release(tmp_path):
    lock = VlxLock("test", "test op")
    lock.acquire()
    assert lock._fd is not None
    lock_file = tmp_path / "vlx-test.lock"
    assert lock_file.exists()
    # Metadata written
    data = json.loads(lock_file.read_text().strip())
    assert data["pid"] == os.getpid()
    assert data["command"] == "test op"
    lock.release()
    assert lock._fd is None


def test_context_manager(tmp_path):
    with VlxLock("ctx", "ctx test") as lock:
        assert lock._fd is not None
        assert (tmp_path / "vlx-ctx.lock").exists()
    assert lock._fd is None


def test_release_idempotent():
    lock = VlxLock("idem", "test")
    lock.acquire()
    lock.release()
    lock.release()  # should not raise


# ── Contention ───────────────────────────────────────

def test_second_acquire_raises(tmp_path):
    """A subprocess holds the lock; parent's acquire raises LockHeld."""
    holder_script = textwrap.dedent(f"""\
        import sys, os, fcntl, time
        sys.path.insert(0, "{Path(__file__).resolve().parent.parent / 'src'}")
        from vlx import lock
        lock.LOCK_DIR = __import__("pathlib").Path("{tmp_path}")
        lk = lock.VlxLock("contend", "holder process")
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
            VlxLock("contend", "parent attempt").acquire()
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
        from vlx import lock
        lock.LOCK_DIR = __import__("pathlib").Path("{tmp_path}")
        lk = lock.VlxLock("die", "dying process")
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
    lock = VlxLock("die", "reacquire")
    lock.acquire()  # should not raise
    lock.release()


# ── Per-model download lock names ────────────────────

def test_different_model_locks_independent(tmp_path):
    lock_a = VlxLock("download-Qwen--Qwen3-27B", "dl model A")
    lock_b = VlxLock("download-Meta--Llama-8B", "dl model B")
    lock_a.acquire()
    lock_b.acquire()  # should not raise — different lock names
    lock_a.release()
    lock_b.release()


def test_same_model_lock_contends(tmp_path):
    """Same lock name from subprocess → parent blocked."""
    holder_script = textwrap.dedent(f"""\
        import sys, time
        sys.path.insert(0, "{Path(__file__).resolve().parent.parent / 'src'}")
        from vlx import lock
        lock.LOCK_DIR = __import__("pathlib").Path("{tmp_path}")
        lk = lock.VlxLock("download-Qwen--Model", "downloading Qwen/Model")
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
            VlxLock("download-Qwen--Model", "parent dl").acquire()
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
    monkeypatch.setattr("vlx.lock.LOCK_DIR", subdir)
    lock = VlxLock("mkdir", "test")
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
        from vlx import lock
        lock.LOCK_DIR = __import__("pathlib").Path("{tmp_path}")
        lk = lock.VlxLock("handoff", "holder")
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
        from vlx import lock
        lock.LOCK_DIR = __import__("pathlib").Path("{tmp_path}")
        lk = lock.VlxLock("stuck", "holder")
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
