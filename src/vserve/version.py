"""Version checking with background PyPI refresh and local cache.

Cache lives at ~/.cache/vserve/update-check.json and is refreshed at most
once every 24 hours via a daemon thread — zero latency impact on CLI commands.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

from packaging.version import InvalidVersion, Version

from vserve import __version__

CACHE_DIR = Path.home() / ".cache" / "vserve"
CACHE_FILE = CACHE_DIR / "update-check.json"
PYPI_URL = "https://pypi.org/pypi/vserve/json"
STALE_SECONDS = 86400  # 24 hours
REQUEST_TIMEOUT = 2  # seconds


class UpdateInfo(NamedTuple):
    current: str
    latest: str


def check_pypi() -> str | None:
    """Fetch the latest version string from PyPI. Returns None on any failure."""
    try:
        req = urllib.request.Request(PYPI_URL, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read())
            return data["info"]["version"]
    except Exception:
        return None


def read_cache() -> dict | None:
    """Read the cache file. Returns None if missing or corrupt."""
    try:
        return json.loads(CACHE_FILE.read_text())
    except Exception:
        return None


def write_cache(latest: str) -> None:
    """Write current + latest version and timestamp to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "current": __version__,
        "latest": latest,
        "checked_at": int(time.time()),
    }
    try:
        CACHE_FILE.write_text(json.dumps(payload) + "\n")
    except OSError:
        pass


def _cache_is_stale(cache: dict | None) -> bool:
    if cache is None:
        return True
    try:
        return time.time() - cache["checked_at"] > STALE_SECONDS
    except (KeyError, TypeError):
        return True


def update_available() -> UpdateInfo | None:
    """Check the cache for a newer version. Returns UpdateInfo or None."""
    cache = read_cache()
    if cache is None:
        return None
    try:
        latest = cache["latest"]
    except (KeyError, TypeError):
        return None
    # Only trust cache written by the currently installed version.
    if cache.get("current") != __version__:
        return None
    if _compare_versions(latest, __version__) > 0:
        return UpdateInfo(current=__version__, latest=latest)
    return None


def background_refresh() -> None:
    """If cache is stale, spawn a daemon thread to refresh it."""
    cache = read_cache()
    if not _cache_is_stale(cache):
        return

    def _refresh() -> None:
        latest = check_pypi()
        # Always write — on failure, use current version to clear stale data
        write_cache(latest or __version__)

    t = threading.Thread(target=_refresh, daemon=True)
    t.start()


def _compare_versions(a: str, b: str) -> int:
    """Compare version strings using PEP 440 ordering."""
    try:
        va = Version(str(a))
        vb = Version(str(b))
    except (InvalidVersion, TypeError, ValueError):
        return 0
    return (va > vb) - (va < vb)
