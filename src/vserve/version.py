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
GITHUB_RELEASES_URL = "https://api.github.com/repos/Gavin-Qiao/vserve/releases"
STALE_SECONDS = 86400  # 24 hours
REQUEST_TIMEOUT = 2  # seconds


class UpdateInfo(NamedTuple):
    current: str
    latest: str


def check_pypi() -> str | None:
    """Fetch the latest version string from PyPI. Returns None on any failure."""
    try:
        data = _fetch_json(PYPI_URL)
    except Exception:
        return None
    latest = _latest_installable_release(data)
    if latest is not None:
        return latest
    if not isinstance(data, dict):
        return None
    info = data.get("info")
    if not isinstance(info, dict):
        return None
    version = info.get("version")
    return version if isinstance(version, str) else None


def _fetch_json(url: str) -> object:
    headers = {
        "Accept": "application/json",
        "User-Agent": f"vserve/{__version__}",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read())


def _latest_installable_release(data: object) -> str | None:
    if not isinstance(data, dict):
        return None
    releases = data.get("releases")
    if not isinstance(releases, dict):
        return None

    github_versions = _github_release_versions()
    candidates: list[tuple[Version, str]] = []
    for version_text, files in releases.items():
        parsed = _parse_public_stable_version(version_text)
        if parsed is None:
            continue
        if _release_is_yanked(files):
            continue
        if github_versions is not None and version_text not in github_versions:
            continue
        candidates.append((parsed, version_text))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _github_release_versions() -> set[str] | None:
    try:
        data = _fetch_json(GITHUB_RELEASES_URL)
    except Exception:
        return None
    if not isinstance(data, list):
        return None

    versions: set[str] = set()
    for release in data:
        if not isinstance(release, dict):
            continue
        if release.get("draft") or release.get("prerelease"):
            continue
        tag = release.get("tag_name")
        if not isinstance(tag, str):
            continue
        version_text = tag[1:] if tag.startswith("v") else tag
        if _parse_public_stable_version(version_text) is not None:
            versions.add(version_text)
    return versions


def _parse_public_stable_version(version_text: object) -> Version | None:
    try:
        parsed = Version(str(version_text))
    except (InvalidVersion, TypeError, ValueError):
        return None
    if parsed.is_prerelease:
        return None
    return parsed


def _release_is_yanked(files: object) -> bool:
    if not isinstance(files, list) or not files:
        return True
    return all(isinstance(file_info, dict) and bool(file_info.get("yanked")) for file_info in files)


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
        "update_available": _compare_versions(latest, __version__) > 0,
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
