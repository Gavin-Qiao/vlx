"""Tests for the version check / update notification feature."""

import json
import time
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from vserve.cli import app
from vserve import version as V

runner = CliRunner()


# ── Cache helpers ──────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Redirect cache to a temp directory for every test."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(V, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(V, "CACHE_FILE", cache_dir / "update-check.json")


def _write_cache(latest="0.2.0", current="0.1.0", age=0):
    """Write a cache file with the given values. age=seconds old."""
    V.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "current": current,
        "latest": latest,
        "checked_at": int(time.time()) - age,
    }
    V.CACHE_FILE.write_text(json.dumps(payload))


# ── _compare_versions ──────────────────────────────────


class TestCompareVersions:
    def test_greater(self):
        assert V._compare_versions("0.2.0", "0.1.0") > 0

    def test_equal(self):
        assert V._compare_versions("0.1.0", "0.1.0") == 0

    def test_less(self):
        assert V._compare_versions("0.1.0", "0.2.0") < 0

    def test_patch_bump(self):
        assert V._compare_versions("0.1.1", "0.1.0") > 0

    def test_major_bump(self):
        assert V._compare_versions("1.0.0", "0.9.9") > 0

    def test_bad_input(self):
        assert V._compare_versions("abc", "0.1.0") == 0

    def test_none_input(self):
        assert V._compare_versions(None, "0.1.0") == 0

    def test_final_release_beats_prerelease(self):
        assert V._compare_versions("0.5.2", "0.5.2a3") > 0

    def test_release_candidate_beats_beta(self):
        assert V._compare_versions("0.5.2rc1", "0.5.2b1") > 0


# ── read_cache / write_cache ───────────────────────────


class TestCache:
    def test_read_missing(self):
        assert V.read_cache() is None

    def test_roundtrip(self):
        V.write_cache("0.3.0")
        cache = V.read_cache()
        assert cache["latest"] == "0.3.0"
        assert cache["current"] == V.__version__
        assert "checked_at" in cache

    def test_read_corrupt(self):
        V.CACHE_FILE.write_text("not json{{{")
        assert V.read_cache() is None

    def test_write_creates_dir(self):
        V.CACHE_DIR.rmdir()
        V.write_cache("1.0.0")
        assert V.CACHE_FILE.exists()

    def test_write_records_update_available_flag(self, monkeypatch):
        monkeypatch.setattr(V, "__version__", "0.5.2b1")
        V.write_cache("0.5.2")
        cache = V.read_cache()
        assert cache["update_available"] is True


# ── _cache_is_stale ───────────────────────────────────


class TestCacheStale:
    def test_none_is_stale(self):
        assert V._cache_is_stale(None)

    def test_fresh(self):
        _write_cache(age=0)
        assert not V._cache_is_stale(V.read_cache())

    def test_old(self):
        _write_cache(age=V.STALE_SECONDS + 1)
        assert V._cache_is_stale(V.read_cache())

    def test_missing_key(self):
        assert V._cache_is_stale({"latest": "0.2.0"})


# ── check_pypi ────────────────────────────────────────


class TestCheckPypi:
    def test_success(self):
        fake_resp = MagicMock()
        fake_resp.read.return_value = json.dumps(
            {"info": {"version": "9.9.9"}}
        ).encode()
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = MagicMock(return_value=False)

        with patch("vserve.version.urllib.request.urlopen", return_value=fake_resp):
            assert V.check_pypi() == "9.9.9"

    def test_timeout(self):
        with patch("vserve.version.urllib.request.urlopen", side_effect=TimeoutError):
            assert V.check_pypi() is None

    def test_http_error(self):
        with patch("vserve.version.urllib.request.urlopen", side_effect=OSError("fail")):
            assert V.check_pypi() is None


# ── update_available ──────────────────────────────────


class TestUpdateAvailable:
    def test_no_cache(self):
        assert V.update_available() is None

    def test_newer(self):
        _write_cache(latest="0.2.0", current="0.1.0")
        with patch.object(V, "__version__", "0.1.0"):
            info = V.update_available()
            assert info is not None
            assert info.current == "0.1.0"
            assert info.latest == "0.2.0"

    def test_same(self):
        _write_cache(latest="0.1.0", current="0.1.0")
        with patch.object(V, "__version__", "0.1.0"):
            assert V.update_available() is None

    def test_older_latest(self):
        _write_cache(latest="0.0.1", current="0.1.0")
        with patch.object(V, "__version__", "0.1.0"):
            assert V.update_available() is None


# ── background_refresh ────────────────────────────────


class TestBackgroundRefresh:
    def test_spawns_thread_when_stale(self):
        with patch("vserve.version.check_pypi", return_value="0.5.0") as mock_pypi:
            V.background_refresh()
            # Give the daemon thread time to finish
            import threading
            for t in threading.enumerate():
                if t.daemon and t.is_alive():
                    t.join(timeout=3)
            mock_pypi.assert_called_once()
            cache = V.read_cache()
            assert cache["latest"] == "0.5.0"

    def test_skips_when_fresh(self):
        _write_cache(latest="0.2.0", age=0)
        with patch("vserve.version.check_pypi") as mock_pypi:
            V.background_refresh()
            mock_pypi.assert_not_called()

    def test_failure_clears_stale_latest(self):
        """PyPI failure writes current version, clearing any bogus latest."""
        _write_cache(latest="9.9.9", current=V.__version__, age=V.STALE_SECONDS + 1)
        with patch("vserve.version.check_pypi", return_value=None):
            V.background_refresh()
            import threading
            for t in threading.enumerate():
                if t.daemon and t.is_alive():
                    t.join(timeout=3)
            cache = V.read_cache()
            # Should have written current version, not kept 9.9.9
            assert cache["latest"] == V.__version__

    def test_failure_no_update_notice(self):
        """After PyPI failure, update_available returns None."""
        _write_cache(latest="9.9.9", current=V.__version__, age=V.STALE_SECONDS + 1)
        with patch("vserve.version.check_pypi", return_value=None):
            V.background_refresh()
            import threading
            for t in threading.enumerate():
                if t.daemon and t.is_alive():
                    t.join(timeout=3)
            assert V.update_available() is None


# ── CLI: vserve version ──────────────────────────────


class TestVersionCommand:
    def test_shows_current(self, monkeypatch):
        monkeypatch.setattr(V, "background_refresh", lambda: None)
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert V.__version__ in result.output

    def test_shows_update(self, monkeypatch):
        monkeypatch.setattr(V, "background_refresh", lambda: None)
        _write_cache(latest="9.9.9", current=V.__version__)
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "9.9.9" in result.output
        assert "Update available" in result.output

    def test_no_update(self, monkeypatch):
        monkeypatch.setattr(V, "background_refresh", lambda: None)
        _write_cache(latest=V.__version__, current=V.__version__)
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "Update available" not in result.output


# ── CLI: vserve update ────────────────────────────────


class TestUpdateCommand:
    def test_already_up_to_date(self, monkeypatch):
        monkeypatch.setattr(V, "background_refresh", lambda: None)
        with patch("vserve.version.check_pypi", return_value=V.__version__):
            result = runner.invoke(app, ["update"])
            assert result.exit_code == 0
            assert "up to date" in result.output

    def test_stable_release_not_treated_as_up_to_date(self, monkeypatch):
        monkeypatch.setattr(V, "background_refresh", lambda: None)
        with patch("vserve.version.check_pypi", return_value="0.5.2"), \
             patch("shutil.which", return_value=None):
            result = runner.invoke(app, ["update"])
            assert result.exit_code == 1
            assert "Already up to date" not in result.output
            assert "New version available: 0.5.2" in result.output

    def test_upgrade_via_uv(self, monkeypatch):
        monkeypatch.setattr(V, "background_refresh", lambda: None)
        with patch("vserve.version.check_pypi", return_value="9.9.9"), \
             patch("shutil.which", return_value="/usr/bin/uv"), \
             patch("subprocess.run") as mock_run, \
             patch("vserve.cli._refresh_banner") as mock_refresh:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="vserve v0.1.0\n", stderr=""),
                MagicMock(returncode=0, stdout="", stderr=""),
            ]
            result = runner.invoke(app, ["update"])
            assert result.exit_code == 0
            calls = [str(c) for c in mock_run.call_args_list]
            assert any("upgrade" in c for c in calls)
            mock_refresh.assert_called_once()

    def test_nightly_upgrade_via_uv_allows_prerelease(self, monkeypatch):
        monkeypatch.setattr(V, "background_refresh", lambda: None)
        with patch("shutil.which", return_value="/usr/bin/uv"), \
             patch("subprocess.run") as mock_run, \
             patch("vserve.cli._refresh_banner") as mock_refresh:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="vserve v0.1.0\n", stderr=""),
                MagicMock(returncode=0, stdout="", stderr=""),
            ]
            result = runner.invoke(app, ["update", "--nightly"])
            assert result.exit_code == 0
            calls = [c.args[0] for c in mock_run.call_args_list]
            assert any("--prerelease" in call and "allow" in call for call in calls)
            mock_refresh.assert_called_once()

    def test_nightly_upgrade_via_pip_uses_pre(self, monkeypatch):
        monkeypatch.setattr(V, "background_refresh", lambda: None)

        def _which(name: str) -> str | None:
            if name == "uv":
                return None
            if name in {"pip", "pip3"}:
                return "/usr/bin/pip"
            return None

        with patch("shutil.which", side_effect=_which), \
             patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="", stderr="")) as mock_run, \
             patch("vserve.cli._refresh_banner") as mock_refresh:
            result = runner.invoke(app, ["update", "--nightly"])
            assert result.exit_code == 0
            assert "--pre" in mock_run.call_args.args[0]
            mock_refresh.assert_called_once()

    def test_uv_upgrade_failure_exits_nonzero(self, monkeypatch):
        monkeypatch.setattr(V, "background_refresh", lambda: None)
        with patch("vserve.version.check_pypi", return_value="9.9.9"), \
             patch("shutil.which", return_value="/usr/bin/uv"), \
             patch("subprocess.run") as mock_run, \
             patch("vserve.cli._refresh_banner") as mock_refresh:
            mock_run.side_effect = [
                MagicMock(returncode=0, stdout="vserve v0.1.0\n", stderr=""),
                MagicMock(returncode=1, stdout="", stderr="boom"),
            ]
            result = runner.invoke(app, ["update"])
            assert result.exit_code == 1
            assert "uv upgrade failed" in result.output
            mock_refresh.assert_not_called()

    def test_uv_tool_list_failure_exits_nonzero(self, monkeypatch):
        monkeypatch.setattr(V, "background_refresh", lambda: None)
        with patch("vserve.version.check_pypi", return_value="9.9.9"), \
             patch("shutil.which", return_value="/usr/bin/uv"), \
             patch("subprocess.run", return_value=MagicMock(returncode=1, stdout="", stderr="broken uv")), \
             patch("vserve.cli._refresh_banner") as mock_refresh:
            result = runner.invoke(app, ["update"])
            assert result.exit_code == 1
            assert "uv inspection failed" in result.output
            mock_refresh.assert_not_called()

    def test_pip_upgrade_failure_exits_nonzero(self, monkeypatch):
        monkeypatch.setattr(V, "background_refresh", lambda: None)

        def _which(name: str) -> str | None:
            if name == "uv":
                return None
            if name in {"pip", "pip3"}:
                return "/usr/bin/pip"
            return None

        with patch("vserve.version.check_pypi", return_value="9.9.9"), \
             patch("shutil.which", side_effect=_which), \
             patch("subprocess.run", return_value=MagicMock(returncode=1, stdout="", stderr="fail")), \
             patch("vserve.cli._refresh_banner") as mock_refresh:
            result = runner.invoke(app, ["update"])
            assert result.exit_code == 1
            assert "pip install failed" in result.output
            mock_refresh.assert_not_called()

    def test_no_package_manager(self, monkeypatch):
        monkeypatch.setattr(V, "background_refresh", lambda: None)
        with patch("vserve.version.check_pypi", return_value="9.9.9"), \
             patch("shutil.which", return_value=None):
            result = runner.invoke(app, ["update"])
            assert result.exit_code == 1
            assert "Could not find" in result.output
