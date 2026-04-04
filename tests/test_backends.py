"""Tests for the backend registry and protocol."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from vserve.backends import register, get_backend, get_backend_by_name, available_backends
from vserve.backends.protocol import Backend


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry before each test."""
    from vserve.backends import _BACKENDS
    saved = _BACKENDS.copy()
    _BACKENDS.clear()
    yield
    _BACKENDS.clear()
    _BACKENDS.extend(saved)


def _make_mock_backend(name: str, can_serve_result: bool = True, has_binary: bool = True) -> Mock:
    b = Mock(spec=Backend)
    b.name = name
    b.display_name = name
    b.can_serve.return_value = can_serve_result
    b.find_entrypoint.return_value = Path("/fake/bin") if has_binary else None
    return b


def test_register_and_get_backend_by_name():
    b = _make_mock_backend("test")
    register(b)
    assert get_backend_by_name("test") is b


def test_get_backend_by_name_unknown():
    with pytest.raises(KeyError):
        get_backend_by_name("nonexistent")


def test_get_backend_auto_detect():
    b1 = _make_mock_backend("a", can_serve_result=False)
    b2 = _make_mock_backend("b", can_serve_result=True)
    register(b1)
    register(b2)

    model = Mock()
    assert get_backend(model) is b2


def test_get_backend_no_match():
    b = _make_mock_backend("a", can_serve_result=False)
    register(b)

    model = Mock()
    with pytest.raises(ValueError, match="No backend"):
        get_backend(model)


def test_available_backends_filters_missing():
    b1 = _make_mock_backend("installed", has_binary=True)
    b2 = _make_mock_backend("missing", has_binary=False)
    register(b1)
    register(b2)

    result = available_backends()
    assert len(result) == 1
    assert result[0].name == "installed"
