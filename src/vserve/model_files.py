"""Helpers for recognizing model weight files."""

from pathlib import Path


WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf")


def has_suffix(name: str | Path, suffix: str) -> bool:
    return str(name).lower().endswith(suffix.lower())


def is_gguf_name(name: str | Path) -> bool:
    return has_suffix(name, ".gguf")


def is_weight_file_name(name: str | Path) -> bool:
    lowered = str(name).lower()
    return lowered.endswith(WEIGHT_SUFFIXES)


def iter_top_level_files_with_suffix(root: Path, suffix: str) -> list[Path]:
    try:
        return sorted(
            path
            for path in root.iterdir()
            if path.is_file() and has_suffix(path.name, suffix)
        )
    except OSError:
        return []


def iter_recursive_files_with_suffix(root: Path, suffix: str) -> list[Path]:
    if not root.exists():
        return []
    try:
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and has_suffix(path.name, suffix)
        )
    except OSError:
        return []


def iter_recursive_weight_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    try:
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and is_weight_file_name(path.name)
        )
    except OSError:
        return []
