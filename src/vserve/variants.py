"""Variant discovery: group HuggingFace repo files into downloadable model variants."""

from dataclasses import dataclass
import re


_WEIGHT_EXTENSIONS = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
_SKIP_PREFIXES = ("metal/", ".git")


@dataclass
class Variant:
    label: str
    files: dict[str, int]  # path -> size in bytes

    @property
    def num_files(self) -> int:
        return len(self.files)

    @property
    def total_bytes(self) -> int:
        return sum(self.files.values())


def discover_variants(
    files: dict[str, int],
    index_contents: dict[str, dict],
    *,
    config: dict | None = None,
) -> tuple[list[Variant], dict[str, int]]:
    """Group repo files into variants and shared files.

    Args:
        files: Map of repo file path -> size in bytes.
        index_contents: Map of index file path -> parsed JSON content.

    Returns:
        (variants, shared_files) where shared_files is path -> size.
    """
    claimed: set[str] = set()
    variants: list[Variant] = []

    # 1. Index-based variants
    for index_path, index_data in sorted(index_contents.items()):
        weight_map = index_data.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))

        index_dir = index_path.rsplit("/", 1)[0] + "/" if "/" in index_path else ""
        variant_files: dict[str, int] = {}

        if index_path in files:
            variant_files[index_path] = files[index_path]

        missing_shards = False
        for shard in shard_names:
            full_path = index_dir + shard
            if full_path in files:
                variant_files[full_path] = files[full_path]
            else:
                missing_shards = True

        if missing_shards:
            claimed.update(variant_files.keys())
            continue

        if variant_files:
            label = _label_from_index_path(index_path)
            variants.append(Variant(label=label, files=variant_files))
            claimed.update(variant_files.keys())

    # Relabel 'default' variant using quantization_config if available
    if config:
        quant_method = (config.get("quantization_config") or {}).get("quant_method")
        if quant_method:
            for v in variants:
                if v.label == "default":
                    v.label = quant_method

    # 2. Orphan weight files (not claimed by any index, not GGUF)
    for path, size in sorted(files.items()):
        if path in claimed or _is_skipped(path):
            continue
        ext = _ext(path)
        if ext in _WEIGHT_EXTENSIONS and ext != ".gguf":
            variants.append(Variant(label=_label_from_orphan(path), files={path: size}))
            claimed.add(path)

    # 3. GGUF files. Split shards are one runnable variant.
    split_gguf: dict[tuple[str, str, int], dict[int, tuple[str, int]]] = {}
    for path, size in sorted(files.items()):
        if path in claimed or _is_skipped(path):
            continue
        if _ext(path) == ".gguf":
            split = _split_gguf_part(path)
            if split is not None:
                group_key, label, idx, total = split
                split_gguf.setdefault((group_key, label, total), {})[idx] = (path, size)
                claimed.add(path)
                continue
            variants.append(Variant(label=_label_from_gguf(path), files={path: size}))
            claimed.add(path)

    for (_base, label, total), grouped_files in sorted(split_gguf.items()):
        if set(grouped_files) != set(range(1, total + 1)):
            continue
        files_for_variant = {
            path: size
            for _idx, (path, size) in sorted(grouped_files.items())
        }
        variants.append(Variant(label=label, files=files_for_variant))

    # 4. Shared files
    shared: dict[str, int] = {}
    for path, size in sorted(files.items()):
        if path in claimed or _is_skipped(path):
            continue
        shared[path] = size

    return variants, shared


def fetch_repo_variants(repo_id: str, api: object) -> tuple[list[Variant], dict[str, int]]:
    """Fetch file listing from HuggingFace and discover variants."""
    import json
    from huggingface_hub import hf_hub_download

    tree = api.list_repo_tree(repo_id, repo_type="model", recursive=True)  # type: ignore[attr-defined]
    files: dict[str, int] = {}
    for item in tree:
        if not hasattr(item, "size"):
            continue
        files[item.path] = item.size or 0

    # Find and download index files
    index_paths = [
        p for p in files
        if p.endswith(".json") and ("safetensors.index" in p or "bin.index" in p)
    ]
    index_contents: dict[str, dict] = {}
    for idx_path in index_paths:
        local = hf_hub_download(repo_id, idx_path)
        with open(local) as f:
            index_contents[idx_path] = json.load(f)

    # Download config.json for quant method labeling
    config: dict | None = None
    if "config.json" in files:
        local = hf_hub_download(repo_id, "config.json")
        with open(local) as f:
            config = json.load(f)

    return discover_variants(files, index_contents, config=config)


def format_variant_line(variant: Variant, index: int) -> str:
    """Format a variant as a display line for the picker."""
    weight_files = [p for p in variant.files if not p.endswith(".json")]
    n = len(weight_files)
    count_str = f"{n} shard{'s' if n != 1 else ''}" if n > 1 else "1 file"
    size_str = _format_bytes(variant.total_bytes)
    return f"  {index}) {variant.label:<20s} {count_str:<12s} {size_str}"


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    elif n < 1024 ** 3:
        return f"{n / 1024**2:.1f} MB"
    else:
        return f"{n / 1024**3:.1f} GB"


def _ext(path: str) -> str:
    dot = path.rfind(".")
    return path[dot:].lower() if dot != -1 else ""


def _is_skipped(path: str) -> bool:
    return any(path.startswith(p) for p in _SKIP_PREFIXES)


def _label_from_index_path(index_path: str) -> str:
    filename = index_path.rsplit("/", 1)[-1]
    directory = index_path.rsplit("/", 1)[0] if "/" in index_path else ""

    if "safetensors.index." in filename:
        after = filename.split("safetensors.index.")[1]
        tag = after.removesuffix(".json")
        if tag and tag != after:  # has .json suffix removed, and something remains
            return f"{directory}/{tag}" if directory else tag

    if "bin.index." in filename:
        after = filename.split("bin.index.")[1]
        tag = after.removesuffix(".json")
        if tag and tag != after:
            return f"{directory}/{tag}" if directory else tag

    if directory:
        return directory

    return "default"


def _label_from_orphan(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _label_from_gguf(path: str) -> str:
    filename = path.rsplit("/", 1)[-1]
    stem = filename.removesuffix(".gguf")
    split = _split_gguf_group(path)
    if split is not None:
        return split[1]
    for sep in ["-", ".", "_"]:
        parts = stem.rsplit(sep, 1)
        if len(parts) == 2 and parts[1].startswith(("Q", "IQ")):
            return parts[1]
    return stem


def _split_gguf_group(path: str) -> tuple[str, str] | None:
    split = _split_gguf_part(path)
    if split is None:
        return None
    group_key, label, _idx, _total = split
    return group_key, label


def _split_gguf_part(path: str) -> tuple[str, str, int, int] | None:
    filename = path.rsplit("/", 1)[-1]
    match = re.match(r"(?P<base>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$", filename)
    if match is None:
        return None
    base = match.group("base")
    label = _label_from_gguf_stem(base)
    directory = path.rsplit("/", 1)[0] if "/" in path else ""
    group_key = f"{directory}/{base}" if directory else base
    return group_key, label, int(match.group("idx")), int(match.group("total"))


def _label_from_gguf_stem(stem: str) -> str:
    for sep in ["-", ".", "_"]:
        parts = stem.rsplit(sep, 1)
        if len(parts) == 2 and parts[1].startswith(("Q", "IQ")):
            return parts[1]
    return stem
