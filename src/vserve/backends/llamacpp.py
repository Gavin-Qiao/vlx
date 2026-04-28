"""llama.cpp backend — serves GGUF models via llama-server."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from vserve.gpu import GpuInfo
    from vserve.models import ModelInfo


class LlamaCppBackend:
    name = "llamacpp"
    display_name = "llama.cpp"

    @property
    def service_name(self) -> str:
        from vserve.config import cfg

        return cfg().llamacpp_service_name

    @property
    def service_user(self) -> str:
        from vserve.config import cfg

        return cfg().llamacpp_service_user

    @property
    def root_dir(self) -> Path:
        from vserve.config import cfg
        root = cfg().llamacpp_root
        return root if root is not None else Path("/opt/llama-cpp")

    def can_serve(self, model: ModelInfo) -> bool:
        """True if model directory contains GGUF files."""
        return getattr(model, "is_gguf", False) or any(model.path.glob("*.gguf"))

    def find_entrypoint(self) -> Path | None:
        candidate = self.root_dir / "bin" / "llama-server"
        if candidate.exists():
            return candidate
        found = shutil.which("llama-server")
        return Path(found) if found else None

    def runtime_info(self) -> dict:
        return {
            "backend": self.name,
            "executable": str(self.find_entrypoint() or ""),
        }

    def compatibility(self) -> dict:
        return {
            "backend": self.name,
            "supported": True,
            "messages": ["llama.cpp runtime is managed separately."],
            "warnings": [],
            "errors": [],
        }

    def tune(self, model: ModelInfo, gpu: GpuInfo, *, gpu_mem_util: float = 0.90) -> dict:
        """Calculate n-gpu-layers, context sizes, and parallel slots."""
        selected = self._select_gguf_model_files(model.path)
        if selected is None:
            raise ValueError(f"No GGUF files in {model.path}")
        primary_file, model_files = selected
        model_size_bytes = sum(f.stat().st_size for f in model_files)
        model_size_gb = model_size_bytes / (1024**3)

        metadata = self._read_gguf_metadata(primary_file)
        num_layers = metadata.get("num_layers", 32)
        max_context = metadata.get("max_context", 4096)
        num_kv_heads = metadata.get("num_kv_heads", 8)
        head_dim = metadata.get("head_dim", 128)

        usable_gb = gpu.vram_total_gb * gpu_mem_util
        layer_size_gb = model_size_gb / num_layers if num_layers > 0 else model_size_gb

        # Calculate n-gpu-layers (with 10% buffer for embeddings/scratch)
        available_for_layers = usable_gb * 0.9
        n_gpu_layers = min(num_layers, int(available_for_layers / layer_size_gb)) if layer_size_gb > 0 else num_layers
        full_offload = n_gpu_layers >= num_layers

        # KV cache: FP16 by default
        kv_per_token_bytes = 2 * num_kv_heads * head_dim * num_layers * 2

        gpu_model_gb = n_gpu_layers * layer_size_gb
        remaining_gb = usable_gb - gpu_model_gb
        remaining_bytes = max(0, remaining_gb) * (1024**3)

        from vserve.probe import _context_steps
        steps = _context_steps(max_context) if max_context >= 4096 else [4096]
        limits: dict[str, int | None] = {}
        for ctx in steps:
            if kv_per_token_bytes > 0:
                total_kv_tokens = int(remaining_bytes // kv_per_token_bytes)
                parallel = total_kv_tokens // ctx if ctx > 0 else 0
            else:
                parallel = 0
            limits[str(ctx)] = parallel if parallel >= 1 else None

        # Embedding model detection
        is_embedding = model.is_embedding
        pooling = metadata.get("pooling")
        if is_embedding and not pooling:
            pooling = self._guess_pooling(model.model_name)

        tool_info = self.detect_tools(model.path) if not is_embedding else {}

        from datetime import datetime, timezone
        result = {
            "backend": "llamacpp",
            "model_path": str(model.path),
            "calculated_at": datetime.now(timezone.utc).isoformat(),
            "vram_total_gb": gpu.vram_total_gb,
            "gpu_memory_utilization": gpu_mem_util,
            "model_size_gb": round(model_size_gb, 1),
            "n_gpu_layers": n_gpu_layers,
            "num_layers": num_layers,
            "full_offload": full_offload,
            "max_context": max_context,
            "supports_tools": tool_info.get("supports_tools", False),
            "supports_reasoning": tool_info.get("supports_reasoning", False),
            "limits": limits,
        }
        if not metadata:
            result["metadata_estimated"] = True
        if is_embedding:
            result["is_embedding"] = True
            result["pooling"] = pooling or "mean"
        return result

    @staticmethod
    def _select_gguf_model_files(model_path: Path) -> tuple[Path, list[Path]] | None:
        """Select one coherent GGUF variant from a model directory."""
        import re

        gguf_files = sorted(model_path.glob("*.gguf"))
        if not gguf_files:
            return None

        shard_pattern = re.compile(r"(?P<base>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")
        groups: dict[str, list[Path]] = {}
        singles: list[Path] = []
        for path in gguf_files:
            match = shard_pattern.match(path.name)
            if match:
                groups.setdefault(match.group("base"), []).append(path)
            else:
                singles.append(path)

        candidates: list[list[Path]] = [sorted(paths) for paths in groups.values()]
        candidates.extend([[path] for path in singles])
        best = max(candidates, key=lambda files: (sum(path.stat().st_size for path in files), files[0].name))
        return best[0], best

    def _read_gguf_metadata(self, gguf_path: Path) -> dict:
        """Read model metadata from GGUF file header."""
        try:
            from gguf import GGUFReader  # type: ignore[import-not-found, import-untyped]
            reader = GGUFReader(str(gguf_path))

            arch = "llama"
            arch_field = reader.fields.get("general.architecture")
            if arch_field is not None:
                raw = arch_field.parts[-1]
                if isinstance(raw, bytes):
                    arch = raw.decode("utf-8")
                elif hasattr(raw, 'tobytes'):
                    arch = raw.tobytes().decode("utf-8")
                else:
                    arch = str(raw)

            def _get_int(key: str, default: int) -> int:
                field = reader.fields.get(key)
                if field is not None:
                    val = field.parts[-1]
                    # gguf returns numpy arrays; extract scalar
                    return int(val[0]) if hasattr(val, '__len__') and not isinstance(val, (str, bytes)) else int(val)
                return default

            num_layers = _get_int(f"{arch}.block_count", 32)
            max_context = _get_int(f"{arch}.context_length", 4096)
            num_kv_heads = _get_int(f"{arch}.attention.head_count_kv", 8)
            num_attn_heads = _get_int(f"{arch}.attention.head_count", 32)
            embedding_length = _get_int(f"{arch}.embedding_length", 4096)
            head_dim = embedding_length // num_attn_heads if num_attn_heads else 128

            # Detect pooling type (embedding models)
            pooling_field = reader.fields.get("tokenizer.ggml.pooling_type")
            pooling = None
            if pooling_field is not None:
                val = pooling_field.parts[-1]
                pval = int(val[0]) if hasattr(val, '__len__') and not isinstance(val, (str, bytes)) else int(val)
                # llama.cpp: 0=none, 1=mean, 2=cls, 3=last, 4=rank
                pooling = {0: "none", 1: "mean", 2: "cls", 3: "last", 4: "rank"}.get(pval)

            return {
                "arch": arch,
                "num_layers": num_layers,
                "max_context": max_context,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "pooling": pooling,
            }
        except ImportError:
            raise ImportError(
                "The 'gguf' package is required for llama.cpp tuning. "
                "Install it with: pip install 'vserve[llamacpp]'"
            ) from None
        except Exception:
            import sys
            print(f"[vserve] warning: failed to read GGUF metadata from {gguf_path}", file=sys.stderr)
            return {}

    @staticmethod
    def _guess_pooling(model_name: str) -> str:
        """Guess pooling type from model name when GGUF metadata is absent."""
        name = model_name.lower()
        if "bge" in name:
            return "cls"
        if "rerank" in name:
            return "rank"
        # nomic, e5, jina, mxbai, snowflake, qwen-embed all use mean
        return "mean"

    def build_config(self, model: ModelInfo, choices: dict) -> dict:
        """Build llama-server JSON config."""
        selected = self._select_gguf_model_files(model.path)
        model_file = str(selected[0]) if selected is not None else str(model.path)

        cfg: dict = {
            "model": model_file,
            "host": "0.0.0.0",
            "port": choices.get("port", 8888),
            "ctx_size": choices["context"],
            "n_gpu_layers": choices["n_gpu_layers"],
            "parallel": choices.get("parallel", 1),
            "flash_attn": True,
        }
        if choices.get("embedding"):
            cfg["embedding"] = True
            if choices.get("pooling"):
                cfg["pooling"] = choices["pooling"]
        elif choices.get("tools"):
            cfg["jinja"] = True
        return cfg

    def quant_flag(self, method: str | None) -> str:
        """llama.cpp quant is baked into GGUF — no CLI flag needed."""
        return ""

    def start(self, config_path: Path) -> None:
        """Write active launch script from JSON config and start systemd service."""
        import json

        # Read config and build CLI flags
        try:
            cfg = json.loads(config_path.read_text())
        except Exception as exc:
            raise RuntimeError(f"Invalid llama.cpp config {config_path}: {exc}") from None
        if not isinstance(cfg, dict):
            raise RuntimeError(f"Invalid llama.cpp config {config_path}: expected JSON object")
        entrypoint = self.find_entrypoint() or "llama-server"
        args = [str(entrypoint)]
        flag_map = {
            "model": "-m",
            "host": "--host",
            "port": "--port",
            "ctx_size": "-c",
            "n_gpu_layers": "-ngl",
            "parallel": "-np",
        }
        for key, flag in flag_map.items():
            if key in cfg:
                args.extend([flag, str(cfg[key])])
        # Boolean flags
        if cfg.get("flash_attn"):
            args.extend(["-fa", "on"])
        if cfg.get("embedding"):
            args.append("--embedding")
        if cfg.get("pooling"):
            args.extend(["--pooling", str(cfg["pooling"])])
        if cfg.get("jinja"):
            args.append("--jinja")

        # Write per-model launch script + JSON alongside the config,
        # then symlink active.sh/active.json to them.
        # Symlink pattern (matching vLLM's active.yaml) avoids permission
        # issues — any user in the llm group can unlink + recreate symlinks
        # in the group-writable configs dir.
        import shlex
        active = self._active_config_path()
        active.parent.mkdir(parents=True, exist_ok=True)

        # Per-model script in configs/models/
        model_script = config_path.with_suffix(".sh")
        script = "#!/bin/bash\nexec " + shlex.join(args) + "\n"
        model_script.write_text(script)
        model_script.chmod(0o755)

        # Per-model JSON
        model_json = config_path

        # Symlink active.sh → per-model script
        active.unlink(missing_ok=True)
        active.symlink_to(model_script.resolve())

        # Symlink active.json → per-model JSON
        json_link = active.with_suffix(".json")
        json_link.unlink(missing_ok=True)
        json_link.symlink_to(model_json.resolve())

        result = subprocess.run(
            ["sudo", "systemctl", "start", self.service_name],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"systemctl start {self.service_name} failed: {result.stderr}")

    def stop(self) -> None:
        result = subprocess.run(
            ["sudo", "systemctl", "stop", self.service_name],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"systemctl stop {self.service_name} failed: {result.stderr}")

    def is_running(self) -> bool:
        result = subprocess.run(
            ["systemctl", "is-active", self.service_name],
            capture_output=True, text=True, timeout=10,
        )
        status = result.stdout.strip().lower()
        if result.returncode == 0 and status == "active":
            return True
        if status in {"inactive", "failed"}:
            return False
        if status in {"activating", "deactivating", "reloading"}:
            raise RuntimeError(f"systemctl is-active {self.service_name} is transitional: {status}")
        if "could not be found" in result.stderr.lower():
            return False
        if result.stderr.strip():
            raise RuntimeError(f"systemctl is-active {self.service_name} failed: {result.stderr.strip()}")
        return False

    def health_url(self, port: int) -> str:
        return f"http://localhost:{port}/health"

    def active_manifest_path(self) -> Path:
        return self.root_dir / "run" / "active-manifest.json"

    def detect_tools(self, model_path: Path) -> dict:
        """Check if model supports tool calling and reasoning via chat template or GGUF metadata."""
        from vserve.tools import supports_tools, _read_chat_template
        import re

        # Try tokenizer_config.json first
        template = _read_chat_template(model_path)
        if template is None:
            # Fall back to GGUF embedded template
            template = self._read_gguf_chat_template(model_path)

        has_tools = bool(template and re.search(r"\btools\b", template))
        has_reasoning = bool(template and ("<think>" in template or "<|channel>" in template))

        return {
            "supports_tools": has_tools or supports_tools(model_path),
            "supports_reasoning": has_reasoning,
        }

    def _read_gguf_chat_template(self, model_path: Path) -> str | None:
        """Read chat template from GGUF file metadata."""
        gguf_files = sorted(model_path.glob("*.gguf"))
        if not gguf_files:
            return None
        try:
            from gguf import GGUFReader  # type: ignore[import-not-found, import-untyped]
            reader = GGUFReader(str(gguf_files[0]))
            field = reader.fields.get("tokenizer.chat_template")
            if field is None:
                return None
            raw = field.parts[-1]
            if hasattr(raw, "tobytes"):
                return raw.tobytes().decode("utf-8", errors="replace")
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="replace")
            return str(raw)
        except Exception:
            return None

    def doctor_checks(self) -> list[tuple[str, Callable[[], bool]]]:
        from vserve.config import find_systemd_unit_path

        def check_binary() -> bool:
            return self.find_entrypoint() is not None

        def check_service() -> bool:
            return find_systemd_unit_path(self.service_name) is not None

        return [
            (f"{self.display_name} binary (llama-server)", check_binary),
            (f"{self.service_name}.service unit", check_service),
        ]

    def _active_config_path(self) -> Path:
        return self.root_dir / "configs" / "active.sh"
