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
    service_name = "llama-cpp"
    service_user = "llama-cpp"

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

    def tune(self, model: ModelInfo, gpu: GpuInfo, *, gpu_mem_util: float = 0.90) -> dict:
        """Calculate n-gpu-layers, context sizes, and parallel slots."""
        gguf_files = sorted(model.path.glob("*.gguf"))
        if not gguf_files:
            raise ValueError(f"No GGUF files in {model.path}")

        model_size_bytes = sum(f.stat().st_size for f in gguf_files)
        model_size_gb = model_size_bytes / (1024**3)

        metadata = self._read_gguf_metadata(gguf_files[0])
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

        supports_tools = self.detect_tools(model.path).get("supports_tools", False)

        from datetime import datetime, timezone
        return {
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
            "supports_tools": supports_tools,
            "limits": limits,
        }

    def _read_gguf_metadata(self, gguf_path: Path) -> dict:
        """Read model metadata from GGUF file header."""
        try:
            from gguf import GGUFReader  # type: ignore[import-untyped]
            reader = GGUFReader(str(gguf_path))

            arch = "llama"
            arch_field = reader.fields.get("general.architecture")
            if arch_field is not None:
                raw = arch_field.parts[-1]
                arch = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

            def _get_int(key: str, default: int) -> int:
                field = reader.fields.get(key)
                if field is not None:
                    return int(field.parts[-1])
                return default

            num_layers = _get_int(f"{arch}.block_count", 32)
            max_context = _get_int(f"{arch}.context_length", 4096)
            num_kv_heads = _get_int(f"{arch}.attention.head_count_kv", 8)
            num_attn_heads = _get_int(f"{arch}.attention.head_count", 32)
            embedding_length = _get_int(f"{arch}.embedding_length", 4096)
            head_dim = embedding_length // num_attn_heads if num_attn_heads else 128

            return {
                "arch": arch,
                "num_layers": num_layers,
                "max_context": max_context,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
            }
        except ImportError:
            return {}
        except Exception:
            return {}

    def build_config(self, model: ModelInfo, choices: dict) -> dict:
        """Build llama-server JSON config."""
        gguf_files = sorted(model.path.glob("*.gguf"))
        model_file = str(gguf_files[0]) if gguf_files else str(model.path)

        cfg: dict = {
            "model": model_file,
            "host": "0.0.0.0",
            "port": choices.get("port", 8888),
            "ctx-size": choices["context"],
            "n-gpu-layers": choices["n_gpu_layers"],
            "parallel": choices.get("parallel", 1),
            "flash-attn": True,
            "cont-batching": True,
        }
        if choices.get("tools"):
            cfg["jinja"] = True
        return cfg

    def quant_flag(self, method: str | None) -> str:
        """llama.cpp quant is baked into GGUF — no CLI flag needed."""
        return ""

    def start(self, config_path: Path) -> None:
        """Copy config to active location and start systemd service."""
        active = self._active_config_path()
        active.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, active)

        result = subprocess.run(
            ["sudo", "systemctl", "start", self.service_name],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"systemctl start {self.service_name} failed: {result.stderr}")

    def stop(self) -> None:
        result = subprocess.run(
            ["sudo", "systemctl", "stop", self.service_name],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"systemctl stop {self.service_name} failed: {result.stderr}")

    def is_running(self) -> bool:
        result = subprocess.run(
            ["sudo", "systemctl", "is-active", self.service_name],
            capture_output=True, text=True,
        )
        return result.returncode == 0 and result.stdout.strip() == "active"

    def health_url(self, port: int) -> str:
        return f"http://localhost:{port}/health"

    def detect_tools(self, model_path: Path) -> dict:
        """Check if model supports tool calling via chat template."""
        from vserve.tools import supports_tools
        return {"supports_tools": supports_tools(model_path)}

    def doctor_checks(self) -> list[tuple[str, Callable[[], bool]]]:
        def check_binary() -> bool:
            return self.find_entrypoint() is not None

        def check_service() -> bool:
            return Path(f"/etc/systemd/system/{self.service_name}.service").exists()

        return [
            (f"{self.display_name} binary (llama-server)", check_binary),
            (f"{self.service_name}.service unit", check_service),
        ]

    def _active_config_path(self) -> Path:
        return self.root_dir / "configs" / "active.json"
