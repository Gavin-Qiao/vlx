"""vLLM backend — wraps existing serve/probe/tools modules."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from vserve.gpu import GpuInfo
    from vserve.models import ModelInfo




class VllmBackend:
    name = "vllm"
    display_name = "vLLM"

    @property
    def service_name(self) -> str:
        from vserve.config import cfg

        return cfg().service_name

    @property
    def service_user(self) -> str:
        from vserve.config import cfg

        return cfg().service_user

    @property
    def root_dir(self) -> Path:
        from vserve.config import cfg
        return cfg().vllm_root

    def can_serve(self, model: ModelInfo) -> bool:
        """True if model has top-level safetensors or .bin weights."""
        p = model.path
        has_safetensors = any(p.glob("*.safetensors"))
        has_bin = any(p.glob("*.bin"))
        return has_safetensors or has_bin

    def find_entrypoint(self) -> Path | None:
        from vserve.config import cfg
        vllm_bin = cfg().vllm_bin
        return vllm_bin if vllm_bin.exists() else None

    def runtime_info(self):
        from vserve.runtime import collect_vllm_runtime_info

        return collect_vllm_runtime_info()

    def compatibility(self):
        from vserve.runtime import check_vllm_compatibility

        return check_vllm_compatibility(self.runtime_info())

    def tune(self, model: ModelInfo, gpu: GpuInfo, *, gpu_mem_util: float = 0.90) -> dict:
        from vserve.probe import calculate_limits
        result = calculate_limits(
            model_info=model,
            vram_total_gb=gpu.vram_total_gb,
            gpu_mem_util=gpu_mem_util,
        )
        result["backend"] = self.name
        return result

    def available_tool_parsers(self) -> set[str] | None:
        """Return tool parsers supported by the installed vLLM runtime."""
        try:
            from vllm.entrypoints.openai.tool_parsers import ToolParserManager  # type: ignore[import-not-found, import-untyped]
            parsers = getattr(ToolParserManager, "tool_parsers", None)
            if isinstance(parsers, dict) and parsers:
                return set(parsers)
        except Exception:
            pass
        return None

    def available_reasoning_parsers(self) -> set[str] | None:
        """Return reasoning parsers supported by the installed vLLM runtime."""
        try:
            from vllm.reasoning import ReasoningParserManager  # type: ignore[import-not-found, import-untyped]
            parsers = getattr(ReasoningParserManager, "reasoning_parsers", None)
            if isinstance(parsers, dict) and parsers:
                return set(parsers)
        except Exception:
            pass
        try:
            from vllm.entrypoints.openai.reasoning_parsers import ReasoningParserManager  # type: ignore[import-not-found, import-untyped]
            parsers = getattr(ReasoningParserManager, "reasoning_parsers", None)
            if isinstance(parsers, dict) and parsers:
                return set(parsers)
        except Exception:
            pass
        return None

    @staticmethod
    def _format_available(values: set[str] | None) -> str:
        return ", ".join(sorted(values)) if values else "(none reported)"

    def _validate_parsers(self, choices: dict) -> None:
        tool_parser = choices.get("tool_parser")
        if tool_parser:
            available = self.available_tool_parsers()
            if available is None:
                raise RuntimeError(
                    "Could not inspect installed vLLM tool parsers. "
                    "Activate the configured vLLM runtime or pass a parser supported by the installed runtime."
                )
            if tool_parser not in available:
                raise ValueError(
                    f"Unknown vLLM tool parser '{tool_parser}'. "
                    f"Available: {self._format_available(available)}"
                )
        reasoning_parser = choices.get("reasoning_parser")
        if reasoning_parser:
            available = self.available_reasoning_parsers()
            if available is None:
                raise RuntimeError(
                    "Could not inspect installed vLLM reasoning parsers. "
                    "Activate the configured vLLM runtime or pass a parser supported by the installed runtime."
                )
            if reasoning_parser not in available:
                raise ValueError(
                    f"Unknown vLLM reasoning parser '{reasoning_parser}'. "
                    f"Available: {self._format_available(available)}"
                )

    def build_config(self, model: ModelInfo, choices: dict) -> dict:
        """Build vLLM serving config dict.

        Expected choices keys:
            context, kv_dtype, slots, batched_tokens, gpu_mem_util,
            port, tools, tool_parser, reasoning_parser
        """
        self._validate_parsers(choices)
        cfg: dict = {
            "model": str(model.path),
            "host": "0.0.0.0",
            "port": choices.get("port", 8888),
            "dtype": "bfloat16",
            "gpu-memory-utilization": choices["gpu_mem_util"],
            "max-model-len": choices["context"],
            "max-num-seqs": choices["slots"],
            "kv-cache-dtype": choices["kv_dtype"],
            "enable-prefix-caching": True,
        }
        if choices.get("trust_remote_code"):
            cfg["trust-remote-code"] = True
        bt = choices.get("batched_tokens")
        if bt is not None:
            cfg["max-num-batched-tokens"] = bt

        qf = self.quant_flag(model.quant_method)
        if qf:
            cfg["quantization"] = qf.split()[-1]

        if choices.get("tools") and choices.get("tool_parser"):
            cfg["enable-auto-tool-choice"] = True
            cfg["tool-call-parser"] = choices["tool_parser"]
        rp = choices.get("reasoning_parser")
        if rp:
            cfg["reasoning-parser"] = rp

        return cfg

    def quant_flag(self, method: str | None) -> str:
        from vserve.models import quant_flag as _qf
        return _qf(method)

    def start(self, config_path: Path) -> None:
        from vserve.serve import start_vllm
        start_vllm(config_path)

    def stop(self) -> None:
        from vserve.serve import stop_vllm
        stop_vllm()

    def is_running(self) -> bool:
        from vserve.serve import is_vllm_running
        return is_vllm_running()

    def health_url(self, port: int) -> str:
        return f"http://localhost:{port}/health"

    def active_manifest_path(self) -> Path:
        from vserve.config import active_manifest_path

        return active_manifest_path()

    def detect_tools(self, model_path: Path) -> dict:
        from vserve.tools import detect_tool_parser, detect_reasoning_parser
        return {
            "tool_call_parser": detect_tool_parser(model_path),
            "reasoning_parser": detect_reasoning_parser(model_path),
        }

    def doctor_checks(self) -> list[tuple[str, Callable[[], bool]]]:
        from vserve.config import find_systemd_unit_path

        def check_binary() -> bool:
            return self.find_entrypoint() is not None

        def check_service() -> bool:
            return find_systemd_unit_path(self.service_name) is not None

        return [
            (f"{self.display_name} binary", check_binary),
            (f"{self.service_name}.service unit", check_service),
        ]
