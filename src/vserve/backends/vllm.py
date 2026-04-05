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
        """True if model has safetensors or .bin weights (not GGUF-only)."""
        p = model.path
        has_safetensors = any(p.glob("*.safetensors"))
        has_bin = any(p.glob("*.bin"))
        has_gguf = any(p.glob("*.gguf"))
        return (has_safetensors or has_bin) or ((p / "config.json").exists() and not has_gguf)

    def find_entrypoint(self) -> Path | None:
        from vserve.config import cfg
        vllm_bin = cfg().vllm_bin
        return vllm_bin if vllm_bin.exists() else None

    def tune(self, model: ModelInfo, gpu: GpuInfo, *, gpu_mem_util: float = 0.90) -> dict:
        from vserve.probe import calculate_limits
        return calculate_limits(
            model_info=model,
            vram_total_gb=gpu.vram_total_gb,
            gpu_mem_util=gpu_mem_util,
        )

    def build_config(self, model: ModelInfo, choices: dict) -> dict:
        """Build vLLM serving config dict.

        Expected choices keys:
            context, kv_dtype, slots, batched_tokens, gpu_mem_util,
            port, tools, tool_parser, reasoning_parser
        """
        cfg: dict = {
            "model": str(model.path),
            "host": "0.0.0.0",
            "port": choices.get("port", 8888),
            "dtype": "bfloat16",
            "trust-remote-code": True,
            "gpu-memory-utilization": choices["gpu_mem_util"],
            "max-model-len": choices["context"],
            "max-num-seqs": choices["slots"],
            "kv-cache-dtype": choices["kv_dtype"],
            "enable-prefix-caching": True,
        }
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
