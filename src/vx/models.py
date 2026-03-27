"""Model scanning, detection, and fuzzy matching."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelInfo:
    path: Path
    provider: str
    model_name: str
    architecture: str
    model_type: str
    quant_method: str | None
    max_position_embeddings: int
    is_moe: bool
    model_size_gb: float

    @property
    def full_name(self) -> str:
        return f"{self.provider}/{self.model_name}"


QUANT_FLAGS = {
    "gptq": "--quantization gptq_marlin",
    "awq": "--quantization awq_marlin",
    "fp8": "--quantization fp8",
    "compressed-tensors": "",
    "none": "",
}


def quant_flag(method: str | None) -> str:
    if method is None:
        return ""
    return QUANT_FLAGS.get(method, "")


def detect_model(model_dir: Path) -> ModelInfo:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {model_dir}")

    with open(config_path) as f:
        config = json.load(f)

    text_config = config.get("text_config", {})

    arch = (config.get("architectures") or ["unknown"])[0]
    model_type = config.get("model_type", "unknown")
    qmethod = (config.get("quantization_config") or {}).get("quant_method")
    max_pos = text_config.get(
        "max_position_embeddings",
        config.get("max_position_embeddings", 32768),
    )
    num_experts = text_config.get("num_experts", config.get("num_experts"))
    is_moe = num_experts is not None and num_experts > 0

    size_bytes = sum(f.stat().st_size for f in model_dir.glob("*.safetensors"))
    size_gb = round(size_bytes / (1024**3), 1)

    return ModelInfo(
        path=model_dir,
        provider=model_dir.parent.name,
        model_name=model_dir.name,
        architecture=arch,
        model_type=model_type,
        quant_method=qmethod,
        max_position_embeddings=max_pos,
        is_moe=is_moe,
        model_size_gb=size_gb,
    )


def scan_models(models_root: Path) -> list[ModelInfo]:
    models: list[ModelInfo] = []
    if not models_root.exists():
        return models
    for provider_dir in sorted(models_root.iterdir()):
        if not provider_dir.is_dir():
            continue
        for model_dir in sorted(provider_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if not (model_dir / "config.json").exists():
                continue
            models.append(detect_model(model_dir))
    return models


def fuzzy_match(query: str, models: list[ModelInfo]) -> list[ModelInfo]:
    for m in models:
        if query == str(m.path):
            return [m]

    for m in models:
        if query == m.full_name:
            return [m]

    query_lower = query.lower()
    return [
        m
        for m in models
        if query_lower in m.model_name.lower()
        or query_lower in m.full_name.lower()
    ]
