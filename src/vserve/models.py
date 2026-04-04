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
    num_kv_heads: int | None = None
    num_layers: int | None = None
    head_dim: int | None = None
    is_gguf: bool = False

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
    gguf_files = list(model_dir.glob("*.gguf"))

    if gguf_files and not config_path.exists():
        # GGUF-only model — no config.json
        size_bytes = sum(f.stat().st_size for f in gguf_files)
        size_gb = round(size_bytes / (1024**3), 1)
        return ModelInfo(
            path=model_dir,
            provider=model_dir.parent.name,
            model_name=model_dir.name,
            architecture="gguf",
            model_type="gguf",
            quant_method=None,
            max_position_embeddings=0,  # read from GGUF metadata at tune time
            is_moe=False,
            model_size_gb=size_gb,
            is_gguf=True,
        )

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
    num_experts = text_config.get(
        "num_experts",
        config.get("num_experts",
                    text_config.get("num_local_experts",
                                    config.get("num_local_experts"))),
    )
    is_moe = num_experts is not None and num_experts > 0

    # Architecture fields for KV cache calculation
    num_kv_heads = text_config.get(
        "num_key_value_heads",
        config.get("num_key_value_heads"),
    )
    num_layers = text_config.get(
        "num_hidden_layers",
        config.get("num_hidden_layers"),
    )
    hidden_size = text_config.get("hidden_size", config.get("hidden_size"))
    num_attn_heads = text_config.get(
        "num_attention_heads",
        config.get("num_attention_heads"),
    )
    head_dim_val = text_config.get("head_dim", config.get("head_dim"))
    if head_dim_val is None and hidden_size and num_attn_heads:
        head_dim_val = hidden_size // num_attn_heads

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
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        head_dim=head_dim_val,
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
            has_config = (model_dir / "config.json").exists()
            has_gguf = any(model_dir.glob("*.gguf"))
            if not has_config and not has_gguf:
                continue
            try:
                models.append(detect_model(model_dir))
            except Exception:
                continue
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
