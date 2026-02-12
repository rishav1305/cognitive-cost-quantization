"""Model loading utilities for FP16, 8-bit, AWQ, and GPTQ quantized models."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Metadata for a loaded model configuration."""

    model_id: str
    quantization: str | None  # None, "8bit", "awq", "gptq"
    bits: int | None  # 4, 8, or None for FP16
    dtype: str  # "float16", "int8", "int4"
    device: str

    @property
    def label(self) -> str:
        """Human-readable label for this config."""
        if self.quantization is None:
            return "fp16"
        if self.quantization == "8bit":
            return "8bit"
        return f"{self.quantization}-{self.bits}bit"


def load_model(
    model_id: str,
    quantization: str | None = None,
    bits: int | None = None,
    device_map: str = "auto",
    trust_remote_code: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, ModelConfig]:
    """Load a model with optional quantization.

    Args:
        model_id: HuggingFace model ID (e.g. "meta-llama/Llama-3.2-3B").
        quantization: None for FP16, "8bit" for bitsandbytes, "awq", or "gptq".
        bits: Number of bits for quantization (4 or 8). Required for awq/gptq.
        device_map: Device mapping strategy (default "auto").
        trust_remote_code: Whether to trust remote code in model configs.

    Returns:
        Tuple of (model, tokenizer, config).
    """
    logger.info("Loading model %s (quantization=%s, bits=%s)", model_id, quantization, bits)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantization is None:
        model, config = _load_fp16(model_id, device_map, trust_remote_code)
    elif quantization == "8bit":
        model, config = _load_8bit(model_id, device_map, trust_remote_code)
    elif quantization == "awq":
        model, config = _load_awq(model_id, device_map, trust_remote_code)
    elif quantization == "gptq":
        model, config = _load_gptq(model_id, device_map, trust_remote_code)
    else:
        raise ValueError(
            f"Unknown quantization: {quantization}. Use None, '8bit', 'awq', 'gptq'."
        )

    logger.info("Model loaded: %s (VRAM: %.2f GB)", config.label, get_vram_usage())
    return model, tokenizer, config


def _load_fp16(
    model_id: str, device_map: str, trust_remote_code: bool
) -> tuple[AutoModelForCausalLM, ModelConfig]:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    config = ModelConfig(
        model_id=model_id,
        quantization=None,
        bits=None,
        dtype="float16",
        device=str(model.device),
    )
    return model, config


def _load_8bit(
    model_id: str, device_map: str, trust_remote_code: bool
) -> tuple[AutoModelForCausalLM, ModelConfig]:
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    config = ModelConfig(
        model_id=model_id,
        quantization="8bit",
        bits=8,
        dtype="int8",
        device=str(model.device),
    )
    return model, config


def _load_awq(
    model_id: str, device_map: str, trust_remote_code: bool
) -> tuple[AutoModelForCausalLM, ModelConfig]:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    config = ModelConfig(
        model_id=model_id,
        quantization="awq",
        bits=4,
        dtype="int4",
        device=str(model.device),
    )
    return model, config


def _load_gptq(
    model_id: str, device_map: str, trust_remote_code: bool
) -> tuple[AutoModelForCausalLM, ModelConfig]:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    config = ModelConfig(
        model_id=model_id,
        quantization="gptq",
        bits=4,
        dtype="int4",
        device=str(model.device),
    )
    return model, config


def get_vram_usage() -> float:
    """Get current GPU memory usage in GB. Returns 0.0 if no GPU."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def reset_vram_tracking() -> None:
    """Reset peak VRAM tracking counter."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
