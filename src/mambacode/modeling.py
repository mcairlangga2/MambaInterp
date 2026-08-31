"""Model loading and reproducibility helpers."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


DEFAULT_MODEL = "tiiuae/falcon-mamba-7b"
SUPPORTED_TRANSFORMERS_VERSION = "4.49.0"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"{device} was requested, but CUDA is unavailable. "
            "Use --device cpu for a small smoke test."
        )
    return resolved


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL,
    device: str = "cuda:0",
    cache_dir: Path | None = None,
    quantize: bool = False,
):
    """Load a causal LM and tokenizer on one device.

    Intervention experiments are defined for ``tiiuae/falcon-mamba-7b``.
    Recall-only evaluation may use another Hugging Face causal LM.
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    resolved_device = validate_device(device)
    common: dict[str, object] = {}
    if cache_dir is not None:
        common["cache_dir"] = str(cache_dir.expanduser().resolve())

    model_kwargs = dict(common)
    if quantize:
        if resolved_device.type != "cuda":
            raise ValueError("4-bit quantization requires CUDA.")
        model_kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            device_map={"": device},
        )
    else:
        model_kwargs["torch_dtype"] = (
            torch.bfloat16 if resolved_device.type == "cuda" else torch.float32
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if not quantize:
        model.to(resolved_device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, **common)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer
