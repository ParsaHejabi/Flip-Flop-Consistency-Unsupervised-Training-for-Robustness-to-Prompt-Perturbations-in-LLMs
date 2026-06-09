"""Helpers for integrating PEFT LoRA adapters.

This module primarily supported decoder-only models, but we also add a
seq2seq (encoder-decoder) path so LoRA can be applied to models like T5.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    _PEFT_AVAILABLE = True
except ImportError:  # pragma: no cover - soft dependency for seq2seq path
    LoraConfig = None  # type: ignore[assignment]
    TaskType = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]

    class _PlaceholderPeftModel(nn.Module):  # minimal stub for isinstance checks
        pass

    PeftModel = _PlaceholderPeftModel  # type: ignore[assignment]
    _PEFT_AVAILABLE = False


_PRESET_TARGET_MODULES = {
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2_moe": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gpt_neox": ["query_key_value"],
    "gptj": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "falcon": ["query_key_value", "dense", "dense_h"],
}

# For encoder-decoder families (e.g., T5), common linear submodules are named
# as below. PEFT will only attach adapters to modules that actually exist, so
# it's fine if some names aren't present in a specific variant.
_SEQ2SEQ_DEFAULT_TARGETS = {
    "t5": ["q", "k", "v", "o"],
    "mt5": ["q", "k", "v", "o"],
    "byt5": ["q", "k", "v", "o"],
}


def is_peft_model(model: nn.Module) -> bool:
    """Return ``True`` when ``model`` is a PEFT-wrapped module."""

    return _PEFT_AVAILABLE and isinstance(model, PeftModel)


def _environment_override() -> Optional[List[str]]:
    override = os.environ.get("LORA_TARGET_MODULES")
    if not override:
        return None
    return [token.strip() for token in override.split(",") if token.strip()]


def _infer_target_modules(model: nn.Module) -> List[str]:
    override = _environment_override()
    if override:
        return override

    model_type = getattr(getattr(model, "config", None), "model_type", "")
    if model_type in _PRESET_TARGET_MODULES:
        return list(_PRESET_TARGET_MODULES[model_type])

    candidates = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "c_attn", "c_proj", "out_proj"}
    discovered = set()
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in candidates and isinstance(module, nn.Linear):
            discovered.add(leaf)
    if discovered:
        return sorted(discovered)

    raise ValueError(
        "Unable to infer LoRA target modules automatically. Set `LORA_TARGET_MODULES` environment variable " "(comma separated) to list the linear submodules that should receive adapters."
    )


def _discover_target_matches(model: nn.Module, target_modules: List[str]) -> List[str]:
    targets = set(target_modules)
    matches = []
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in targets:
            matches.append(name)
    return matches


def _infer_target_modules_seq2seq(model: nn.Module) -> List[str]:
    override = _environment_override()
    if override:
        return override

    model_type = getattr(getattr(model, "config", None), "model_type", "")
    if model_type in _SEQ2SEQ_DEFAULT_TARGETS:
        return list(_SEQ2SEQ_DEFAULT_TARGETS[model_type])

    # Fallback: discover linear leaves with T5-like names
    candidates = {"q", "k", "v", "o", "wi", "wo"}
    discovered = set()
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in candidates and isinstance(module, nn.Linear):
            discovered.add(leaf)
    if discovered:
        return sorted(discovered)

    # If nothing matched, nudge the user to specify targets explicitly
    raise ValueError("Unable to infer LoRA target modules for seq2seq model. Set `LORA_TARGET_MODULES` environment variable " "(comma separated) to list the linear submodules (e.g., q,v or q,k,v,o).")


def apply_peft_lora_to_decoder_model(
    model: nn.Module,
    test_args: Any,
    training_args: Any,
) -> Tuple[nn.Module, bool]:
    """Attach LoRA adapters to a decoder-only model using PEFT.

    Returns the (possibly wrapped) model and a flag indicating if PEFT-based LoRA was applied.
    """

    if getattr(test_args, "peft_option", "none") != "lora" or getattr(model.config, "is_encoder_decoder", False):
        return model, False

    if not _PEFT_AVAILABLE:
        raise ImportError("peft is required for decoder-only LoRA fine-tuning. Install it with `pip install peft` before running.")

    assert getattr(test_args, "lora_rank") is not None
    assert getattr(test_args, "lora_alpha") is not None
    assert getattr(test_args, "lora_dropout") is not None

    rank = int(getattr(test_args, "lora_rank"))
    alpha = float(getattr(test_args, "lora_alpha"))
    dropout = float(getattr(test_args, "lora_dropout"))
    target_modules = _infer_target_modules(model)
    logger.info("Applying PEFT LoRA with rank=%s, alpha=%s, dropout=%s to modules=%s", rank, alpha, dropout, target_modules)

    matched_module_names = _discover_target_matches(model, target_modules)
    if not matched_module_names:
        raise ValueError("None of the requested LoRA target modules were found in the model. " "Inspect model.named_modules() and adjust LORA_TARGET_MODULES accordingly.")

    logger.info("LoRA target matches (%d): %s", len(matched_module_names), ", ".join(sorted(matched_module_names)))

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )

    wrapped = get_peft_model(model, lora_config)

    if getattr(training_args, "gradient_checkpointing", False) or getattr(training_args, "do_train", False):
        wrapped.config.use_cache = False

    return wrapped, True


def apply_peft_lora_to_seq2seq_model(
    model: nn.Module,
    test_args: Any,
    training_args: Any,
) -> Tuple[nn.Module, bool]:
    """Attach LoRA adapters to an encoder-decoder (seq2seq) model using PEFT.

    Returns the (possibly wrapped) model and a flag indicating if PEFT-based LoRA was applied.
    """

    if getattr(test_args, "peft_option", "none") != "lora" or not getattr(model.config, "is_encoder_decoder", False):
        return model, False

    if not _PEFT_AVAILABLE:
        raise ImportError("peft is required for seq2seq LoRA fine-tuning. Install it with `pip install peft` before running.")

    assert getattr(test_args, "lora_rank") is not None
    assert getattr(test_args, "lora_alpha") is not None
    assert getattr(test_args, "lora_dropout") is not None

    rank = max(1, int(getattr(test_args, "lora_rank")))
    alpha = float(getattr(test_args, "lora_alpha"))
    dropout = float(getattr(test_args, "lora_dropout"))
    target_modules = _infer_target_modules_seq2seq(model)
    logger.info(
        "Applying PEFT LoRA (seq2seq) with rank=%s, alpha=%s, dropout=%s to modules=%s",
        rank,
        alpha,
        dropout,
        target_modules,
    )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=target_modules,
    )

    wrapped = get_peft_model(model, lora_config)

    if getattr(training_args, "gradient_checkpointing", False):
        wrapped.config.use_cache = False

    return wrapped, True


__all__ = [
    "apply_peft_lora_to_decoder_model",
    "apply_peft_lora_to_seq2seq_model",
    "is_peft_model",
]
