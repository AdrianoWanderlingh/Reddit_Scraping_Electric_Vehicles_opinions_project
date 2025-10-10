"""Shared model utilities for the EV stance pipeline."""
from __future__ import annotations

import os
from typing import Dict, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_DEFAULT_MODEL = "facebook/bart-large-mnli"
_FAST_MODEL = "prajjwal1/bert-tiny-mnli"

_MODEL_CACHE: Dict[str, Tuple] = {}


def _resolve_model_name(model_name: str | None, fast_model: bool) -> str:
    if model_name:
        return model_name
    env_override = os.getenv("EVREPO_FAST_NLI_MODEL")
    if fast_model and env_override:
        return env_override
    if fast_model:
        return _FAST_MODEL
    env_default = os.getenv("EVREPO_NLI_MODEL")
    return env_default or _DEFAULT_MODEL


def load_nli_model(
    model_name: str | None = None,
    fast_model: bool = False,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, Dict[str, int]]:
    """Return a shared MNLI model/tokenizer pair (loaded lazily, cached globally)."""

    resolved = _resolve_model_name(model_name, fast_model)
    if resolved in _MODEL_CACHE:
        return _MODEL_CACHE[resolved]

    tokenizer = AutoTokenizer.from_pretrained(resolved)
    model = AutoModelForSequenceClassification.from_pretrained(resolved)

    if not torch.cuda.is_available():
        try:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        except Exception:
            pass
    model.eval()

    label2id = getattr(model.config, "label2id", None) or {}
    label_indices = {label.lower(): idx for label, idx in label2id.items()}
    if "entailment" not in label_indices or "contradiction" not in label_indices:
        label_indices = {"contradiction": 0, "neutral": 1, "entailment": 2}

    _MODEL_CACHE[resolved] = (model, tokenizer, label_indices)
    return _MODEL_CACHE[resolved]
