# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Zero-shot MNLI scoring with shared model cache and unified batching."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch

from .models import load_nli_model

SUBJECTS = ("product", "mandate", "policy")

_RAW_HYPOTHESES: Dict[str, Dict[str, str]] = {
    "product": {
        "pro": "This text supports electric cars as a product.",
        "anti": "This text opposes electric cars as a product.",
    },
    "mandate": {
        "pro": "This text supports mandates or requirements for electric vehicles.",
        "anti": "This text opposes mandates or requirements for electric vehicles.",
    },
    "policy": {
        "pro": "This text supports EV-related policies such as subsidies or regulations (excluding mandates).",
        "anti": "This text opposes EV-related policies such as subsidies or regulations (excluding mandates).",
    },
}


class NliScore:
    """Container for MNLI probabilities."""

    __slots__ = ("pro", "anti")

    def __init__(self, pro: float, anti: float) -> None:
        self.pro = pro
        self.anti = anti

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"NliScore(pro={self.pro:.4f}, anti={self.anti:.4f})"


class ZeroShotScorer:
    """MNLI-based stance scorer using a shared HF model/tokenizer."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "auto",
        fast_model: bool = False,
        backend: str = "torch",
    ) -> None:
        if backend != "torch":
            raise NotImplementedError("Only torch backend is currently supported. TODO: add ONNX runtime support.")

        model, tokenizer, label_indices = load_nli_model(model_name=model_name, fast_model=fast_model)
        self.model = model
        self.tokenizer = tokenizer
        self.backend = backend
        max_len = getattr(self.tokenizer, "model_max_length", 512) or 512
        if not isinstance(max_len, int) or max_len <= 0 or max_len > 2048:
            max_len = 512
        self.max_length = max_len
        self.entail_idx = label_indices.get("entailment", 2)
        self.contra_idx = label_indices.get("contradiction")
        if self.contra_idx is None:
            self.contra_idx = 0 if self.entail_idx != 0 else 1
        self.device = self._resolve_device(device)
        self._move_model(self.device)

        self._subject_templates: Dict[str, Sequence[str]] = {
            subject: (
                _RAW_HYPOTHESES[subject]["pro"],
                _RAW_HYPOTHESES[subject]["anti"],
            )
            for subject in SUBJECTS
        }
        self._flattened_templates: Sequence[str] = [
            template
            for subject in SUBJECTS
            for template in self._subject_templates[subject]
        ]

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _move_model(self, device: torch.device) -> None:
        if self.model.device != device:
            self.model.to(device)

    def score(self, text: str, subject: str) -> NliScore:
        if not text or subject not in SUBJECTS:
            return NliScore(0.0, 0.0)
        scores = self.score_all([text], batch_size=1)
        return scores[subject][0]

    def score_batch(
        self,
        texts: List[str],
        subject: str,
        batch_size: int = 32,
    ) -> List[NliScore]:
        results = self.score_all(texts, batch_size=batch_size)
        return results.get(subject, [NliScore(0.0, 0.0) for _ in texts])

    def score_all(
        self,
        texts: List[str],
        batch_size: int = 32,
        device: str | None = None,
    ) -> Dict[str, List[NliScore]]:
        if not texts:
            return {subject: [] for subject in SUBJECTS}

        target_device = self._resolve_device(device) if device else self.device
        self._move_model(target_device)

        total_hypotheses = len(self._flattened_templates)
        per_subject = 2

        results: Dict[str, List[NliScore]] = {
            subject: [NliScore(0.0, 0.0) for _ in texts] for subject in SUBJECTS
        }

        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                repeated_texts = []
                hypotheses = []
                for text in batch_texts:
                    repeated_texts.extend([text] * total_hypotheses)
                    hypotheses.extend(self._flattened_templates)

                tokenized = self.tokenizer(
                    repeated_texts,
                    hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokenized = {k: v.to(target_device) for k, v in tokenized.items()}
                logits = self.model(**tokenized).logits

                view_shape = (len(batch_texts), total_hypotheses, logits.shape[-1])
                logits = logits.view(view_shape)
                entail_logits = logits[..., self.entail_idx]
                contra_logits = logits[..., self.contra_idx]
                pair_logits = torch.stack([contra_logits, entail_logits], dim=-1)
                pair_probs = torch.softmax(pair_logits, dim=-1)[..., 1]

                for batch_idx, text_idx in enumerate(range(start, min(start + batch_size, len(texts)))):
                    offset = 0
                    for subject in SUBJECTS:
                        subject_probs = pair_probs[batch_idx, offset : offset + per_subject]
                        results[subject][text_idx] = NliScore(
                            pro=float(subject_probs[0].item()),
                            anti=float(subject_probs[1].item()),
                        )
                        offset += per_subject

        return results
