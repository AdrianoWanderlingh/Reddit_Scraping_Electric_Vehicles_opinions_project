# nli_zeroshot.py
# Copyright (c) 2025 OpenFis
# Licensed under the MIT License (see LICENSE file for details).
"""Zero-shot MNLI scoring with shared model cache, calibration, and template ensembling."""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from .models import load_nli_model

# Subjects scored in the pipeline
SUBJECTS = ("product", "mandate", "policy")

# Base single-sentence variants kept for reference/back-compat (not used directly).
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

# Symmetric, low-valence paraphrases for ensembling (aim to reduce hypothesis-only bias).
# Keep wording parallel across pro/anti. 2–4 per side is usually sufficient for tiny models.
_ENSEMBLE_HYPOTHESES: Dict[str, Dict[str, Sequence[str]]] = {
    "product": {
        "pro": (
            "The author expresses a favorable view of electric cars as a product.",
            "This text supports electric cars as a product.",
            "The writer is positive about EVs as consumer products.",
        ),
        "anti": (
            "The author expresses an unfavorable view of electric cars as a product.",
            "This text opposes electric cars as a product.",
            "The writer is negative about EVs as consumer products.",
        ),
    },
    "mandate": {
        "pro": (
            "The author expresses a favorable view of mandates or requirements for electric vehicles.",
            "This text supports EV mandates or requirements.",
            "The writer is positive about requiring EV adoption.",
        ),
        "anti": (
            "The author expresses an unfavorable view of mandates or requirements for electric vehicles.",
            "This text opposes EV mandates or requirements.",
            "The writer is negative about requiring EV adoption.",
        ),
    },
    "policy": {
        "pro": (
            "The author expresses a favorable view of EV-related policies such as subsidies or regulations, excluding mandates.",
            "This text supports EV-related policies such as subsidies or non-mandate regulations.",
            "The writer is positive about EV subsidies or similar policies (not mandates).",
        ),
        "anti": (
            "The author expresses an unfavorable view of EV-related policies such as subsidies or regulations, excluding mandates.",
            "This text opposes EV-related policies such as subsidies or non-mandate regulations.",
            "The writer is negative about EV subsidies or similar policies (not mandates).",
        ),
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
    """MNLI-based stance scorer using a shared HF model/tokenizer.

    Key changes vs. the earlier version:
      - Uses full 3-way softmax (keeps neutral mass; no [contra, entail] renormalization).
      - Optional contextual calibration (null-prompt bias subtraction).
      - Template ensembling: multiple symmetric paraphrases per (subject, class).
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "auto",
        fast_model: bool = False,
        backend: str = "torch",
        calibrate: bool = True,
        templates: str = "full",   # "full" | "lite" | "none"
    ) -> None:
        if backend != "torch":
            raise NotImplementedError(
                "Only torch backend is currently supported. TODO: add ONNX runtime support."
            )

        model, tokenizer, label_indices = load_nli_model(
            model_name=model_name, fast_model=fast_model
        )
        self.model = model
        self.tokenizer = tokenizer
        self.backend = backend

        # model_max_length can be very large in some tokenizers; cap reasonably
        max_len = getattr(self.tokenizer, "model_max_length", 512) or 512
        if not isinstance(max_len, int) or max_len <= 0 or max_len > 2048:
            max_len = 512
        self.max_length = max_len

        # Label indices from config or sensible default handled by load_nli_model
        self.entail_idx = label_indices.get("entailment", 2)
        self.contra_idx = label_indices.get("contradiction")
        if self.contra_idx is None:
            self.contra_idx = 0 if self.entail_idx != 0 else 1

        # Device placement
        self.device = self._resolve_device(device)

        # Calibration state
        self.calibrate = calibrate
        self.templates = templates  # store mode
        self._bias_logits = None  # torch.Tensor of shape (num_labels,)

        # Build templates and index mapping for ensembling
        self._build_templates()

        # Move model and compute bias logits if needed
        self._move_model(self.device)

    # -------------------------
    # Internal helpers
    # -------------------------

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _move_model(self, device: torch.device) -> None:
        if self.model.device != device:
            self.model.to(device)
        # Keep bias logits on the right device
        self._maybe_refresh_bias(device)

    def _build_templates(self) -> None:
        """Create flattened hypothesis list and per-subject index mapping."""
        # pick how many paraphrases per side based on mode
        k = {"full": 9999, "lite": 2, "none": 1}.get(self.templates, 9999)
        self._subject_templates: Dict[str, Dict[str, Sequence[str]]] = {}
        for s in SUBJECTS:
            pro_all = list(_ENSEMBLE_HYPOTHESES[s]["pro"])
            anti_all = list(_ENSEMBLE_HYPOTHESES[s]["anti"])
            self._subject_templates[s] = {
                "pro": tuple(pro_all[:k]),
                "anti": tuple(anti_all[:k]),
            }  
       

        flattened: List[str] = []
        index_map: Dict[str, Dict[str, List[int]]] = {s: {"pro": [], "anti": []} for s in SUBJECTS}

        for subject in SUBJECTS:
            for hyp in self._subject_templates[subject]["pro"]:
                index_map[subject]["pro"].append(len(flattened))
                flattened.append(hyp)
            for hyp in self._subject_templates[subject]["anti"]:
                index_map[subject]["anti"].append(len(flattened))
                flattened.append(hyp)

        self._flattened_templates: Sequence[str] = tuple(flattened)
        self._index_map = index_map  # subject -> {"pro":[...], "anti":[...]}

    def _maybe_refresh_bias(self, device: torch.device) -> None:
        """Compute average logits for empty-premise inputs to subtract as bias."""
        if not self.calibrate:
            self._bias_logits = None
            return
        if not self._flattened_templates:
            self._bias_logits = None
            return

        null_premises = [""] * len(self._flattened_templates)
        tok = self.tokenizer(
            null_premises,
            list(self._flattened_templates),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        with torch.inference_mode():
            bias = self.model(**tok).logits  # (H, 3)
        self._bias_logits = bias.mean(dim=0).detach()  # (3,)

    # -------------------------
    # Public API
    # -------------------------

    def score(self, text: str, subject: str) -> NliScore:
        """Score a single text for a single subject."""
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
        """Score multiple texts for a single subject."""
        results = self.score_all(texts, batch_size=batch_size)
        return results.get(subject, [NliScore(0.0, 0.0) for _ in texts])

    def score_all(
        self,
        texts: List[str],
        batch_size: int = 32,
        device: str | None = None,
    ) -> Dict[str, List[NliScore]]:
        """Score multiple texts for all subjects.

        Returns a dict: subject -> List[NliScore] aligned with the input texts.
        """
        if not texts:
            return {subject: [] for subject in SUBJECTS}

        target_device = self._resolve_device(device) if device else self.device
        self._move_model(target_device)

        H = len(self._flattened_templates)  # total hypotheses across all subjects
        results: Dict[str, List[NliScore]] = {
            subject: [NliScore(0.0, 0.0) for _ in texts] for subject in SUBJECTS
        }

        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]

                repeated_texts: List[str] = []
                hypotheses: List[str] = []
                for text in batch_texts:
                    repeated_texts.extend([text] * H)
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
                logits = self.model(**tokenized).logits  # (B*H, 3)

                # Contextual calibration: subtract mean null-prompt logits
                if self.calibrate and self._bias_logits is not None:
                    logits = logits - self._bias_logits.to(target_device)

                # Full 3-way softmax; take p(entailment) for each hypothesis
                probs = torch.softmax(logits, dim=-1)  # (B*H, 3)
                entail = probs[..., self.entail_idx]    # (B*H,)

                # Reshape to (B, H)
                B = len(batch_texts)
                entail = entail.view(B, H)

                # Aggregate by subject via mean over paraphrase indices
                for b, text_idx in enumerate(range(start, min(start + batch_size, len(texts)))):
                    for subject in SUBJECTS:
                        pro_idxs = self._index_map[subject]["pro"]
                        anti_idxs = self._index_map[subject]["anti"]
                        pro_p = float(entail[b, pro_idxs].mean().item()) if pro_idxs else 0.0
                        anti_p = float(entail[b, anti_idxs].mean().item()) if anti_idxs else 0.0
                        results[subject][text_idx] = NliScore(pro=pro_p, anti=anti_p)

        return results
