"""TyposTransform — random char-level perturbations for robustness testing.

HELM-style invariance check: a small typo rate should not change the model's
output meaningfully. Used by the robustness axis to detect models that
collapse under minor surface-form perturbations.
"""

from __future__ import annotations

import random

from .base import AttackTransform, register_transform
from ..probes import LiabilityProbe, TransformedProbe


@register_transform("typos")
class TyposTransform(AttackTransform):
    """Apply random typos to ~15% of words in the probe's prompt.

    Args:
        typo_rate: fraction of words to typo. Default 0.15 (HELM's default).
        seed: random seed for reproducibility. Default 42.
    """

    def __init__(self, typo_rate: float = 0.15, seed: int = 42):
        if not 0.0 <= typo_rate <= 1.0:
            raise ValueError(f"typo_rate must be in [0, 1], got {typo_rate}")
        self.typo_rate = typo_rate
        self.seed = seed

    def apply(self, probe: LiabilityProbe) -> TransformedProbe:
        rng = random.Random(self.seed + hash(probe.id) % 10000)
        words = probe.prompt.split()
        if not words:
            return TransformedProbe(probe=probe, transform_name=self.name, transformed_prompt=probe.prompt)
        n_to_typo = max(1, int(len(words) * self.typo_rate))
        indices = rng.sample(range(len(words)), min(n_to_typo, len(words)))
        for i in indices:
            words[i] = self._introduce_typo(words[i], rng)
        return TransformedProbe(
            probe=probe,
            transform_name=self.name,
            transformed_prompt=" ".join(words),
        )

    @staticmethod
    def _introduce_typo(word: str, rng: random.Random) -> str:
        if len(word) < 2:
            return word
        op = rng.choice(("swap", "delete", "duplicate"))
        if op == "swap":
            i = rng.randrange(len(word) - 1)
            return word[:i] + word[i + 1] + word[i] + word[i + 2:]
        if op == "delete":
            i = rng.randrange(len(word))
            return word[:i] + word[i + 1:]
        # duplicate
        i = rng.randrange(len(word))
        return word[:i] + word[i] + word[i:]


__all__ = ["TyposTransform"]
