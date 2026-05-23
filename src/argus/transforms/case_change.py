"""CaseChangeTransform — capitalisation perturbation for robustness testing."""

from __future__ import annotations

import random
from typing import Literal

from .base import AttackTransform, register_transform
from ..probes import LiabilityProbe, TransformedProbe

CaseStyle = Literal["upper", "lower", "title", "random"]


@register_transform("case_change")
class CaseChangeTransform(AttackTransform):
    """Apply a case transformation to the entire prompt.

    Args:
        style: ``"upper"`` | ``"lower"`` | ``"title"`` | ``"random"``.
        seed: random seed (only used when style="random").
    """

    def __init__(self, style: CaseStyle = "upper", seed: int = 42):
        if style not in {"upper", "lower", "title", "random"}:
            raise ValueError(f"unknown case style: {style!r}")
        self.style: CaseStyle = style
        self.seed = seed

    def apply(self, probe: LiabilityProbe) -> TransformedProbe:
        text = probe.prompt
        if self.style == "upper":
            transformed = text.upper()
        elif self.style == "lower":
            transformed = text.lower()
        elif self.style == "title":
            transformed = text.title()
        else:
            rng = random.Random(self.seed + hash(probe.id) % 10000)
            transformed = "".join(c.upper() if rng.random() > 0.5 else c.lower() for c in text)
        return TransformedProbe(
            probe=probe,
            transform_name=f"{self.name}:{self.style}",
            transformed_prompt=transformed,
        )


__all__ = ["CaseChangeTransform"]
