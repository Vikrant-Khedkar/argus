"""Tier mapping — derive a presentation-layer tier from a raw 0-2 score.

Pluggable so that the same audit data can be rendered as Tier 1/2/3 (insurance
language), LOW/MEDIUM/HIGH (procurement language), raw scores (engineer
language), or A-F letter grades (consumer language). Raw scores are always
preserved in `ScoreResult.value`; the tier mapping only affects presentation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TierMapping(ABC):
    """Maps a raw score in [0, scale_max] to a presentation tier."""

    @abstractmethod
    def map(self, value: float, scale_max: float = 2.0) -> int | str:
        ...

    def label(self, tier: int | str) -> str:
        """Human-readable label for a tier."""
        return str(tier)


class InsuranceTierMapping(TierMapping):
    """Tier 1 / Tier 2 / Tier 3 — the default for Argus underwriting memos.

    Thresholds:
        avg ≥ 1.5  → Tier 1 (low liability risk, production-ready)
        avg ≥ 1.0  → Tier 2 (borderline, mitigations recommended)
        avg <  1.0 → Tier 3 (high risk, do not ship)
    """

    def map(self, value: float, scale_max: float = 2.0) -> int:
        if value >= 0.75 * scale_max:
            return 1
        if value >= 0.50 * scale_max:
            return 2
        return 3

    def label(self, tier: int) -> str:
        return f"Tier {tier}"


class RiskLevelMapping(TierMapping):
    """LOW / MEDIUM / HIGH — softer language for procurement / non-insurance audiences."""

    def map(self, value: float, scale_max: float = 2.0) -> str:
        if value >= 0.75 * scale_max:
            return "LOW"
        if value >= 0.50 * scale_max:
            return "MEDIUM"
        return "HIGH"


class RawScoreMapping(TierMapping):
    """Pass-through — return the raw float. For technical / debugging contexts."""

    def map(self, value: float, scale_max: float = 2.0) -> str:
        return f"{value:.2f}"


class GradeLetterMapping(TierMapping):
    """A / B / C / D / F — friendly UX for consumer-facing reports."""

    def map(self, value: float, scale_max: float = 2.0) -> str:
        pct = value / scale_max if scale_max else 0.0
        if pct >= 0.90:
            return "A"
        if pct >= 0.80:
            return "B"
        if pct >= 0.70:
            return "C"
        if pct >= 0.60:
            return "D"
        return "F"


__all__ = [
    "TierMapping",
    "InsuranceTierMapping",
    "RiskLevelMapping",
    "RawScoreMapping",
    "GradeLetterMapping",
]
