"""Core data types for Argus v2.

Every scorer produces a `ScoreResult`. Every audit produces a `RiskResult`
composed of per-axis `ScoreResult`s. `Instance` carries probe input + references.
`JudgeVerdict` is one judge's contribution to a `MultiJudgeResult` ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Instance:
    """Input + optional reference data for one scoring call.

    Used by `LiabilityProbe` and consumed by `Scorer` subclasses that need
    reference data (e.g., `KeyFactsScorer` looks at `references`).
    """

    prompt: str
    references: list[str] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class JudgeVerdict:
    """One LLM judge's contribution to a multi-judge ensemble."""

    judge_name: str
    score: float
    rationale: str = ""
    latency_s: float = 0.0
    cost_usd: float = 0.0


@dataclass
class MultiJudgeResult:
    """Aggregated verdicts across multiple judges (Legion mode)."""

    per_judge: dict[str, JudgeVerdict]
    aggregated_score: float
    aggregator: str
    disagreement: float = 0.0
    total_cost_usd: float = 0.0
    total_latency_s: float = 0.0


@dataclass
class ScoreResult:
    """One scorer's verdict on (prompt, response).

    A primitive scorer (regex, classifier, LLM judge) sets the atomic fields:
    `value`, `rationale`, `scorer_name`, `confidence`, etc.

    A composite scorer aggregates several `ScoreResult`s and surfaces them in
    `primary_scorers`; if its fallback fired, it stores the multi-judge bundle
    in `llm_fallback`.

    `tier` is populated upstream by `RiskScorer` after applying the axis's
    `TierMapping` to `value`. Atomic scorers leave it as None.
    """

    value: float
    rationale: str = ""
    scorer_name: str = ""
    scorer_model: str | None = None
    confidence: float = 1.0
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    # Populated only for composite / aggregated results
    tier: int | str | None = None
    primary_scorers: list["ScoreResult"] = field(default_factory=list)
    fallback_fired: bool = False
    llm_fallback: MultiJudgeResult | None = None
    aggregator: str = "single"
    disagreement: float = 0.0
    guardrail_action: str | None = None


@dataclass
class RiskResult:
    """Top-level audit result: N axis scores + overall verdict."""

    scores: dict[str, ScoreResult]
    overall_tier: int | str | None = None
    overall_score: float = 0.0
    latency_s: float = 0.0
    timestamp: str = field(default_factory=_now_iso)


__all__ = [
    "Instance",
    "JudgeVerdict",
    "MultiJudgeResult",
    "ScoreResult",
    "RiskResult",
]
