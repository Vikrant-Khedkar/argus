"""Post-flight guards — run after the model produced a response.

`PostFlightRegenGuard` uses a Scorer (typically a CompositeScorer that
includes Llama Guard + LLM judge fallback) to score the response. If any
axis falls below the threshold, it asks the inner provider to regenerate
with a strengthened system prompt that addresses the failure mode.

`PostFlightHardRefuseGuard` is the conservative variant — instead of
regenerating, it replaces the unsafe response with a fixed refusal string.
Used for severity-extreme axes (safety) where regen would be too risky.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..scorers.base import Scorer
from ..types import ScoreResult

REGEN_ADDENDA = {
    "output_liability": (
        "Be extra careful about accuracy. If you are not certain about specific facts, "
        "dates, names, or entities, say so explicitly. Do not fabricate."
    ),
    "discrimination_liability": (
        "Do not assume or generalise based on gender, race, age, occupation, nationality, "
        "religion, or other protected attributes. Treat individuals equitably; if a "
        "question implies a stereotype, push back."
    ),
    "calibration": (
        "If the prompt covers something you don't have reliable information about, "
        "express appropriate uncertainty rather than guessing."
    ),
}

DEFAULT_HARD_REFUSAL = (
    "I can't help with that. The response triggered a safety guardrail."
)


@dataclass
class PostFlightResult:
    """Outcome of a post-flight check."""

    action: str  # "none" | "regenerated" | "hard_refuse"
    final_response: str
    triggered_axis: str | None = None
    triggering_score: float | None = None
    rationale: str = ""


class PostFlightRegenGuard:
    """Score the response; regenerate if any axis is below threshold.

    Args:
        scorer: a Scorer (typically a CompositeScorer + LLM-judge fallback)
            used to score the model's response.
        threshold: any score below this triggers regeneration. Default 1.0.
        max_retries: cap regeneration attempts. Default 1 — preventing
            infinite loops.
        addenda: per-axis system-prompt addenda to inject on regeneration.
    """

    def __init__(
        self,
        scorer: Scorer,
        threshold: float = 1.0,
        max_retries: int = 1,
        addenda: dict[str, str] | None = None,
    ):
        self.scorer = scorer
        self.threshold = threshold
        self.max_retries = max_retries
        self.addenda = addenda or REGEN_ADDENDA

    def check_and_regen(
        self,
        prompt: str,
        response: str,
        regenerate_fn,
    ) -> PostFlightResult:
        """Score the response; if low, ask `regenerate_fn(addendum)` for a new one.

        Args:
            regenerate_fn: callable taking a system-prompt addendum string and
                returning a regenerated response string.
        """
        score: ScoreResult = self.scorer.score(prompt, response)
        if score.value >= self.threshold:
            return PostFlightResult(
                action="none",
                final_response=response,
                triggering_score=score.value,
                rationale=f"Post-flight passed (score={score.value:.2f})",
            )

        # Pick which addendum to inject — by scorer name if it matches an axis key,
        # otherwise generic.
        axis_key = score.scorer_name or "output_liability"
        addendum = self.addenda.get(axis_key, self.addenda.get("output_liability", ""))

        regenerated = regenerate_fn(addendum)
        return PostFlightResult(
            action="regenerated",
            final_response=regenerated,
            triggered_axis=axis_key,
            triggering_score=score.value,
            rationale=f"Regenerated with addendum for {axis_key} (initial score {score.value:.2f})",
        )


class PostFlightHardRefuseGuard:
    """If the scorer says critical-failure, replace response with refusal."""

    def __init__(
        self,
        scorer: Scorer,
        threshold: float = 0.5,
        refusal_text: str = DEFAULT_HARD_REFUSAL,
    ):
        self.scorer = scorer
        self.threshold = threshold
        self.refusal_text = refusal_text

    def check(self, prompt: str, response: str) -> PostFlightResult:
        score: ScoreResult = self.scorer.score(prompt, response)
        if score.value < self.threshold:
            return PostFlightResult(
                action="hard_refuse",
                final_response=self.refusal_text,
                triggering_score=score.value,
                rationale=f"Hard refuse: score {score.value:.2f} below {self.threshold}",
            )
        return PostFlightResult(action="none", final_response=response, triggering_score=score.value)


__all__ = [
    "PostFlightRegenGuard",
    "PostFlightHardRefuseGuard",
    "PostFlightResult",
    "REGEN_ADDENDA",
    "DEFAULT_HARD_REFUSAL",
]
