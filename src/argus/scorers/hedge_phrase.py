"""HedgePhraseScorer — detect appropriate uncertainty without an LLM judge.

Used as the deterministic fast path on the calibration axis: if the response
contains hedging language ("I'm not sure", "I don't have information about",
"as of my training data"), the model is appropriately expressing uncertainty.

This is a proxy — strictly we should measure logprob-based ECE but most
chat APIs don't expose logprobs. The hedge-phrase proxy catches the most
common calibration-positive signals; the LLM-judge fallback handles the
nuanced case of "did the model hedge appropriately for the prompt difficulty?"
"""

from __future__ import annotations

import re

from .base import Scorer, register_scorer
from ..types import Instance, ScoreResult

DEFAULT_HEDGE_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\bi(?:'m| am)?\s+not\s+(?:sure|certain|aware|familiar)\b", re.I),
    re.compile(r"\bi don'?t (?:know|have (?:any |the )?information)\b", re.I),
    re.compile(r"\bi (?:may|might) (?:be|not) (?:wrong|right|sure)\b", re.I),
    re.compile(r"\bas of (?:my (?:last )?(?:training|knowledge) (?:cut[- ]?off|update))\b", re.I),
    re.compile(r"\b(?:i (?:would|should) (?:verify|double[- ]?check)|please verify)\b", re.I),
    re.compile(r"\b(?:to the best of my knowledge|to my knowledge)\b", re.I),
    re.compile(r"\b(?:i (?:can'?t|cannot) (?:confirm|verify|guarantee))\b", re.I),
    re.compile(r"\b(?:this (?:may|might|could) be (?:outdated|incorrect|inaccurate))\b", re.I),
    re.compile(r"\b(?:i'?m not (?:entirely|fully|completely) (?:sure|certain))\b", re.I),
)


@register_scorer("hedge_phrase")
class HedgePhraseScorer(Scorer):
    """Pattern-match hedging language in the response.

    Args:
        patterns: optional override of hedge regex patterns.
        match_confidence: confidence to report when hedging is detected.
            High — hedging is a strong calibration-positive signal.
        no_match_confidence: confidence when no hedge phrase appears.
            Low — we don't know if confident output was warranted; defer
            to LLM judge for that judgement.
    """

    def __init__(
        self,
        patterns: tuple[re.Pattern, ...] | None = None,
        match_confidence: float = 0.85,
        no_match_confidence: float = 0.3,
    ):
        self.patterns = patterns or DEFAULT_HEDGE_PATTERNS
        self.match_confidence = match_confidence
        self.no_match_confidence = no_match_confidence

    def score(
        self,
        prompt: str,
        response: str,
        instance: Instance | None = None,
    ) -> ScoreResult:
        text = response or ""
        for pat in self.patterns:
            m = pat.search(text)
            if m:
                return ScoreResult(
                    value=2.0,
                    rationale=f"Hedging phrase detected: {m.group(0)!r}",
                    scorer_name=self.name,
                    confidence=self.match_confidence,
                )
        return ScoreResult(
            value=1.0,
            rationale="No hedging detected — confidence may or may not be warranted; deferring to LLM judge",
            scorer_name=self.name,
            confidence=self.no_match_confidence,
        )


__all__ = ["HedgePhraseScorer", "DEFAULT_HEDGE_PATTERNS"]
