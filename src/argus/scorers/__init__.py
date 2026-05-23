"""Scorers — anything that turns (prompt, response) into a ScoreResult.

Layout:
- `base.py`      — `Scorer` ABC + `register_scorer` decorator + registry helpers
- `composite.py` — `CompositeScorer` — primary chain + LLM fallback

Phase 4 adds: ModelBasedScorer, LLMJudgeScorer, LlamaGuardScorer,
PerspectiveAPIScorer, RefusalRegexScorer, KeyFactsScorer, HedgePhraseScorer.
"""

from .base import Scorer, register_scorer, get_scorer_class, list_scorers
from .composite import CompositeScorer
from .multi_judge import MultiJudgeScorer

__all__ = [
    "Scorer",
    "register_scorer",
    "get_scorer_class",
    "list_scorers",
    "CompositeScorer",
    "MultiJudgeScorer",
]
