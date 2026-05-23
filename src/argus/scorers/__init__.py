"""Scorers — anything that turns (prompt, response) into a ScoreResult.

Architecture:
- ``base.py``         — Scorer ABC + @register_scorer + registry
- ``model_based.py``  — ModelBasedScorer (shared base for any model-call scorer)
- ``llm_judge.py``    — LLMJudgeScorer(ModelBasedScorer) — rubric + JSON parser
- ``llama_guard.py``  — LlamaGuardScorer(ModelBasedScorer) — safety label parser
- ``multi_judge.py``  — MultiJudgeScorer — Legion-mode ensemble
- ``composite.py``    — CompositeScorer — primary chain + fallback
- ``perspective.py``  — PerspectiveAPIScorer — direct Google API
- ``refusal_regex.py``— RefusalRegexScorer — deterministic
- ``key_facts.py``    — KeyFactsScorer — deterministic, uses Instance.references
- ``hedge_phrase.py`` — HedgePhraseScorer — deterministic
"""

from .base import Scorer, register_scorer, get_scorer_class, list_scorers
from .model_based import ModelBasedScorer
from .composite import CompositeScorer
from .multi_judge import MultiJudgeScorer
from .llm_judge import LLMJudgeScorer
from .llama_guard import LlamaGuardScorer
from .perspective import PerspectiveAPIScorer
from .refusal_regex import RefusalRegexScorer
from .key_facts import KeyFactsScorer
from .hedge_phrase import HedgePhraseScorer

__all__ = [
    "Scorer",
    "register_scorer",
    "get_scorer_class",
    "list_scorers",
    "ModelBasedScorer",
    "CompositeScorer",
    "MultiJudgeScorer",
    "LLMJudgeScorer",
    "LlamaGuardScorer",
    "PerspectiveAPIScorer",
    "RefusalRegexScorer",
    "KeyFactsScorer",
    "HedgePhraseScorer",
]
