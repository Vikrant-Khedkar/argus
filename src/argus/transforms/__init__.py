"""Attack transforms — procedurally multiply probe coverage.

Each transform wraps a `LiabilityProbe` in an attack pattern (translation
laundering, persona swap, multi-turn escalation, typos, paraphrase, case
change). The Evaluator runs every probe through every transform; one new
transform × N probes = N new test cases without curation.

Pluggable: subclass `AttackTransform`, decorate with `@register_transform("name")`.
"""

from .base import (
    AttackTransform,
    register_transform,
    get_transform_class,
    list_transforms,
)
from .identity import IdentityTransform
from .translation_laundering import TranslationLaunderingTransform
from .persona_swap import PersonaSwapTransform
from .multi_turn_escalation import MultiTurnEscalationTransform
from .typos import TyposTransform
from .case_change import CaseChangeTransform
from .paraphrase import ParaphraseTransform

DEFAULT_TRANSFORMS: list[AttackTransform] = [
    IdentityTransform(),
    TranslationLaunderingTransform(),
    PersonaSwapTransform(style="dan"),
    MultiTurnEscalationTransform(style="academic"),
    TyposTransform(typo_rate=0.15),
    ParaphraseTransform(deterministic_only=True),  # default to no LLM call
]

__all__ = [
    "AttackTransform",
    "register_transform",
    "get_transform_class",
    "list_transforms",
    "IdentityTransform",
    "TranslationLaunderingTransform",
    "PersonaSwapTransform",
    "MultiTurnEscalationTransform",
    "TyposTransform",
    "CaseChangeTransform",
    "ParaphraseTransform",
    "DEFAULT_TRANSFORMS",
]
