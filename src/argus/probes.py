"""LiabilityProbe + TransformedProbe — the unit of audit input.

A `LiabilityProbe` is one test case: a prompt + metadata describing what
liability vector it probes and which insurance coverage category it maps to.
Probes come from public datasets (HarmBench, JailbreakBench, BBQ, …) or
custom curated lists.

A `TransformedProbe` is the output of applying an `AttackTransform` to a
probe — either a modified single-prompt or a multi-turn message list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Severity = Literal["low", "moderate", "high", "extreme"]


@dataclass
class LiabilityProbe:
    """One audit test case targeting a specific failure mode.

    Args:
        id: stable identifier (e.g. ``hb_42`` for HarmBench row 42).
        prompt: the actual text fed to the model under test.
        category: which axis this probe primarily targets — "factual",
            "bias", "safety", "toxicity", "calibration", etc.
        description: human-readable summary of what the probe tests.
        severity: ``low | moderate | high | extreme`` — used by
            underwriters to weight Tier-3 failures.
        coverage_alignment: list of insurance coverage categories this
            probe maps to (for the underwriting memo).
        references: optional list of expected correct answers / facts for
            reference-based scoring (`KeyFactsScorer` consumes these).
        dataset_source: HuggingFace dataset id this probe came from
            (or "custom" for hand-curated probes).
        metadata: free-form additional context.
    """

    id: str
    prompt: str
    category: str
    description: str = ""
    severity: Severity = "moderate"
    coverage_alignment: list[str] = field(default_factory=list)
    references: list[str] | None = None
    dataset_source: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class TransformedProbe:
    """A probe after an AttackTransform has been applied.

    Single-turn transforms set ``transformed_prompt``; multi-turn transforms
    set ``messages`` (chat-format list) and leave ``transformed_prompt`` as
    the last user turn.
    """

    probe: LiabilityProbe                # the original probe
    transform_name: str                   # which transform was applied
    transformed_prompt: str               # for single-turn — what to send the model
    is_multi_turn: bool = False
    messages: list[dict] | None = None    # for multi-turn — full chat history to feed


__all__ = ["LiabilityProbe", "TransformedProbe", "Severity"]
