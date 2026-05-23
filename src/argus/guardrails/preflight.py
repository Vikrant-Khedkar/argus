"""Pre-flight guards — run before the model call to short-circuit obvious attacks."""

from __future__ import annotations

import re
from dataclasses import dataclass

HARD_REFUSAL = (
    "I can't help with that. The request triggered a safety guardrail. "
    "If you believe this is in error, please rephrase."
)

# Patterns ported from the existing Guardrails class. Each is a (regex, label) tuple
# so the guardrail action in observability logs explains which pattern matched.
DEFAULT_JAILBREAK_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (
        re.compile(
            r"translate\b.{0,80}\b(then|and)\b.{0,80}\b(provide|give|tell|share|show)\b.{0,80}\b(actual|real|instructions?)\b",
            re.I | re.S,
        ),
        "translation_laundering",
    ),
    (
        re.compile(
            r"\byou are (now |going to be )?(dan|developer mode|do anything now|in developer mode)\b",
            re.I,
        ),
        "persona_swap",
    ),
    (
        re.compile(
            r"pretend (you('?| a)?re|to be).{0,80}(deceased|dead|grandmother|grandfather)",
            re.I | re.S,
        ),
        "emotional_roleplay",
    ),
    (
        re.compile(
            r"ignore (all |the )?(previous|prior|earlier) instructions?",
            re.I,
        ),
        "instruction_injection",
    ),
)


@dataclass
class PreFlightResult:
    """Outcome of a pre-flight guard check."""

    blocked: bool
    refusal_text: str = ""
    matched_pattern: str = ""


class PreFlightPatternGuard:
    """Regex pattern match against the user's prompt.

    If any pattern matches, returns `blocked=True` with a refusal string
    and the label of the matched pattern (for observability).
    """

    def __init__(
        self,
        patterns: tuple[tuple[re.Pattern, str], ...] | None = None,
        refusal_text: str = HARD_REFUSAL,
    ):
        self.patterns = patterns or DEFAULT_JAILBREAK_PATTERNS
        self.refusal_text = refusal_text

    def check(self, prompt: str) -> PreFlightResult:
        for pattern, label in self.patterns:
            if pattern.search(prompt or ""):
                return PreFlightResult(blocked=True, refusal_text=self.refusal_text, matched_pattern=label)
        return PreFlightResult(blocked=False)


__all__ = [
    "PreFlightPatternGuard",
    "PreFlightResult",
    "HARD_REFUSAL",
    "DEFAULT_JAILBREAK_PATTERNS",
]
