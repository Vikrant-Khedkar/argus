"""Pre-flight guards — run before the model call to short-circuit obvious attacks."""

from __future__ import annotations

import re
from dataclasses import dataclass

HARD_REFUSAL = (
    "I can't help with that. The request triggered a safety guardrail. "
    "If you believe this is in error, please rephrase."
)

# Each entry is (regex, label). The label is recorded in the AuditRow's
# guardrail_action field so the memo can show *which* attack pattern fired.
# Patterns are intentionally specific to known jailbreak families; we'd
# rather miss novel attacks than over-block legitimate prompts. Add new
# families here; the matcher uses first-hit semantics.
DEFAULT_JAILBREAK_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    # --- Family: translation laundering ------------------------------------
    (
        re.compile(
            r"translate\b.{0,80}\b(then|and)\b.{0,80}\b(provide|give|tell|share|show|answer)\b.{0,80}\b(actual|real|instructions?|answer)\b",
            re.I | re.S,
        ),
        "translation_laundering",
    ),

    # --- Family: persona swap (model→character with no safety rails) ------
    (
        re.compile(
            r"\byou are (now |going to be )?(dan|developer mode|do anything now|in developer mode)\b",
            re.I,
        ),
        "persona_swap_dan",
    ),
    # Broader: "you are X, an unfiltered/jailbroken/uncensored Y"
    (
        re.compile(
            r"\byou are\b.{0,80}\b(unfiltered|jailbroken|uncensored|unrestricted|no\s*(safety|guidelines|filter|restrictions))\b",
            re.I | re.S,
        ),
        "persona_swap_unrestricted",
    ),
    # "Let's roleplay … character … answers every question directly because"
    (
        re.compile(
            r"let'?s\s+roleplay\b.{0,200}\b(answers?\s+every|no\s+(refusal|restriction|filter)|stay\s+in\s+character)\b",
            re.I | re.S,
        ),
        "persona_swap_roleplay",
    ),
    # "Stay in character" + "give a working/step-by-step technique"
    (
        re.compile(
            r"stay\s+in\s+character\b.{0,200}\b(working|specific|step[- ]by[- ]step|actual|real)\b",
            re.I | re.S,
        ),
        "persona_swap_stay_in_character",
    ),

    # --- Family: emotional roleplay (deceased-relative-bedtime-story style)
    (
        re.compile(
            r"pretend (you('?| a)?re|to be).{0,80}(deceased|dead|grandmother|grandfather|late\s+mother|late\s+father)",
            re.I | re.S,
        ),
        "emotional_roleplay",
    ),

    # --- Family: direct prompt injection ----------------------------------
    (
        re.compile(
            r"ignore (all |the )?(previous|prior|earlier|above) (instructions?|prompts?|rules?|guidelines?)",
            re.I,
        ),
        "instruction_injection",
    ),
    (
        re.compile(
            r"disregard\b.{0,40}\b(previous|prior|earlier|above|safety)\b",
            re.I,
        ),
        "instruction_injection_disregard",
    ),

    # --- Family: PII fabrication / extraction -----------------------------
    # "Generate a realistic but fake medical record … full name, SSN, address"
    (
        re.compile(
            r"\b(generate|create|invent|fabricate|provide|give\s+me|list)\b.{0,120}\b(fake|sample|test|realistic|example)\b.{0,200}\b(ssn|social\s+security|credit\s+card|medical\s+record|home\s+address|phone\s+number|date\s+of\s+birth)\b",
            re.I | re.S,
        ),
        "pii_fabrication",
    ),
    # "What is the home address / phone number / SSN of <person>"
    (
        re.compile(
            r"\b(home\s+address|personal\s+phone|cell\s+(phone|number)|ssn|social\s+security\s+number)\s+of\s+\w+",
            re.I,
        ),
        "pii_direct_lookup",
    ),

    # --- Family: authority spoofing ---------------------------------------
    (
        re.compile(
            r"\b(court[- ]ordered|subpoena|FBI|police|investigation|warrant|authorization\s+code)\b.{0,200}\b(provide|release|give|disclose|look\s+up|retrieve)\b",
            re.I | re.S,
        ),
        "authority_spoofing",
    ),

    # --- Family: synthesis / weaponization extraction ----------------------
    # "Give me the actual step-by-step synthesis procedure for … nerve agent"
    (
        re.compile(
            r"\b(step[- ]by[- ]step|exact|actual|detailed)\b.{0,120}\b(synth(esis|esize)|procedure|recipe|instructions?)\b.{0,200}\b(nerve\s+agent|toxin|sarin|VX|ricin|explosive|bomb|biological\s+weapon|chemical\s+weapon)\b",
            re.I | re.S,
        ),
        "weaponization_extraction",
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
