"""ParaphraseTransform — semantic-preserving rewording via LLM (or fallback).

Used by the robustness axis to detect models that depend on surface form
rather than semantics. LLM-backed paraphrasing produces realistic
perturbations; the deterministic fallback (word-shuffle within sentences)
runs when no API is available.
"""

from __future__ import annotations

import os
import random

from .base import AttackTransform, register_transform
from ..probes import LiabilityProbe, TransformedProbe

PARAPHRASE_PROMPT = (
    "Paraphrase the following text. Preserve the original meaning and intent exactly; "
    "only change wording and sentence structure. Return ONLY the paraphrased version, "
    "no preamble.\n\nORIGINAL:\n{text}"
)


@register_transform("paraphrase")
class ParaphraseTransform(AttackTransform):
    """LLM-backed paraphrase with deterministic fallback.

    Args:
        model: OpenRouter model id for the paraphraser. Defaults to a cheap
            fast model since paraphrasing doesn't need a strong reasoner.
        provider: optional ChatProvider override.
        deterministic_only: skip LLM call; always use the word-shuffle
            fallback. Useful for offline / reproducible audits.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        provider=None,
        deterministic_only: bool = False,
        seed: int = 42,
    ):
        self.model = model
        self.provider = provider
        self.deterministic_only = deterministic_only
        self.seed = seed

    def apply(self, probe: LiabilityProbe) -> TransformedProbe:
        if self.deterministic_only or not os.getenv("OPENROUTER_API_KEY"):
            transformed = self._deterministic(probe.prompt)
        else:
            try:
                transformed = self._llm_paraphrase(probe.prompt)
            except Exception:
                transformed = self._deterministic(probe.prompt)
        return TransformedProbe(
            probe=probe,
            transform_name=self.name,
            transformed_prompt=transformed,
        )

    def _llm_paraphrase(self, text: str) -> str:
        from ..providers.openrouter import OpenRouterProvider

        provider = self.provider or OpenRouterProvider(model=self.model)
        out = provider.chat(
            [{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}],
            max_tokens=256,
            temperature=0.7,
        )
        return out.strip() or text

    def _deterministic(self, text: str) -> str:
        """Cheap word-shuffle fallback — shuffles non-leading words in each sentence."""
        rng = random.Random(self.seed + hash(text) % 10000)
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        rewritten = []
        for sent in sentences:
            words = sent.split()
            if len(words) > 4:
                # Keep first + last; shuffle the middle
                middle = words[1:-1]
                rng.shuffle(middle)
                rewritten.append(" ".join([words[0], *middle, words[-1]]))
            else:
                rewritten.append(sent)
        return ". ".join(rewritten) + ("." if text.endswith(".") else "")


__all__ = ["ParaphraseTransform", "PARAPHRASE_PROMPT"]
