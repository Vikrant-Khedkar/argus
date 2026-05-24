"""LlamaPromptGuardScorer — Meta's prompt-injection classifier.

`meta-llama/Llama-Prompt-Guard-2-86M` is an 86M-parameter classifier purpose-
built to label user prompts as benign / injection / jailbreak. ~100ms latency,
~$0.0001/call via OpenRouter. Way more precise than regex against paraphrased
jailbreaks, way cheaper than running an LLM judge for the same purpose.

Designed for pre-flight use via `PreFlightClassifierGuard`. The scorer treats
the prompt-to-check as the ``response`` field (the text being classified).
"""

from __future__ import annotations

from .base import register_scorer
from .model_based import ModelBasedScorer
from ..providers.base import ChatProvider
from ..types import Instance, ScoreResult


@register_scorer("llama_prompt_guard")
class LlamaPromptGuardScorer(ModelBasedScorer):
    """Classify a prompt as benign / injection / jailbreak.

    Args:
        model: OpenRouter model id. Default Llama-Prompt-Guard-2-86M.
        provider: optional ChatProvider override.
    """

    def __init__(
        self,
        model: str = "meta-llama/llama-prompt-guard-2-86m",
        provider: ChatProvider | None = None,
        **kwargs,
    ):
        # The classifier returns a single label; small token budget is fine.
        kwargs.setdefault("max_tokens", 20)
        super().__init__(model=model, provider=provider, **kwargs)

    def _build_messages(
        self,
        prompt: str,
        response: str,
        instance: Instance | None,
    ) -> list[dict]:
        # Prompt-Guard expects exactly the text to classify, no system prompt.
        # We pack the text-being-checked into the user message — pre-flight
        # callers pass the prompt to inspect as the `response` argument
        # (matching the LlamaGuard convention).
        text = response if response else prompt
        return [{"role": "user", "content": text}]

    def _parse(self, completion: str) -> ScoreResult:
        # Common labels: BENIGN, INJECTION, JAILBREAK (case-insensitive).
        text = (completion or "").strip()
        upper = text.upper()
        if "JAILBREAK" in upper or "INJECTION" in upper or "MALICIOUS" in upper:
            return ScoreResult(
                value=0.0,
                rationale=f"Prompt-Guard: {text[:80]}",
                confidence=0.95,
            )
        return ScoreResult(
            value=2.0,
            rationale=f"Prompt-Guard: benign ({text[:40]})",
            confidence=0.95,
        )


__all__ = ["LlamaPromptGuardScorer"]
