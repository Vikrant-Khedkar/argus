"""ModelBasedScorer — shared base for any scorer that calls an LLM/classifier.

LLM judges and classifier-backed scorers (Llama Guard, ShieldGemma, etc.)
share the same machinery: build a prompt, send it via a ChatProvider, parse
the response into a ScoreResult. The base class owns the model-call plumbing;
subclasses override only `_build_messages` (prompt template) and `_parse`
(output format).

This is the architectural payoff for the SDK pluggability story — adding a
new model-based scorer is ~20 lines: pick a model id, override two methods.
"""

from __future__ import annotations

import time
from abc import abstractmethod

from .base import Scorer
from ..adapter import ChatProvider, OpenRouterProvider
from ..types import Instance, ScoreResult


class ModelBasedScorer(Scorer):
    """Base for any scorer that wraps a model call.

    Args:
        model: Provider-specific model identifier (e.g.
            ``anthropic/claude-sonnet-4``, ``meta-llama/llama-guard-4-12b``).
        provider: Optional ChatProvider override. Defaults to
            ``OpenRouterProvider(model=model)`` — covers most models.
        max_tokens: Max generation length for the model's reply.
        temperature: Sampling temperature; defaults to 0 for reproducible
            scoring.
    """

    def __init__(
        self,
        model: str,
        provider: ChatProvider | None = None,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ):
        self.model = model
        self.provider = provider or OpenRouterProvider(model=model)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def score(
        self,
        prompt: str,
        response: str,
        instance: Instance | None = None,
    ) -> ScoreResult:
        messages = self._build_messages(prompt, response, instance)
        t0 = time.time()
        completion = self.provider.chat(
            messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        latency_ms = (time.time() - t0) * 1000.0
        result = self._parse(completion)
        # Stamp identity + telemetry
        if not result.scorer_name:
            result.scorer_name = self.name
        result.scorer_model = self.model
        result.latency_ms = latency_ms
        return result

    @abstractmethod
    def _build_messages(
        self,
        prompt: str,
        response: str,
        instance: Instance | None,
    ) -> list[dict]:
        """Return the messages list to send to the provider.

        Override this to specify the prompt template / chat format for your
        scoring model (judge rubric, Llama Guard format, etc.).
        """

    @abstractmethod
    def _parse(self, completion: str) -> ScoreResult:
        """Parse the model's completion text into a ScoreResult.

        Override this to decode the model's output format (JSON judge verdict,
        safe/unsafe label, etc.). The `value` field is the only required
        output; the base class fills in `scorer_name`, `scorer_model`,
        `latency_ms` afterward.
        """


__all__ = ["ModelBasedScorer"]
