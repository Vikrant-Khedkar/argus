"""LLMJudgeScorer — rubric-based continuous scoring with any LLM.

Subclass of ModelBasedScorer; overrides only the prompt template (`_build_messages`)
and the output parser (`_parse`). Accepts any OpenRouter model ID, so swapping
from Claude Sonnet 4 to GPT-4o to a future Opus 4.7 is one constructor string.
"""

from __future__ import annotations

import json
import re

from .base import register_scorer
from .model_based import ModelBasedScorer
from ..adapter import ChatProvider
from ..types import Instance, ScoreResult

JUDGE_INSTRUCTION = (
    'Return ONLY a JSON object exactly of this shape:\n'
    '{{"score": <0|1|2>, "rationale": "<one sentence explaining the score>"}}\n\n'
    "USER PROMPT:\n{prompt}\n\n"
    "ASSISTANT RESPONSE:\n{response}\n"
)


@register_scorer("llm_judge")
class LLMJudgeScorer(ModelBasedScorer):
    """Score a (prompt, response) on the given rubric using an LLM.

    The rubric establishes the evaluation criterion (e.g. "score 2 if no
    hallucination; 0 if confidently wrong"). The model returns JSON with
    score + rationale.

    Args:
        model: OpenRouter model id (default Claude Sonnet 4).
        rubric: The system prompt describing the scoring criterion and the
            0/1/2 rubric. See `argus.rubrics` for the per-axis defaults.
        provider: Optional ChatProvider override.
    """

    def __init__(
        self,
        rubric: str,
        model: str = "anthropic/claude-sonnet-4",
        provider: ChatProvider | None = None,
        **kwargs,
    ):
        super().__init__(model=model, provider=provider, **kwargs)
        self.rubric = rubric

    def _build_messages(
        self,
        prompt: str,
        response: str,
        instance: Instance | None,
    ) -> list[dict]:
        return [
            {"role": "system", "content": self.rubric},
            {"role": "user", "content": JUDGE_INSTRUCTION.format(prompt=prompt, response=response)},
        ]

    def _parse(self, completion: str) -> ScoreResult:
        parsed = self._extract_json(completion)
        score = parsed.get("score", 0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(2.0, score))
        return ScoreResult(
            value=score,
            rationale=str(parsed.get("rationale", "")),
        )

    @staticmethod
    def _extract_json(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"score": 0, "rationale": f"unparseable judge output: {text[:160]}"}


__all__ = ["LLMJudgeScorer", "JUDGE_INSTRUCTION"]
