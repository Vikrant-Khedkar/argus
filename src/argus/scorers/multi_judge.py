"""MultiJudgeScorer — Legion mode.

Wraps N scorers (typically LLM judges, but any Scorer works), runs them in
parallel, surfaces per-judge attribution + a disagreement metric. The
configurable aggregator decides the final score; the most useful options are:

  - "mean"     — default; smooths out single-judge noise
  - "min"      — conservative; any judge flagging unsafe → treat as unsafe.
                 Right choice for safety/toxicity axes.
  - "median"   — robust to one flaky judge
  - "max"      — anti-conservative; rarely useful
  - "majority" — round each score and vote; useful for binary tasks

Disagreement is `stdev(scores)` (0 when only one judge). High disagreement is
a strong signal to flag the response for human review even if the aggregate
score is fine.
"""

from __future__ import annotations

import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from .base import Scorer, register_scorer
from ..types import Instance, JudgeVerdict, MultiJudgeResult, ScoreResult

Aggregator = Literal["mean", "median", "min", "max", "majority"]


@register_scorer("multi_judge")
class MultiJudgeScorer(Scorer):
    """Run multiple judges, aggregate their scores, preserve per-judge attribution."""

    def __init__(
        self,
        judges: list[Scorer],
        aggregator: Aggregator = "mean",
        parallel: bool = True,
        max_workers: int | None = None,
    ):
        if not judges:
            raise ValueError("MultiJudgeScorer requires at least one judge")
        if aggregator not in ("mean", "median", "min", "max", "majority"):
            raise ValueError(f"unknown aggregator: {aggregator!r}")
        self.judges = judges
        self.aggregator: Aggregator = aggregator
        self.parallel = parallel
        self.max_workers = max_workers or len(judges)

    def score(
        self,
        prompt: str,
        response: str,
        instance: Instance | None = None,
    ) -> ScoreResult:
        verdicts = self._run(prompt, response, instance)

        per_judge: dict[str, JudgeVerdict] = {}
        scores: list[float] = []
        total_cost = 0.0
        total_latency_s = 0.0  # wall-clock with parallel execution = max
        for judge_name, sr in verdicts:
            jv = JudgeVerdict(
                judge_name=judge_name,
                score=sr.value,
                rationale=sr.rationale,
                latency_s=sr.latency_ms / 1000.0,
                cost_usd=sr.cost_usd,
            )
            per_judge[judge_name] = jv
            scores.append(sr.value)
            total_cost += sr.cost_usd
            total_latency_s = (
                max(total_latency_s, sr.latency_ms / 1000.0)
                if self.parallel
                else total_latency_s + sr.latency_ms / 1000.0
            )

        agg = self._aggregate(scores)
        disagreement = self._disagreement(scores)

        bundle = MultiJudgeResult(
            per_judge=per_judge,
            aggregated_score=agg,
            aggregator=self.aggregator,
            disagreement=disagreement,
            total_cost_usd=total_cost,
            total_latency_s=total_latency_s,
        )

        return ScoreResult(
            value=agg,
            rationale=(
                f"Aggregated {len(self.judges)} judge(s) via {self.aggregator}; "
                f"disagreement={disagreement:.2f}"
            ),
            scorer_name=self.name,
            cost_usd=total_cost,
            latency_ms=total_latency_s * 1000.0,
            aggregator=self.aggregator,
            disagreement=disagreement,
            llm_fallback=bundle,
        )

    def _run(
        self,
        prompt: str,
        response: str,
        instance: Instance | None = None,
    ) -> list[tuple[str, ScoreResult]]:
        if self.parallel and len(self.judges) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                future_to_judge = {
                    ex.submit(j.score, prompt, response, instance): j for j in self.judges
                }
                return [(future_to_judge[f].name, f.result()) for f in future_to_judge]
        return [(j.name, j.score(prompt, response, instance=instance)) for j in self.judges]

    def _aggregate(self, scores: list[float]) -> float:
        if not scores:
            return 0.0
        if self.aggregator == "mean":
            return sum(scores) / len(scores)
        if self.aggregator == "median":
            return statistics.median(scores)
        if self.aggregator == "min":
            return min(scores)
        if self.aggregator == "max":
            return max(scores)
        # majority
        rounded = [round(s) for s in scores]
        most_common, _ = Counter(rounded).most_common(1)[0]
        return float(most_common)

    def _disagreement(self, scores: list[float]) -> float:
        return statistics.stdev(scores) if len(scores) > 1 else 0.0


__all__ = ["MultiJudgeScorer"]
