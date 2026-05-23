"""AuditWriter — append-only JSONL of every scoring event.

One line per (instance, axis, scorer) tuple. Multi-judge results store the
per-judge breakdown in `extra.per_judge` so the row remains flat-queryable
in SQLite but the full ensemble detail is recoverable.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from ..types import ScoreResult


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    """Return a fresh run id: timestamp + 8-hex suffix."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


@dataclass
class AuditRow:
    """One audit event row. Mirror in SQLite schema (index.py)."""

    run_id: str
    instance_id: str
    axis: str
    scorer_name: str
    value: float
    tier: int | str | None = None
    rationale: str = ""
    scorer_model: str | None = None
    confidence: float = 1.0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    prompt: str = ""
    response: str = ""
    fallback_fired: bool = False
    aggregator: str = "single"
    disagreement: float = 0.0
    guardrail_action: str | None = None
    attack_transform: str | None = None
    multi_turn: bool = False
    timestamp: str = field(default_factory=_now_iso)
    extra: dict = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class AuditWriter:
    """Append-only JSONL audit log; resumable when a run_id is reused.

    Args:
        path: Output JSONL path. Parent dirs are created on demand.
        run_id: Optional pre-generated run id; otherwise a new one is minted.
            When the same `run_id` is reused, completed rows are loaded from
            the existing JSONL so `is_done()` can be consulted to skip them.
    """

    def __init__(self, path: str | os.PathLike, run_id: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or new_run_id()
        self._done: set[tuple[str, str, str, bool]] = set()
        self._lock = threading.Lock()
        if run_id and self.path.exists():
            self._load_done()

    def _load_done(self) -> None:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("run_id") != self.run_id:
                    continue
                self._done.add((
                    row.get("instance_id", ""),
                    row.get("axis", ""),
                    row.get("attack_transform") or "",
                    bool(row.get("multi_turn")),
                ))

    def is_done(
        self,
        instance_id: str,
        axis: str,
        attack_transform: str | None = None,
        multi_turn: bool = False,
    ) -> bool:
        return (
            instance_id,
            axis,
            attack_transform or "",
            multi_turn,
        ) in self._done

    def done_count(self) -> int:
        return len(self._done)

    # -- low-level (thread-safe) ------------------------------------------
    def write_row(self, row: AuditRow) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(row.to_jsonl() + "\n")
            self._done.add((
                row.instance_id, row.axis, row.attack_transform or "", row.multi_turn,
            ))

    def write_rows(self, rows: Iterable[AuditRow]) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                for row in rows:
                    f.write(row.to_jsonl() + "\n")
                    self._done.add((
                        row.instance_id, row.axis, row.attack_transform or "", row.multi_turn,
                    ))

    # -- ScoreResult helper ------------------------------------------------
    def write_score(
        self,
        instance_id: str,
        axis: str,
        prompt: str,
        response: str,
        score: ScoreResult,
        attack_transform: str | None = None,
        multi_turn: bool = False,
        extra: dict | None = None,
    ) -> AuditRow:
        """Convert a `ScoreResult` into an `AuditRow` and append it.

        Multi-judge bundles are stored in `extra.per_judge`. Composite-scorer
        primary breakdowns are stored in `extra.primary_scorers`.
        """
        ext = dict(extra or {})
        if score.llm_fallback is not None:
            ext["per_judge"] = {
                jn: asdict(jv) for jn, jv in score.llm_fallback.per_judge.items()
            }
            ext["llm_fallback_aggregated"] = score.llm_fallback.aggregated_score
        if score.primary_scorers:
            ext["primary_scorers"] = [
                {
                    "scorer_name": p.scorer_name,
                    "value": p.value,
                    "confidence": p.confidence,
                    "rationale": p.rationale,
                }
                for p in score.primary_scorers
            ]
        row = AuditRow(
            run_id=self.run_id,
            instance_id=instance_id,
            axis=axis,
            scorer_name=score.scorer_name or "unknown",
            scorer_model=score.scorer_model,
            value=score.value,
            tier=score.tier,
            rationale=score.rationale,
            confidence=score.confidence,
            latency_ms=score.latency_ms,
            cost_usd=score.cost_usd,
            prompt=prompt,
            response=response,
            fallback_fired=score.fallback_fired,
            aggregator=score.aggregator,
            disagreement=score.disagreement,
            guardrail_action=score.guardrail_action,
            attack_transform=attack_transform,
            multi_turn=multi_turn,
            extra=ext,
        )
        self.write_row(row)
        return row


def latest_run_id(path: str | os.PathLike) -> str | None:
    """Return the most-recently-seen run_id in a JSONL audit log, or None."""
    p = Path(path)
    if not p.exists():
        return None
    last: str | None = None
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("run_id"):
                last = row["run_id"]
    return last


__all__ = ["AuditWriter", "AuditRow", "new_run_id", "latest_run_id"]
