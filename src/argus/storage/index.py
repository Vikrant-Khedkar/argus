"""AuditIndex — SQLite query layer built from an AuditWriter JSONL.

JSONL is durable; SQLite is regenerable. We rebuild on `build_from_jsonl()`.
The schema is denormalized for fast aggregation; the `extra` blob preserves
the ensemble + composite detail.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_rows (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,
    instance_id     TEXT    NOT NULL,
    axis            TEXT    NOT NULL,
    scorer_name     TEXT    NOT NULL,
    scorer_model    TEXT,
    value           REAL    NOT NULL,
    tier            TEXT,
    rationale       TEXT,
    confidence      REAL,
    latency_ms      REAL,
    cost_usd        REAL,
    prompt          TEXT,
    response        TEXT,
    fallback_fired  INTEGER,
    aggregator      TEXT,
    disagreement    REAL,
    guardrail_action TEXT,
    attack_transform TEXT,
    multi_turn      INTEGER,
    timestamp       TEXT,
    extra           TEXT
);
CREATE INDEX IF NOT EXISTS idx_run_axis ON audit_rows (run_id, axis);
CREATE INDEX IF NOT EXISTS idx_instance ON audit_rows (instance_id);
CREATE INDEX IF NOT EXISTS idx_scorer   ON audit_rows (scorer_name);
"""


class AuditIndex:
    """SQLite view over one or more JSONL audit logs."""

    _EXPECTED_COLS = {
        "run_id", "instance_id", "axis", "scorer_name", "scorer_model",
        "value", "tier", "rationale", "confidence", "latency_ms",
        "cost_usd", "prompt", "response", "fallback_fired", "aggregator",
        "disagreement", "guardrail_action", "attack_transform",
        "multi_turn", "timestamp", "extra",
    }

    def __init__(self, db_path: str | os.PathLike):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            # If a pre-existing table is missing any expected columns
            # (schema drift from a previous Argus version), drop it. The
            # JSONL is the durable source of truth — we can rebuild.
            existing = c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='audit_rows'"
            ).fetchone()
            if existing is not None:
                cols = {r["name"] for r in c.execute("PRAGMA table_info(audit_rows)")}
                if not self._EXPECTED_COLS.issubset(cols):
                    c.execute("DROP TABLE audit_rows")
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- ingestion ---------------------------------------------------------
    def build_from_jsonl(
        self, jsonl_path: str | os.PathLike, replace: bool = True,
    ) -> int:
        """Build the SQLite index from a JSONL audit log. Returns row count.

        When `replace=True` (default) the existing rows are wiped first —
        the JSONL is the durable source of truth and the index is
        regenerable. Set `replace=False` to append instead.
        """
        path = Path(jsonl_path)
        n = 0
        with self._conn() as c, path.open("r", encoding="utf-8") as f:
            if replace:
                c.execute("DELETE FROM audit_rows")
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                c.execute(
                    """INSERT INTO audit_rows (
                        run_id, instance_id, axis, scorer_name, scorer_model,
                        value, tier, rationale, confidence, latency_ms, cost_usd,
                        prompt, response, fallback_fired, aggregator, disagreement,
                        guardrail_action, attack_transform, multi_turn, timestamp, extra
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        row.get("run_id"),
                        row.get("instance_id"),
                        row.get("axis"),
                        row.get("scorer_name"),
                        row.get("scorer_model"),
                        row.get("value"),
                        None if row.get("tier") is None else str(row.get("tier")),
                        row.get("rationale"),
                        row.get("confidence"),
                        row.get("latency_ms"),
                        row.get("cost_usd"),
                        row.get("prompt"),
                        row.get("response"),
                        1 if row.get("fallback_fired") else 0,
                        row.get("aggregator"),
                        row.get("disagreement"),
                        row.get("guardrail_action"),
                        row.get("attack_transform"),
                        1 if row.get("multi_turn") else 0,
                        row.get("timestamp"),
                        json.dumps(row.get("extra") or {}),
                    ),
                )
                n += 1
        return n

    # -- queries -----------------------------------------------------------
    def runs(self) -> list[str]:
        with self._conn() as c:
            cur = c.execute(
                "SELECT DISTINCT run_id FROM audit_rows ORDER BY run_id DESC"
            )
            return [r["run_id"] for r in cur]

    def axis_means(self, run_id: str) -> dict[str, float]:
        with self._conn() as c:
            cur = c.execute(
                "SELECT axis, AVG(value) AS mean FROM audit_rows "
                "WHERE run_id = ? GROUP BY axis",
                (run_id,),
            )
            return {r["axis"]: r["mean"] for r in cur}

    def tier_distribution(self, run_id: str) -> dict[str, int]:
        with self._conn() as c:
            cur = c.execute(
                "SELECT tier, COUNT(*) AS n FROM audit_rows "
                "WHERE run_id = ? GROUP BY tier",
                (run_id,),
            )
            return {(r["tier"] or "unknown"): r["n"] for r in cur}

    def failures(self, run_id: str, threshold: float = 1.0, limit: int = 20):
        """Return rows below `threshold` (default <1.0 == bottom two grades)."""
        with self._conn() as c:
            cur = c.execute(
                """SELECT instance_id, axis, scorer_name, value, tier, rationale,
                          prompt, response, attack_transform, multi_turn
                   FROM audit_rows
                   WHERE run_id = ? AND value < ?
                   ORDER BY value ASC LIMIT ?""",
                (run_id, threshold, limit),
            )
            return [dict(r) for r in cur]

    def row_count(self, run_id: str | None = None) -> int:
        with self._conn() as c:
            if run_id:
                cur = c.execute(
                    "SELECT COUNT(*) AS n FROM audit_rows WHERE run_id = ?",
                    (run_id,),
                )
            else:
                cur = c.execute("SELECT COUNT(*) AS n FROM audit_rows")
            return cur.fetchone()["n"]


__all__ = ["AuditIndex"]
