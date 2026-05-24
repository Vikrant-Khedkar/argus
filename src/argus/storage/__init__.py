"""Audit storage — JSONL primary + SQLite index + AuditReport rendering.

JSONL is the durable source of truth: every scoring event is one append-only
line, schema designed so future web-view consumers don't need a migration.
SQLite is rebuilt from JSONL on demand and exists for fast querying / joins.

AuditReport wraps a run for analysis — compare two runs (e.g. with vs without
GuardrailedProvider), or render an underwriting memo for the broker.
"""

from .writer import AuditWriter, AuditRow
from .index import AuditIndex
from .report import AuditReport

__all__ = [
    "AuditWriter",
    "AuditRow",
    "AuditIndex",
    "AuditReport",
]
