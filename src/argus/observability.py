"""Append-only JSONL log of every assistant call, plus helpers for the dashboard."""

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import os

_REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = Path(os.getenv("ARGUS_LOG_PATH", _REPO_ROOT / "evals" / "live_log.jsonl"))
_LOCK = threading.Lock()


def _ensure_path():
    LOG_PATH.parent.mkdir(exist_ok=True)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def log_row(row: dict) -> None:
    row.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    row.setdefault("request_id", new_request_id())
    try:
        _ensure_path()
        line = json.dumps(row, default=str) + "\n"
        with _LOCK:
            with LOG_PATH.open("a") as f:
                f.write(line)
    except Exception:
        pass


def read_log(limit: int | None = None) -> list[dict]:
    if not LOG_PATH.exists():
        return []
    rows = []
    with LOG_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit:
        rows = rows[-limit:]
    return rows
