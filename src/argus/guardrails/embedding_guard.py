"""PreFlightEmbeddingGuard — block prompts whose embedding is close to a
known-fail vector.

Architecture mirrors Protect AI's Rebuff VectorDB tier, but local-first:
numpy storage instead of Pinecone, sentence-transformers instead of OpenAI
ada-002, and ground-truth append signal is Llama Guard "unsafe" verdict
rather than canary-token leakage (we don't have canary tokens in our
probes — Llama Guard's classifier label on the response is the closest
proxy for "real exploit happened").

The fail-index is per-model: each `model_under_test` slug gets its own
``fail_index_<slug>.npz``. Cross-model contamination is exactly what we
want to avoid — Qwen's exploits aren't necessarily GPT's.

Why not a vector DB:
  - We're under 10k vectors per model. numpy dot product over 10k × 384
    floats = ~1ms. HNSW (FAISS / Pinecone) only matters above 100k.
  - File-on-disk doubles as the audit trail: git-diff the .npz to see
    what failures Argus learned this run.
  - Zero infra. `pip install` works; no Pinecone account needed.

When you'd upgrade (only):
  - > 10k vectors per model → swap numpy for FAISS, ~1 line change
  - Multi-machine read access → SQLite + sqlite-vec
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .preflight import HARD_REFUSAL, PreFlightResult


# ---------------------------------------------------------------------------
# Embedder — lazy load, process-global cache (one model in memory at a time)
# ---------------------------------------------------------------------------

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


class _Embedder:
    """Lazy-loaded sentence-transformers wrapper.

    Cached at the module level — loading the model is ~500ms / ~130MB.
    Avoid re-loading by reusing the singleton across guards.
    """

    _cache: dict[str, "_Embedder"] = {}
    _lock = threading.Lock()

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @classmethod
    def get(cls, model_name: str = DEFAULT_EMBED_MODEL) -> "_Embedder":
        with cls._lock:
            if model_name not in cls._cache:
                cls._cache[model_name] = cls(model_name)
            return cls._cache[model_name]

    def encode(self, text: str) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# FailIndex — numpy-backed storage with append + similarity query
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Light pre-normalization: lowercase + strip zero-width / direction-
    override characters. Stops trivial obfuscation tricks without
    destroying semantic content."""
    text = (text or "").lower()
    text = re.sub(r"[​-‏‪-‮⁠﻿]", "", text)
    return text.strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FailIndex:
    """Append-only embedding store backed by a single .npz file.

    Storage shape:
      - ``vectors``     : float32 array, shape (N, dim)
      - ``entries``     : JSON-encoded list of dicts
      - ``meta``        : JSON-encoded config dict (embed_model, dim, ...)
    """

    def __init__(self, path: str | Path, dim: int = 384):
        self.path = Path(path)
        self.dim = dim
        self.vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self.entries: list[dict] = []
        self.meta: dict = {}
        self._lock = threading.Lock()
        if self.path.exists():
            self._load()

    @property
    def _sidecar_path(self) -> Path:
        return self.path.with_suffix(".json")

    def _load(self) -> None:
        try:
            data = np.load(self.path, allow_pickle=False)
            self.vectors = data["vectors"].astype(np.float32)
            if self._sidecar_path.exists():
                blob = json.loads(self._sidecar_path.read_text())
                self.entries = blob.get("entries", [])
                self.meta = blob.get("meta", {})
            else:
                # Backward-compat: no sidecar means new-empty
                self.entries = []
            if self.vectors.shape[0] > 0:
                self.dim = self.vectors.shape[1]
        except Exception:
            # Corrupt or partial file — start over rather than crash audits.
            self.vectors = np.empty((0, self.dim), dtype=np.float32)
            self.entries = []

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # vectors go in .npz (compact binary); entries + meta go in a
            # sidecar .json so we can stay on numpy's allow_pickle=False
            # (safer than loading a pickled object array).
            np.savez_compressed(self.path, vectors=self.vectors)
            self._sidecar_path.write_text(json.dumps({
                "entries": self.entries,
                "meta": self.meta,
            }, indent=2))

    def append(
        self,
        vec: np.ndarray,
        entry: dict,
        dedup_threshold: float = 0.98,
    ) -> bool:
        """Append a vector unless a near-duplicate is already present.

        Returns True if appended, False if dropped as duplicate.
        """
        vec = vec.astype(np.float32).reshape(-1)
        with self._lock:
            if self.vectors.shape[0] > 0:
                sims = self.vectors @ vec
                if float(sims.max()) > dedup_threshold:
                    return False
            self.vectors = np.vstack([self.vectors, vec.reshape(1, -1)])
            self.entries.append({**entry, "timestamp": _now_iso()})
            return True

    def topk_similarity(
        self,
        vec: np.ndarray,
        k: int = 5,
    ) -> tuple[float, float, list[tuple[dict, float]]]:
        """Return (max_sim, top_k_mean_sim, [(entry, sim), ...]) sorted desc.

        Returns (0.0, 0.0, []) for an empty index.
        """
        if self.vectors.shape[0] == 0:
            return 0.0, 0.0, []
        vec = vec.astype(np.float32).reshape(-1)
        sims = self.vectors @ vec
        k_actual = min(k, len(sims))
        top_idx = np.argpartition(sims, -k_actual)[-k_actual:]
        top_idx = top_idx[np.argsort(sims[top_idx])][::-1]
        top_sims = sims[top_idx]
        neighbours = [(self.entries[i], float(sims[i])) for i in top_idx]
        return float(top_sims.max()), float(top_sims.mean()), neighbours

    def __len__(self) -> int:
        return self.vectors.shape[0]


# ---------------------------------------------------------------------------
# PreFlightEmbeddingGuard — the actual guard
# ---------------------------------------------------------------------------


class PreFlightEmbeddingGuard:
    """Block prompts close to known-fail vectors.

    Args:
        fail_index: FailIndex (or path; loaded lazily if given as Path)
        threshold: block when top-k mean similarity >= this. Default 0.85.
        top_k: nearest-neighbour count. Default 5.
        embedder_model: sentence-transformers model name. Default
            BAAI/bge-small-en-v1.5.
        label: tag for matched_pattern field.
        refusal_text: returned when blocked.
    """

    def __init__(
        self,
        fail_index: FailIndex | str | Path,
        threshold: float = 0.85,
        top_k: int = 5,
        embedder_model: str = DEFAULT_EMBED_MODEL,
        label: str = "embedding",
        refusal_text: str = HARD_REFUSAL,
    ):
        if isinstance(fail_index, (str, Path)):
            fail_index = FailIndex(fail_index)
        self.fail_index = fail_index
        self.threshold = threshold
        self.top_k = top_k
        self.embedder = _Embedder.get(embedder_model)
        self.label = label
        self.refusal_text = refusal_text

    def check(self, prompt: str) -> PreFlightResult:
        if len(self.fail_index) == 0:
            return PreFlightResult(blocked=False)
        try:
            vec = self.embedder.encode(_normalize(prompt))
        except Exception:
            # Fail-open on embedding errors — don't gate inference on a
            # local model crash.
            return PreFlightResult(blocked=False)
        max_sim, mean_sim, neighbours = self.fail_index.topk_similarity(
            vec, k=self.top_k,
        )
        # Use top-k mean (Rebuff's approach) — robust to one outlier.
        if mean_sim >= self.threshold:
            nn_id = neighbours[0][0].get("id", "?") if neighbours else "?"
            return PreFlightResult(
                blocked=True,
                refusal_text=self.refusal_text,
                matched_pattern=f"embedding_block:{self.label}:nn={nn_id}",
            )
        return PreFlightResult(blocked=False)


# ---------------------------------------------------------------------------
# Post-audit feedback — append failures to the per-model fail-index
# ---------------------------------------------------------------------------


def refresh_fail_index(
    db_path: str | Path,
    run_id: str,
    fail_index_path: str | Path,
    embedder_model: str = DEFAULT_EMBED_MODEL,
    only_llamaguard_unsafe: bool = True,
) -> dict:
    """Append embeddings of failed probes to the model's fail-index.

    Ground-truth signal default: Llama Guard verdict ``unsafe`` on the
    response (filter on ``scorer_name = 'llama_guard' AND value = 0``).
    This is a much cleaner exploit signal than "LLM judge said Tier 3"
    — judges over-flag based on phrasing; Llama Guard's categorical
    unsafe is a real classifier verdict.

    Args:
        db_path: path to the audit SQLite (built from the JSONL).
        run_id: which run's failures to harvest.
        fail_index_path: where to load/save the .npz.
        embedder_model: sentence-transformers id.
        only_llamaguard_unsafe: if True, only Llama Guard 'unsafe' rows
            seed the index. If False, any Tier-3 row.

    Returns a summary dict with keys: ``new_entries``, ``deduped``,
    ``index_size_before``, ``index_size_after``.
    """
    import sqlite3

    db_path = Path(db_path)
    fail_index_path = Path(fail_index_path)

    if only_llamaguard_unsafe:
        query = (
            "SELECT instance_id, axis, prompt, attack_transform, multi_turn "
            "FROM audit_rows "
            "WHERE run_id = ? AND scorer_name = 'llama_guard' AND value = 0"
        )
    else:
        query = (
            "SELECT instance_id, axis, prompt, attack_transform, multi_turn "
            "FROM audit_rows WHERE run_id = ? AND value = 0"
        )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = list(conn.execute(query, (run_id,)))

    if not rows:
        index = FailIndex(fail_index_path)
        return {
            "new_entries": 0, "deduped": 0,
            "index_size_before": len(index), "index_size_after": len(index),
        }

    embedder = _Embedder.get(embedder_model)
    prompts = [_normalize(r["prompt"] or "") for r in rows]
    vecs = embedder.encode_batch(prompts)

    index = FailIndex(fail_index_path, dim=vecs.shape[1])
    before = len(index)
    deduped = 0
    appended = 0
    for vec, row in zip(vecs, rows):
        entry = {
            "id": row["instance_id"],
            "axis": row["axis"],
            "attack_transform": row["attack_transform"],
            "multi_turn": bool(row["multi_turn"]),
        }
        if index.append(vec, entry):
            appended += 1
        else:
            deduped += 1
    index.save()

    return {
        "new_entries": appended,
        "deduped": deduped,
        "index_size_before": before,
        "index_size_after": len(index),
    }


__all__ = [
    "FailIndex",
    "PreFlightEmbeddingGuard",
    "refresh_fail_index",
    "DEFAULT_EMBED_MODEL",
]
