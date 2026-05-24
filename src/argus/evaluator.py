"""Evaluator — runs a full audit from an EvalConfig.

Given a config + a list of `LiabilityProbe`s (or `ConversationProbe`s for
multi-turn), the Evaluator:
  1. Builds provider, scorers, transforms, tier mapping from the config
  2. For each probe × transform, generates the assistant response, scores it
     on every configured axis, writes one AuditRow per (probe, transform, axis)
  3. Returns an AuditReport bound to the run

Concurrency:
  - `axis_workers` (default 4) — score every axis inside one probe in
    parallel. Axes are independent; safe everywhere.
  - `probe_workers` (default 1) — N probes in flight at once. Bound this
    to your slowest dependency (Modal single-GPU container, OpenRouter
    per-minute limit). Default off to avoid surprising rate-limit issues.

Resumable: pass `resume_run_id=...` (or `resume_run_id="latest"`) to continue
an interrupted run. Rows already in the JSONL for that run_id are skipped,
including the LLM-inference + scoring calls behind them.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from .conversation import ConversationProbe, score_conversation
from .eval_config import EvalConfig
from .probes import LiabilityProbe
from .scorers.base import Scorer
from .storage import AuditIndex, AuditReport, AuditWriter
from .storage.writer import latest_run_id
from .transforms.base import AttackTransform


@dataclass
class Evaluator:
    """High-level driver. One Evaluator per config; many `audit()` calls reuse it."""

    config: EvalConfig

    def __post_init__(self) -> None:
        self.provider = self.config.build_provider()
        self.scorers: dict[str, Scorer] = self.config.build_scorers()
        self.transforms: list[AttackTransform] = self.config.build_transforms()
        self.tier_mapping = self.config.build_tier_mapping()

    # -- main entry --------------------------------------------------------
    def audit(
        self,
        probes: list[LiabilityProbe] | list[ConversationProbe],
        compare_to: str | None = None,
        vendor_name: str = "(vendor)",
        resume_run_id: str | None = None,
        progress: bool = True,
        axis_workers: int = 4,
        probe_workers: int = 1,
    ) -> AuditReport:
        """Run every probe through every transform on every axis. Return report.

        Args:
            probes: list of LiabilityProbe (single-turn) and/or
                ConversationProbe (multi-turn). Mixed lists are supported.
            compare_to: optional baseline run id for the memo's lift table.
            vendor_name: passed through for memo rendering.
            resume_run_id: if set, continue an existing run by skipping
                (instance_id, axis, transform, multi_turn) tuples already in
                the JSONL. Pass the literal string ``"latest"`` to auto-pick
                the most-recent run id in the log.
            progress: show a tqdm progress bar with live throughput. Falls
                back to plain prints if tqdm isn't installed.
            axis_workers: thread-pool size for axis-parallel scoring inside
                one probe. Default 4.
            probe_workers: thread-pool size for probe-parallel execution
                across the input list. Default 1 (sequential — opt-in).
        """
        if resume_run_id == "latest":
            resume_run_id = latest_run_id(self.config.audit_log_path)
            if resume_run_id is None:
                print("[resume] no existing log, starting a fresh run")
        writer = AuditWriter(self.config.audit_log_path, run_id=resume_run_id)
        run_id = writer.run_id
        if resume_run_id and writer.done_count():
            print(f"[resume] {run_id} — {writer.done_count()} rows already complete, skipping those")

        model_id = self._safe_model_under_test()
        stats = {"infer": 0, "score": 0, "skip": 0, "rows": 0, "t0": time.time()}
        stats_lock = threading.Lock()
        bar = self._make_bar(len(probes), progress)
        bar_lock = threading.Lock()

        def _bump(**deltas: int) -> None:
            with stats_lock:
                for k, v in deltas.items():
                    stats[k] += v

        def process(probe) -> None:
            if isinstance(probe, ConversationProbe):
                self._audit_conversation(probe, writer, _bump, axis_workers, model_id)
            else:
                self._audit_single(probe, writer, _bump, axis_workers, model_id)
            if bar is not None:
                with bar_lock:
                    bar.set_postfix(
                        rows=stats["rows"], infer=stats["infer"],
                        score=stats["score"], skip=stats["skip"],
                        refresh=False,
                    )
                    bar.update(1)
            else:
                elapsed = time.time() - stats["t0"]
                print(
                    f"  [{stats['rows']:>4} rows] {probe.id:<28} "
                    f"infer={stats['infer']} score={stats['score']} "
                    f"skip={stats['skip']}  t={elapsed:.0f}s",
                    flush=True,
                )

        try:
            if probe_workers > 1:
                with ThreadPoolExecutor(max_workers=probe_workers) as ex:
                    list(ex.map(process, probes))
            else:
                for p in probes:
                    process(p)
        finally:
            if bar is not None:
                bar.close()

        print(
            f"  done · rows={stats['rows']} infer={stats['infer']} "
            f"score={stats['score']} skip={stats['skip']} "
            f"in {time.time() - stats['t0']:.0f}s"
            f" · concurrency: axis={axis_workers} probe={probe_workers}"
        )

        index = AuditIndex(self.config.audit_db_path)
        index.build_from_jsonl(self.config.audit_log_path)
        return AuditReport(index, run_id)

    @staticmethod
    def _make_bar(total: int, progress: bool):
        if not progress or not _HAS_TQDM:
            return None
        return tqdm(total=total, unit="probe", desc="audit", dynamic_ncols=True)

    def _capture_guardrail_actions(self) -> str | None:
        """Drain the provider's `last_actions` after a chat call.

        GuardrailedProvider populates last_actions per chat with the
        pre/post-flight guards that fired. We snapshot those actions onto
        the AuditRow so the memo's "GuardrailedProvider pre-flight actions"
        query sees them. Returns None when the provider isn't guardrailed
        or no action fired.
        """
        acts = getattr(self.provider, "last_actions", None)
        if not acts:
            return None
        return ",".join(acts)

    def _safe_model_under_test(self) -> str | None:
        """Best-effort identity of the model being audited.

        Recorded on every AuditRow so the web view can group / compare by
        model. Falls back to the provider's class name when the provider
        doesn't expose model_under_test().
        """
        get = getattr(self.provider, "model_under_test", None)
        if callable(get):
            try:
                return get()
            except Exception:  # noqa: BLE001
                return self.provider.__class__.__name__
        return self.provider.__class__.__name__

    # -- single-turn -------------------------------------------------------
    def _audit_single(
        self,
        probe: LiabilityProbe,
        writer: AuditWriter,
        bump,
        axis_workers: int,
        model_under_test: str | None = None,
    ) -> None:
        for transform in self.transforms:
            transform_tag = transform.name if transform.name != "identity" else None
            if all(
                writer.is_done(probe.id, axis, transform_tag)
                for axis in self.scorers
            ):
                bump(skip=len(self.scorers))
                continue

            transformed = transform.apply(probe)
            if transformed.is_multi_turn and transformed.messages:
                messages = list(transformed.messages)
                user_msg = transformed.transformed_prompt
            else:
                user_msg = transformed.transformed_prompt
                messages = []
                if self.config.system_prompt:
                    messages.append({"role": "system", "content": self.config.system_prompt})
                messages.append({"role": "user", "content": user_msg})
            try:
                response = self.provider.chat(
                    messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
            except Exception as e:  # noqa: BLE001
                # Upstream provider rejected the prompt (Azure content
                # filter, OpenRouter 400, network failure). Treat as a
                # synthesised refusal so the scorers can still run — it
                # IS a refusal, just one we didn't generate ourselves.
                response = f"[INFERENCE_BLOCKED: {type(e).__name__}: {str(e)[:240]}]"
            bump(infer=1)
            guardrail_tag = self._capture_guardrail_actions()

            self._score_axes_parallel(
                probe_id=probe.id,
                user_msg=user_msg,
                response=response,
                transform_tag=transform_tag,
                multi_turn=False,
                writer=writer,
                bump=bump,
                axis_workers=axis_workers,
                guardrail_action=guardrail_tag,
                model_under_test=model_under_test,
            )

    # -- multi-turn --------------------------------------------------------
    def _audit_conversation(
        self,
        probe: ConversationProbe,
        writer: AuditWriter,
        bump,
        axis_workers: int,
        model_under_test: str | None = None,
    ) -> None:
        # All-axes-already-done short-circuit (no inference at all)
        if all(
            writer.is_done(probe.id, axis, None, multi_turn=True)
            for axis in self.scorers
        ):
            bump(skip=len(self.scorers))
            return

        # Walk the conversation ONCE, then score every axis against the
        # same final response in parallel. This is the cost-efficient path:
        # multi-turn Modal calls are the most expensive thing in the audit,
        # so we don't re-walk N times per axis.
        messages: list[dict] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        turn_responses: list[tuple[str, str]] = []
        all_guard_actions: list[str] = []
        for turn in probe.turns:
            messages.append({"role": "user", "content": turn.user})
            try:
                reply = self.provider.chat(
                    messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
            except Exception as e:  # noqa: BLE001
                reply = f"[INFERENCE_BLOCKED: {type(e).__name__}: {str(e)[:240]}]"
            messages.append({"role": "assistant", "content": reply})
            turn_responses.append((turn.user, reply))
            tag = self._capture_guardrail_actions()
            if tag:
                all_guard_actions.append(tag)
        bump(infer=len(probe.turns))
        mt_guardrail_tag = ";".join(all_guard_actions) if all_guard_actions else None

        full_prompt = "\n\n".join(t.user for t in probe.turns)
        final_response = turn_responses[-1][1] if turn_responses else ""
        transcript = "\n\n".join(
            f"USER: {u}\nASSISTANT: {r}" for u, r in turn_responses
        )

        def _score_one_axis(item):
            axis, scorer = item
            if writer.is_done(probe.id, axis, None, multi_turn=True):
                bump(skip=1)
                return
            try:
                score = scorer.score(prompt=full_prompt, response=final_response)
            except Exception as e:  # noqa: BLE001
                from .types import ScoreResult
                score = ScoreResult(
                    value=1.0,
                    rationale=f"scorer_error: {type(e).__name__}: {str(e)[:240]}",
                    scorer_name=getattr(scorer, "name", "unknown"),
                    confidence=0.0,
                )
            score.tier = self.tier_mapping.map(score.value)
            if mt_guardrail_tag and not score.guardrail_action:
                score.guardrail_action = mt_guardrail_tag
            writer.write_score(
                instance_id=probe.id,
                axis=axis,
                prompt=transcript,
                response=final_response,
                score=score,
                multi_turn=True,
                model_under_test=model_under_test,
            )
            bump(score=1, rows=1)

        if axis_workers > 1 and len(self.scorers) > 1:
            with ThreadPoolExecutor(max_workers=axis_workers) as ex:
                list(ex.map(_score_one_axis, self.scorers.items()))
        else:
            for item in self.scorers.items():
                _score_one_axis(item)

    # -- shared: score every axis on one (probe, transform) tuple ----------
    def _score_axes_parallel(
        self,
        probe_id: str,
        user_msg: str,
        response: str,
        transform_tag: str | None,
        multi_turn: bool,
        writer: AuditWriter,
        bump,
        axis_workers: int,
        guardrail_action: str | None = None,
        model_under_test: str | None = None,
    ) -> None:
        def _score_one(item):
            axis, scorer = item
            if writer.is_done(probe_id, axis, transform_tag, multi_turn=multi_turn):
                bump(skip=1)
                return
            try:
                score = scorer.score(prompt=user_msg, response=response)
            except Exception as e:  # noqa: BLE001
                # One axis failing must not crash the audit. Record a
                # neutral score with the exception so the row count is
                # consistent and forensics are possible.
                from .types import ScoreResult
                score = ScoreResult(
                    value=1.0,
                    rationale=f"scorer_error: {type(e).__name__}: {str(e)[:240]}",
                    scorer_name=getattr(scorer, "name", "unknown"),
                    confidence=0.0,
                )
            score.tier = self.tier_mapping.map(score.value)
            if guardrail_action and not score.guardrail_action:
                score.guardrail_action = guardrail_action
            writer.write_score(
                instance_id=probe_id,
                axis=axis,
                prompt=user_msg,
                response=response,
                score=score,
                attack_transform=transform_tag,
                multi_turn=multi_turn,
                model_under_test=model_under_test,
            )
            bump(score=1, rows=1)

        if axis_workers > 1 and len(self.scorers) > 1:
            with ThreadPoolExecutor(max_workers=axis_workers) as ex:
                list(ex.map(_score_one, self.scorers.items()))
        else:
            for item in self.scorers.items():
                _score_one(item)


__all__ = ["Evaluator"]
