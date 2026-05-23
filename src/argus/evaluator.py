"""Evaluator — runs a full audit from an EvalConfig.

Given a config + a list of `LiabilityProbe`s (or `ConversationProbe`s for
multi-turn), the Evaluator:
  1. Builds provider, scorers, transforms, tier mapping from the config
  2. For each probe × transform, generates the assistant response, scores it
     on every configured axis, writes one AuditRow per (probe, transform, axis)
  3. Returns an AuditReport bound to the run

Resumable: pass `resume_run_id=...` (or `resume_run_id="latest"`) to continue
an interrupted run. Rows already in the JSONL for that run_id are skipped,
including the LLM-inference + scoring calls behind them.
"""

from __future__ import annotations

import time
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
        """
        if resume_run_id == "latest":
            resume_run_id = latest_run_id(self.config.audit_log_path)
            if resume_run_id is None:
                print("[resume] no existing log, starting a fresh run")
        writer = AuditWriter(self.config.audit_log_path, run_id=resume_run_id)
        run_id = writer.run_id
        if resume_run_id and writer.done_count():
            print(f"[resume] {run_id} — {writer.done_count()} rows already complete, skipping those")

        # Live counters surfaced in the tqdm postfix
        stats = {"infer": 0, "score": 0, "skip": 0, "rows": 0, "t0": time.time()}
        bar = self._make_bar(len(probes), progress)

        try:
            for probe in probes:
                if isinstance(probe, ConversationProbe):
                    self._audit_conversation(probe, writer, stats)
                else:
                    self._audit_single(probe, writer, stats)
                if bar is not None:
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
        finally:
            if bar is not None:
                bar.close()

        print(
            f"  done · rows={stats['rows']} infer={stats['infer']} "
            f"score={stats['score']} skip={stats['skip']} "
            f"in {time.time() - stats['t0']:.0f}s"
        )

        index = AuditIndex(self.config.audit_db_path)
        index.build_from_jsonl(self.config.audit_log_path)
        return AuditReport(index, run_id)

    @staticmethod
    def _make_bar(total: int, progress: bool):
        if not progress or not _HAS_TQDM:
            return None
        return tqdm(total=total, unit="probe", desc="audit", dynamic_ncols=True)

    # -- single-turn -------------------------------------------------------
    def _audit_single(
        self, probe: LiabilityProbe, writer: AuditWriter, stats: dict,
    ) -> None:
        for transform in self.transforms:
            transform_tag = transform.name if transform.name != "identity" else None
            # If every axis is already done for this (probe, transform), skip
            # the inference call entirely.
            if all(
                writer.is_done(probe.id, axis, transform_tag)
                for axis in self.scorers
            ):
                stats["skip"] += len(self.scorers)
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
            response = self.provider.chat(messages)
            stats["infer"] += 1

            for axis, scorer in self.scorers.items():
                if writer.is_done(probe.id, axis, transform_tag):
                    stats["skip"] += 1
                    continue
                score = scorer.score(prompt=user_msg, response=response)
                score.tier = self.tier_mapping.map(score.value)
                writer.write_score(
                    instance_id=probe.id,
                    axis=axis,
                    prompt=user_msg,
                    response=response,
                    score=score,
                    attack_transform=transform_tag,
                )
                stats["score"] += 1
                stats["rows"] += 1

    # -- multi-turn --------------------------------------------------------
    def _audit_conversation(
        self, probe: ConversationProbe, writer: AuditWriter, stats: dict,
    ) -> None:
        for axis, scorer in self.scorers.items():
            if writer.is_done(probe.id, axis, None, multi_turn=True):
                stats["skip"] += 1
                continue
            result = score_conversation(
                probe=probe,
                provider=self.provider,
                scorer=scorer,
                system=self.config.system_prompt,
            )
            stats["infer"] += len(probe.turns)
            stats["score"] += 1
            score = result.final_score
            score.tier = self.tier_mapping.map(score.value)
            transcript = "\n\n".join(
                f"USER: {t.user}\nASSISTANT: {t.response}" for t in result.turns
            )
            writer.write_score(
                instance_id=probe.id,
                axis=axis,
                prompt=transcript,
                response=result.turns[-1].response if result.turns else "",
                score=score,
                multi_turn=True,
            )
            stats["rows"] += 1


__all__ = ["Evaluator"]
