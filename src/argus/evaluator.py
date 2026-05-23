"""Evaluator — runs a full audit from an EvalConfig.

Given a config + a list of `LiabilityProbe`s (or `ConversationProbe`s for
multi-turn), the Evaluator:
  1. Builds provider, scorers, transforms, tier mapping from the config
  2. For each probe × transform, generates the assistant response, scores it
     on every configured axis, writes one AuditRow per (probe, transform, axis)
  3. Returns an AuditReport bound to the run
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .conversation import ConversationProbe, score_conversation
from .eval_config import EvalConfig
from .probes import LiabilityProbe
from .scorers.base import Scorer
from .storage import AuditIndex, AuditReport, AuditWriter
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
    ) -> AuditReport:
        """Run every probe through every transform on every axis. Return report.

        `compare_to` is an optional baseline run id; if provided, the
        returned report's `to_underwriting_memo` will render lift vs that run.
        """
        writer = AuditWriter(self.config.audit_log_path)
        run_id = writer.run_id

        for probe in probes:
            if isinstance(probe, ConversationProbe):
                self._audit_conversation(probe, writer)
            else:
                self._audit_single(probe, writer)

        index = AuditIndex(self.config.audit_db_path)
        index.build_from_jsonl(self.config.audit_log_path)
        return AuditReport(index, run_id)

    # -- single-turn -------------------------------------------------------
    def _audit_single(self, probe: LiabilityProbe, writer: AuditWriter) -> None:
        for transform in self.transforms:
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

            for axis, scorer in self.scorers.items():
                score = scorer.score(prompt=user_msg, response=response)
                score.tier = self.tier_mapping.map(score.value)
                writer.write_score(
                    instance_id=probe.id,
                    axis=axis,
                    prompt=user_msg,
                    response=response,
                    score=score,
                    attack_transform=transform.name if transform.name != "identity" else None,
                )

    # -- multi-turn --------------------------------------------------------
    def _audit_conversation(self, probe: ConversationProbe, writer: AuditWriter) -> None:
        for axis, scorer in self.scorers.items():
            result = score_conversation(
                probe=probe,
                provider=self.provider,
                scorer=scorer,
                system=self.config.system_prompt,
            )
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


__all__ = ["Evaluator"]
