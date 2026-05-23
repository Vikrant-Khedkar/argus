"""AuditReport — high-level analytics over one or two audit runs.

Wraps `AuditIndex`. Two killer methods:
  - `compare(other_run_id)` — per-axis lift between two runs (e.g. with vs
    without guardrails). The whole point of GuardrailedProvider.
  - `to_underwriting_memo()` — markdown summary for the broker, with axis
    means, tier distribution, worst probes, and (if compared) lift.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .index import AuditIndex


@dataclass
class AxisLift:
    axis: str
    before: float
    after: float
    delta: float
    pct_change: float


class AuditReport:
    """Renderable view over one audit run, optionally compared against another."""

    def __init__(self, index: AuditIndex, run_id: str):
        self.index = index
        self.run_id = run_id

    # -- summaries ---------------------------------------------------------
    def axis_means(self) -> dict[str, float]:
        return self.index.axis_means(self.run_id)

    def tier_distribution(self) -> dict[str, int]:
        return self.index.tier_distribution(self.run_id)

    def worst_rows(self, threshold: float = 1.0, limit: int = 10):
        return self.index.failures(self.run_id, threshold=threshold, limit=limit)

    def row_count(self) -> int:
        return self.index.row_count(self.run_id)

    def models(self) -> list[str]:
        """Distinct model_under_test values seen in this run."""
        with self.index._conn() as c:
            cur = c.execute(
                "SELECT DISTINCT model_under_test FROM audit_rows "
                "WHERE run_id = ? AND model_under_test IS NOT NULL",
                (self.run_id,),
            )
            return [r["model_under_test"] for r in cur]

    # -- comparison --------------------------------------------------------
    def compare(self, other_run_id: str) -> dict[str, AxisLift]:
        """Per-axis lift: `self` (after) minus `other_run_id` (before).

        Positive delta = improvement (higher score = lower liability).
        """
        before = self.index.axis_means(other_run_id)
        after = self.axis_means()
        axes = set(before) | set(after)
        result: dict[str, AxisLift] = {}
        for axis in sorted(axes):
            b = before.get(axis, 0.0)
            a = after.get(axis, 0.0)
            delta = a - b
            pct = (delta / b * 100.0) if b else float("inf") if delta else 0.0
            result[axis] = AxisLift(
                axis=axis, before=b, after=a, delta=delta, pct_change=pct,
            )
        return result

    # -- rendering ---------------------------------------------------------
    def to_underwriting_memo(
        self,
        baseline_run_id: str | None = None,
        title: str = "AI Vendor Risk Audit",
        vendor_name: str = "(vendor)",
    ) -> str:
        """Markdown memo for the broker / underwriter.

        If `baseline_run_id` is provided, includes a guardrail-lift section.
        """
        lines: list[str] = []
        lines.append(f"# {title} — {vendor_name}")
        lines.append("")
        lines.append(f"**Run ID:** `{self.run_id}`")
        lines.append(f"**Generated:** {datetime.utcnow().isoformat()}Z")
        lines.append(f"**Total scoring events:** {self.row_count()}")
        models_seen = self.models()
        if models_seen:
            lines.append(f"**Model under test:** {', '.join(f'`{m}`' for m in models_seen)}")
        lines.append("")

        # Axis means
        lines.append("## Axis means (0 = high liability, 2 = low liability)")
        lines.append("")
        means = self.axis_means()
        if means:
            lines.append("| Axis | Mean score |")
            lines.append("| --- | --- |")
            for axis, mean in sorted(means.items()):
                lines.append(f"| {axis} | {mean:.3f} |")
        else:
            lines.append("_No data._")
        lines.append("")

        # Tier distribution
        lines.append("## Tier distribution")
        lines.append("")
        tiers = self.tier_distribution()
        if tiers:
            lines.append("| Tier | Count |")
            lines.append("| --- | --- |")
            for tier, n in sorted(tiers.items(), key=lambda kv: str(kv[0])):
                lines.append(f"| {tier} | {n} |")
        else:
            lines.append("_No tier mapping applied._")
        lines.append("")

        # Guardrail lift
        if baseline_run_id:
            lines.append(f"## Guardrail lift vs `{baseline_run_id}`")
            lines.append("")
            lifts = self.compare(baseline_run_id)
            lines.append("| Axis | Before | After | Δ | % change |")
            lines.append("| --- | --- | --- | --- | --- |")
            for axis, lift in lifts.items():
                arrow = "▲" if lift.delta > 0 else ("▼" if lift.delta < 0 else "—")
                pct = "n/a" if lift.pct_change == float("inf") else f"{lift.pct_change:+.1f}%"
                lines.append(
                    f"| {axis} | {lift.before:.3f} | {lift.after:.3f} | "
                    f"{arrow} {lift.delta:+.3f} | {pct} |"
                )
            lines.append("")

        # Worst rows
        lines.append("## Worst-scoring probes (sample)")
        lines.append("")
        worst = self.worst_rows(threshold=1.0, limit=8)
        if not worst:
            lines.append("_No failures below threshold._")
        for row in worst:
            tag = []
            if row.get("attack_transform"):
                tag.append(f"transform={row['attack_transform']}")
            if row.get("multi_turn"):
                tag.append("multi-turn")
            tag_s = f" _({', '.join(tag)})_" if tag else ""
            lines.append(
                f"- **{row['axis']}** · `{row['instance_id']}` · "
                f"score **{row['value']:.2f}** by `{row['scorer_name']}`{tag_s}"
            )
            if row.get("rationale"):
                lines.append(f"  - rationale: {row['rationale']}")
        lines.append("")

        return "\n".join(lines)


__all__ = ["AuditReport", "AxisLift"]
