"""Argus Web View — multi-tab Streamlit dashboard over audit stores.

Auto-discovers every ``examples/out/*.db``, treats them as a virtual index
of runs, models, axes, probes. Inspired by LangSmith's experiment-view
patterns: KPI strip, scorecards, distribution histograms, side-by-side
response diff, scatter-of-scores, auto-detected run badges.

Run with:
    streamlit run argus_dashboard.py

Read-only. Writes nothing back to the audit stores.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Argus Audit Viewer",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

DEFAULT_AUDIT_DIR = Path("examples/out")

# Per-axis interpretive thresholds for color coding (Tier-1/2/3 on 0-2 scale).
AXIS_GREEN = 1.5
AXIS_YELLOW = 1.0

# Approximate OpenRouter / OpenAI pricing for telemetry. Order of magnitude
# only — model_under_test cost isn't tracked per token, so we estimate from
# latency_ms × tier multiplier.
COST_PER_SEC = {
    "anthropic/claude-sonnet-4":      0.00033,  # ~$0.005 per ~15s call
    "openai/gpt-4o":                  0.00020,
    "openai/gpt-4o-mini":             0.00003,
    "google/gemini-2.0-flash-001":    0.00002,
    "meta-llama/llama-guard-4-12b":   0.00005,
    "meta-llama/llama-prompt-guard-2-86m": 0.000005,
}


# ----------------------------------------------------------------------------
# Data discovery + queries (cached)
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def discover_dbs(audit_dir: str) -> list[Path]:
    p = Path(audit_dir)
    return sorted(p.glob("*.db")) if p.exists() else []


@st.cache_data(show_spinner="Loading audit data…")
def load_all_rows(db_paths: tuple[str, ...]) -> pd.DataFrame:
    """Read every audit row into one DataFrame (cached on db_paths tuple)."""
    frames = []
    for p in db_paths:
        try:
            with sqlite3.connect(p) as conn:
                df = pd.read_sql_query(
                    """SELECT
                          run_id, instance_id, axis, scorer_name, scorer_model,
                          model_under_test, value, tier, rationale, confidence,
                          latency_ms, cost_usd, prompt, response, fallback_fired,
                          aggregator, disagreement, guardrail_action,
                          attack_transform, multi_turn, timestamp, extra
                       FROM audit_rows""",
                    conn,
                )
            df["__db"] = Path(p).name
            frames.append(df)
        except Exception as e:  # noqa: BLE001
            st.warning(f"Could not read {p}: {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["model_under_test"] = out["model_under_test"].fillna("(unknown)")
    out["attack_transform"] = out["attack_transform"].fillna("identity")
    out["guardrail_action"] = out["guardrail_action"].fillna("")
    out["tier"] = out["tier"].astype(str)
    return out


def estimate_cost_usd(row) -> float:
    """Per-row cost estimate via scorer_model × latency."""
    model = (row.get("scorer_model") or "").lower()
    if not model or pd.isna(row.get("latency_ms")):
        return 0.0
    per_sec = COST_PER_SEC.get(model, 0.00005)
    return float(row["latency_ms"]) / 1000.0 * per_sec


def detect_run_badge(run_df: pd.DataFrame) -> str:
    """Auto-label a run based on which guardrail actions fired."""
    actions = set(run_df["guardrail_action"].unique()) - {""}
    if not actions:
        return "baseline"
    tags = []
    if any("pattern_block" in a for a in actions):
        tags.append("pattern")
    if any("embedding_block" in a for a in actions):
        tags.append("embedding")
    if any("classifier_block" in a for a in actions):
        tags.append("classifier")
    return f"guarded({'+'.join(tags)})" if tags else "guarded"


def runs_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for (run_id, mut), group in df.groupby(["run_id", "model_under_test"]):
        avg = float(group["value"].mean())
        tier1_pct = (group["tier"] == "1").mean() * 100.0
        tier3_pct = (group["tier"] == "3").mean() * 100.0
        p50_latency = float(group["latency_ms"].median() / 1000.0) if "latency_ms" in group else 0.0
        est_cost = float(group.apply(estimate_cost_usd, axis=1).sum())
        rows.append({
            "run_id": run_id,
            "model_under_test": mut,
            "badge": detect_run_badge(group),
            "rows": len(group),
            "avg_score": avg,
            "tier1_pct": tier1_pct,
            "tier3_pct": tier3_pct,
            "p50_latency_s": p50_latency,
            "est_cost_usd": est_cost,
            "first_ts": group["timestamp"].min(),
        })
    return pd.DataFrame(rows).sort_values("first_ts", ascending=False).reset_index(drop=True)


def axis_means_for_run(df: pd.DataFrame, run_id: str) -> pd.Series:
    return df[df["run_id"] == run_id].groupby("axis")["value"].mean().sort_index()


def tier_distribution(df: pd.DataFrame, run_id: str) -> pd.Series:
    return df[df["run_id"] == run_id]["tier"].value_counts().sort_index()


def guardrail_action_counts(df: pd.DataFrame, run_id: str) -> pd.Series:
    s = df[df["run_id"] == run_id]["guardrail_action"]
    return s[s != ""].value_counts()


def worst_rows(df: pd.DataFrame, run_id: str, threshold: float = 1.0, limit: int = 50):
    return (
        df[(df["run_id"] == run_id) & (df["value"] < threshold)]
        .sort_values("value")
        .head(limit)
        [["instance_id", "axis", "value", "tier", "scorer_name",
          "attack_transform", "multi_turn", "rationale"]]
    )


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------

def axis_color(value: float) -> str:
    if value >= AXIS_GREEN:
        return "#16a34a"  # green
    if value >= AXIS_YELLOW:
        return "#ca8a04"  # amber
    return "#dc2626"      # red


def scorecard(label: str, value: float, max_value: float = 2.0):
    color = axis_color(value)
    fill_pct = (value / max_value) * 100.0
    st.markdown(
        f"""
        <div style="border:1px solid #e5e7eb;border-radius:8px;padding:12px;">
          <div style="font-size:0.8em;color:#6b7280;text-transform:uppercase;letter-spacing:0.04em;">{label}</div>
          <div style="font-size:1.6em;font-weight:600;color:{color};margin-top:4px;">{value:.3f}</div>
          <div style="background:#f3f4f6;border-radius:4px;height:6px;margin-top:8px;overflow:hidden;">
            <div style="background:{color};height:6px;width:{fill_pct:.1f}%;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_badge(badge: str) -> str:
    color_map = {
        "baseline": "#94a3b8",
        "guarded(pattern)":            "#3b82f6",
        "guarded(pattern+embedding)":  "#6366f1",
        "guarded(pattern+classifier)": "#0ea5e9",
        "guarded(pattern+embedding+classifier)": "#8b5cf6",
        "guarded(embedding)":          "#6366f1",
        "guarded(classifier)":         "#0ea5e9",
        "guarded":                     "#0ea5e9",
    }
    color = color_map.get(badge, "#64748b")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:10px;font-size:0.85em;font-weight:500;">{badge}</span>'
    )


# ----------------------------------------------------------------------------
# Sidebar — data source + filters
# ----------------------------------------------------------------------------

st.sidebar.title("🔭 Argus")
st.sidebar.caption("AI vendor risk audit viewer")

audit_dir = st.sidebar.text_input("Audit directory", value=str(DEFAULT_AUDIT_DIR))
dbs = discover_dbs(audit_dir)
if not dbs:
    st.warning(
        f"No audit `.db` files found in `{audit_dir}`. "
        "Run `examples/kitchen_sink.py` first, or point the sidebar at a "
        "directory containing audit databases."
    )
    st.stop()

selected_db_names = st.sidebar.multiselect(
    "Audit stores",
    options=[p.name for p in dbs],
    default=[p.name for p in dbs],
)
selected_dbs = tuple(str(p) for p in dbs if p.name in selected_db_names)
if not selected_dbs:
    st.info("Select at least one audit store in the sidebar.")
    st.stop()

df = load_all_rows(selected_dbs)
if df.empty:
    st.info("No audit rows in the selected stores.")
    st.stop()

# Cross-tab filters
all_models = sorted(df["model_under_test"].unique().tolist())
model_filter = st.sidebar.multiselect("Model", options=all_models, default=all_models)
all_axes = sorted(df["axis"].unique().tolist())
axis_filter = st.sidebar.multiselect("Axis", options=all_axes, default=all_axes)
all_transforms = sorted(df["attack_transform"].unique().tolist())
transform_filter = st.sidebar.multiselect("Transform", options=all_transforms, default=all_transforms)

df = df[
    df["model_under_test"].isin(model_filter)
    & df["axis"].isin(axis_filter)
    & df["attack_transform"].isin(transform_filter)
]

# ----------------------------------------------------------------------------
# Top KPI strip — always visible
# ----------------------------------------------------------------------------

n_runs = df["run_id"].nunique()
n_models = df["model_under_test"].nunique()
n_rows = len(df)
avg_score = float(df["value"].mean())
tier1_pct = (df["tier"] == "1").mean() * 100.0
total_cost = float(df.apply(estimate_cost_usd, axis=1).sum())

st.markdown("### Audit overview")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Runs", n_runs)
k2.metric("Models", n_models)
k3.metric("Rows", f"{n_rows:,}")
k4.metric("Avg score", f"{avg_score:.3f}")
k5.metric("Tier-1 share", f"{tier1_pct:.1f}%")
k6.metric("Est. cost", f"${total_cost:.2f}")

st.divider()

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------

tab_runs, tab_detail, tab_compare, tab_board, tab_probe = st.tabs([
    "📋 Runs", "🔍 Run Detail", "⚖️ Compare", "🏆 Leaderboard", "🎯 Probe",
])


# ===========================================================================
# Tab 1 — Runs
# ===========================================================================

with tab_runs:
    st.subheader("All audit runs")
    summary = runs_summary(df)
    if summary.empty:
        st.info("No runs.")
    else:
        # Render badge column as styled HTML — Streamlit dataframes don't
        # render HTML, so we use an explicit Markdown table for the
        # at-a-glance view + a regular dataframe below for sortability.
        st.dataframe(
            summary[["model_under_test", "badge", "run_id", "rows",
                     "avg_score", "tier1_pct", "tier3_pct",
                     "p50_latency_s", "est_cost_usd", "first_ts"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "model_under_test": "Model",
                "badge": "Guard config",
                "run_id": "Run ID",
                "rows": st.column_config.NumberColumn("Rows", format="%d"),
                "avg_score": st.column_config.NumberColumn(
                    "Avg score", format="%.3f",
                    help="Mean across all axes. Higher = lower liability.",
                ),
                "tier1_pct": st.column_config.ProgressColumn(
                    "Tier-1 %", format="%.1f%%", min_value=0, max_value=100,
                ),
                "tier3_pct": st.column_config.NumberColumn(
                    "Tier-3 %", format="%.1f%%",
                ),
                "p50_latency_s": st.column_config.NumberColumn(
                    "p50 latency (s)", format="%.2f",
                ),
                "est_cost_usd": st.column_config.NumberColumn(
                    "Est. cost (USD)", format="$%.2f",
                ),
                "first_ts": "First row",
            },
        )


# ===========================================================================
# Tab 2 — Run Detail
# ===========================================================================

with tab_detail:
    st.subheader("Run detail")
    summary = runs_summary(df)
    if summary.empty:
        st.info("No runs.")
    else:
        summary["__label"] = (
            summary["model_under_test"] + " · " + summary["badge"]
            + " · " + summary["run_id"] + " (" + summary["rows"].astype(str) + " rows)"
        )
        chosen = st.selectbox("Pick a run", options=summary["__label"].tolist(),
                              key="detail_pick")
        chosen_row = summary[summary["__label"] == chosen].iloc[0]
        run_id = chosen_row["run_id"]
        run_df = df[df["run_id"] == run_id]

        # Header strip with badge
        h1, h2, h3, h4 = st.columns([2, 2, 1, 1])
        h1.markdown(f"**Model:** `{chosen_row['model_under_test']}`")
        h2.markdown(f"**Config:** {run_badge(chosen_row['badge'])}", unsafe_allow_html=True)
        h3.metric("Rows", int(chosen_row["rows"]))
        h4.metric("Cost", f"${chosen_row['est_cost_usd']:.2f}")

        st.markdown("---")

        # Axis scorecards (colored)
        st.markdown("**Axis means** (0 = high liability, 2 = low liability)")
        means = axis_means_for_run(run_df, run_id)
        cols = st.columns(len(means))
        for col, (axis, val) in zip(cols, means.items()):
            with col:
                scorecard(axis, val)

        st.markdown("---")

        # Score distribution per axis (LangSmith-style)
        st.markdown("**Score distributions**")
        for axis in sorted(run_df["axis"].unique()):
            axis_df = run_df[run_df["axis"] == axis]
            counts = axis_df["value"].round(1).value_counts().sort_index()
            with st.expander(f"`{axis}` — {len(axis_df)} rows, mean {axis_df['value'].mean():.3f}", expanded=False):
                col_dist, col_tier = st.columns(2)
                col_dist.bar_chart(counts.rename("count"))
                tier_counts = axis_df["tier"].value_counts().sort_index()
                col_tier.bar_chart(tier_counts.rename("count"))

        st.markdown("---")

        col_actions, col_tiers = st.columns(2)
        with col_actions:
            st.markdown("**Pre-flight guardrail actions**")
            actions = guardrail_action_counts(run_df, run_id)
            if actions.empty:
                st.info("No guardrail actions recorded.")
            else:
                st.dataframe(
                    actions.reset_index().rename(
                        columns={actions.name: "Count", "index": "Action"},
                    ),
                    use_container_width=True, hide_index=True,
                )
        with col_tiers:
            st.markdown("**Tier distribution**")
            tiers = tier_distribution(run_df, run_id)
            st.dataframe(
                tiers.reset_index().rename(
                    columns={tiers.name: "Count", "index": "Tier"},
                ),
                use_container_width=True, hide_index=True,
            )

        st.markdown("---")

        st.markdown("**Worst-scoring probes**")
        threshold = st.slider("Show rows below score", 0.0, 2.0, 1.0, 0.1, key="worst_thresh")
        st.dataframe(
            worst_rows(run_df, run_id, threshold=threshold, limit=100),
            use_container_width=True, hide_index=True,
        )


# ===========================================================================
# Tab 3 — Compare (LangSmith experiment-view inspired)
# ===========================================================================

with tab_compare:
    st.subheader("Compare two runs")
    st.caption(
        "Side-by-side lift table + score-vs-score scatter + tier-flip diff. "
        "Pick two runs of the same model for the cleanest A/B (e.g. baseline "
        "vs guarded). Cross-model comparisons show axis-level lift only."
    )
    summary = runs_summary(df)
    if len(summary) < 2:
        st.info("Need at least two runs.")
    else:
        summary["__label"] = (
            summary["model_under_test"] + " · " + summary["badge"]
            + " · " + summary["run_id"] + " (" + summary["rows"].astype(str) + " rows)"
        )
        c_left, c_right = st.columns(2)
        with c_left:
            label_a = st.selectbox("Baseline (before)",
                                   options=summary["__label"].tolist(),
                                   index=min(1, len(summary) - 1),
                                   key="cmp_a")
            row_a = summary[summary["__label"] == label_a].iloc[0]
            run_a = row_a["run_id"]
            st.markdown(run_badge(row_a["badge"]), unsafe_allow_html=True)
        with c_right:
            label_b = st.selectbox("Treatment (after)",
                                   options=summary["__label"].tolist(),
                                   index=0, key="cmp_b")
            row_b = summary[summary["__label"] == label_b].iloc[0]
            run_b = row_b["run_id"]
            st.markdown(run_badge(row_b["badge"]), unsafe_allow_html=True)

        means_a = axis_means_for_run(df, run_a)
        means_b = axis_means_for_run(df, run_b)
        axes_all = sorted(set(means_a.index) | set(means_b.index))
        lift = []
        for ax in axes_all:
            b = means_a.get(ax, 0.0); a = means_b.get(ax, 0.0)
            delta = a - b
            pct = (delta / b * 100.0) if b else 0.0
            lift.append({"axis": ax, "before": b, "after": a, "delta": delta, "pct_change": pct})
        lift_df = pd.DataFrame(lift)

        st.markdown("**Axis lift**")
        st.dataframe(
            lift_df,
            use_container_width=True, hide_index=True,
            column_config={
                "before": st.column_config.NumberColumn("Before", format="%.3f"),
                "after": st.column_config.NumberColumn("After", format="%.3f"),
                "delta": st.column_config.NumberColumn("Δ", format="%+.3f"),
                "pct_change": st.column_config.NumberColumn("% change", format="%+.1f%%"),
            },
        )

        chart_data = pd.DataFrame({
            "before": means_a.reindex(axes_all, fill_value=0.0),
            "after":  means_b.reindex(axes_all, fill_value=0.0),
        }, index=axes_all)
        st.bar_chart(chart_data)

        st.markdown("---")

        # Score-vs-score scatter (LangSmith experiment view)
        if row_a["model_under_test"] == row_b["model_under_test"]:
            st.markdown("**Score-vs-score per probe**")
            df_a = df[df["run_id"] == run_a][["instance_id", "axis", "attack_transform", "value"]]
            df_b = df[df["run_id"] == run_b][["instance_id", "axis", "attack_transform", "value"]]
            merged = df_a.merge(df_b, on=["instance_id", "axis", "attack_transform"],
                                suffixes=("_before", "_after"))
            if not merged.empty:
                scatter = pd.DataFrame({
                    "before": merged["value_before"],
                    "after":  merged["value_after"],
                })
                st.scatter_chart(scatter, x="before", y="after")

            # Tier flips
            flips = merged[
                (merged["value_before"] != merged["value_after"])
                & (merged["value_after"] < 1.0) != (merged["value_before"] < 1.0)
            ]
            if not flips.empty:
                up = flips[flips["value_after"] > flips["value_before"]]
                down = flips[flips["value_after"] < flips["value_before"]]
                cu, cd = st.columns(2)
                cu.metric("Improvements (▲)", len(up))
                cd.metric("Regressions (▼)", len(down))
                st.markdown("**Probe-level flips**")
                st.dataframe(flips, use_container_width=True, hide_index=True)
        else:
            st.caption(
                "Per-probe scatter and tier-flip diff are only meaningful "
                "within the same model. Showing axis-level lift only."
            )

        st.markdown("---")

        # Side-by-side response diff for a specific probe
        st.markdown("**Side-by-side response viewer**")
        common_probes = (
            set(df[df["run_id"] == run_a]["instance_id"])
            & set(df[df["run_id"] == run_b]["instance_id"])
        )
        if common_probes:
            chosen_probe = st.selectbox("Probe id", options=sorted(common_probes),
                                        key="cmp_probe")
            common_transforms = (
                set(df[(df["run_id"] == run_a) & (df["instance_id"] == chosen_probe)]["attack_transform"])
                & set(df[(df["run_id"] == run_b) & (df["instance_id"] == chosen_probe)]["attack_transform"])
            )
            chosen_transform = st.selectbox("Transform", options=sorted(common_transforms),
                                            key="cmp_transform")
            row_a_resp = df[(df["run_id"] == run_a) & (df["instance_id"] == chosen_probe)
                            & (df["attack_transform"] == chosen_transform)].iloc[0]
            row_b_resp = df[(df["run_id"] == run_b) & (df["instance_id"] == chosen_probe)
                            & (df["attack_transform"] == chosen_transform)].iloc[0]
            st.code(row_a_resp["prompt"][:500], language=None)
            col_la, col_lb = st.columns(2)
            with col_la:
                st.markdown(f"**Before** — score={row_a_resp['value']:.2f} · tier={row_a_resp['tier']}")
                if row_a_resp["guardrail_action"]:
                    st.caption(f"Guard: `{row_a_resp['guardrail_action']}`")
                st.markdown(row_a_resp["response"][:2000])
            with col_lb:
                st.markdown(f"**After** — score={row_b_resp['value']:.2f} · tier={row_b_resp['tier']}")
                if row_b_resp["guardrail_action"]:
                    st.caption(f"Guard: `{row_b_resp['guardrail_action']}`")
                st.markdown(row_b_resp["response"][:2000])
        else:
            st.caption("No probes appear in both runs (different probe sets).")


# ===========================================================================
# Tab 4 — Leaderboard
# ===========================================================================

with tab_board:
    st.subheader("Cross-model leaderboard")
    leaderboard = df.groupby(["model_under_test", "axis"])["value"].mean().unstack()
    if leaderboard.empty:
        st.info("Not enough data.")
    else:
        st.markdown("**Per-axis mean** (green = low liability, red = high)")
        st.dataframe(
            leaderboard.style
                .format("{:.3f}")
                .background_gradient(cmap="RdYlGn", axis=None, vmin=0.0, vmax=2.0),
            use_container_width=True,
        )

        st.markdown("---")

        # Tier-1 share + cost efficiency
        col_t1, col_cost = st.columns(2)
        with col_t1:
            st.markdown("**Tier-1 share by model**")
            tier1 = (
                df.groupby("model_under_test")["tier"]
                  .apply(lambda s: (s == "1").mean() * 100.0)
                  .sort_values(ascending=False)
            )
            st.bar_chart(tier1)
        with col_cost:
            st.markdown("**Estimated cost by model**")
            cost_by_model = (
                df.assign(cost=df.apply(estimate_cost_usd, axis=1))
                  .groupby("model_under_test")["cost"].sum()
                  .sort_values(ascending=False)
            )
            st.bar_chart(cost_by_model)

        # Efficiency: tier-1 share per dollar
        if cost_by_model.sum() > 0:
            st.markdown("**Cost efficiency** — Tier-1 share per dollar spent")
            efficiency = (tier1 / cost_by_model.replace(0, np.nan)).dropna().sort_values(ascending=False)
            st.dataframe(
                efficiency.reset_index().rename(
                    columns={efficiency.name: "Tier-1 % per $", "model_under_test": "Model"},
                ),
                use_container_width=True, hide_index=True,
            )


# ===========================================================================
# Tab 5 — Probe drill-down
# ===========================================================================

with tab_probe:
    st.subheader("Probe drill-down")
    st.caption(
        "Pick a probe id to see every model's response + per-axis verdict + "
        "Legion-mode per-judge breakdown."
    )
    probe_ids = sorted(df["instance_id"].unique())
    chosen_probe = st.selectbox("Probe id", options=probe_ids, key="probe_pick")
    sub = df[df["instance_id"] == chosen_probe]
    if sub.empty:
        st.info("No data for this probe.")
    else:
        prompt_text = sub["prompt"].iloc[0]
        st.markdown("**Prompt**")
        st.code(prompt_text[:1000], language=None)

        # Per-model summary table
        st.markdown("**Per-model verdicts** (Tier 1 = pass, Tier 3 = fail)")
        per_model = (
            sub.groupby("model_under_test")
               .agg(
                   avg_score=("value", "mean"),
                   worst_axis=("value", "min"),
                   transforms=("attack_transform", lambda s: ", ".join(sorted(s.unique()))),
                   guardrail_actions=("guardrail_action", lambda s: ", ".join(sorted([a for a in s.unique() if a])) or "(none)"),
               )
               .reset_index()
               .sort_values("avg_score", ascending=False)
        )
        st.dataframe(per_model, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Expanders per (model, transform) showing the response + axis breakdown
        st.markdown("**Responses**")
        groups = sub.groupby(["model_under_test", "attack_transform"], dropna=False)
        for (mut, transform), gdf in groups:
            header = f"**{mut}** · transform=`{transform}`"
            if any(gdf["guardrail_action"] != ""):
                header += " · ⛔️ guarded"
            with st.expander(header, expanded=False):
                response_text = gdf["response"].iloc[0]
                guard = next((a for a in gdf["guardrail_action"] if a), "")
                if guard:
                    st.caption(f"Guardrail action: `{guard}`")
                st.markdown(response_text[:2000])

                axis_table = gdf[["axis", "value", "tier", "scorer_name",
                                  "disagreement", "rationale"]].copy()
                st.dataframe(axis_table, use_container_width=True, hide_index=True)

                # Legion-mode per-judge breakdown
                for _, row in gdf.iterrows():
                    extra_str = row.get("extra", "")
                    if not extra_str:
                        continue
                    try:
                        extra = json.loads(extra_str) if isinstance(extra_str, str) else {}
                    except Exception:
                        continue
                    if "per_judge" in extra:
                        st.caption(f"Per-judge breakdown — axis `{row['axis']}`")
                        per_judge = pd.DataFrame.from_dict(
                            extra["per_judge"], orient="index",
                        )
                        st.dataframe(per_judge, use_container_width=True)


# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.caption(
    "Read-only viewer. Add data: "
    "`uv run python examples/kitchen_sink.py` with any provider/model combo."
)
