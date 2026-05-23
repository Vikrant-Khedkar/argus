"""Argus Web View — Streamlit dashboard over audit stores.

Run with:
    uv run streamlit run argus_dashboard.py

Auto-discovers every ``examples/out/*.db``. Read-only.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Argus",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_AUDIT_DIR = Path("examples/out")

# Per-axis interpretive thresholds (Tier-1/2/3 on 0-2 scale)
AXIS_GREEN, AXIS_YELLOW = 1.5, 1.0

# Order-of-magnitude pricing for cost estimation
COST_PER_SEC = {
    "anthropic/claude-sonnet-4":      0.00033,
    "openai/gpt-4o":                  0.00020,
    "openai/gpt-4o-mini":             0.00003,
    "google/gemini-2.0-flash-001":    0.00002,
    "meta-llama/llama-guard-4-12b":   0.00005,
    "meta-llama/llama-prompt-guard-2-86m": 0.000005,
}

# Distinct, accessible colors for per-model identity. Cycled deterministically
# from a hash of model_under_test, so the same model gets the same color
# everywhere in the UI.
MODEL_PALETTE = [
    "#3b82f6",  # blue
    "#10b981",  # emerald
    "#f59e0b",  # amber
    "#8b5cf6",  # violet
    "#ec4899",  # pink
    "#06b6d4",  # cyan
    "#ef4444",  # red
    "#84cc16",  # lime
]


def color_for_model(model: str) -> str:
    h = int(hashlib.md5(model.encode("utf-8")).hexdigest(), 16)
    return MODEL_PALETTE[h % len(MODEL_PALETTE)]


def short_model_name(model: str) -> str:
    """`openrouter:openai/gpt-4o-mini` -> `gpt-4o-mini`."""
    if not model or model == "(unknown)":
        return "?"
    # Strip provider prefix
    if ":" in model:
        model = model.split(":", 1)[1]
    # Strip vendor prefix
    if "/" in model:
        model = model.split("/", 1)[1]
    return model


def short_run_id(run_id: str) -> str:
    """Last 8 chars of the run id for disambiguation."""
    return run_id[-8:] if run_id else "?"


def humanize_ts(ts: str) -> str:
    """ISO timestamp -> 'just now' / '5m ago' / 'May 23 21:22'."""
    if not ts:
        return "?"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return ts[:16]
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt
    sec = delta.total_seconds()
    if sec < 60:
        return "just now"
    if sec < 3600:
        return f"{int(sec / 60)}m ago"
    if sec < 86400:
        return f"{int(sec / 3600)}h ago"
    if sec < 86400 * 7:
        return f"{int(sec / 86400)}d ago"
    return dt.strftime("%b %d %H:%M")


# ----------------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def discover_dbs(audit_dir: str) -> list[Path]:
    p = Path(audit_dir)
    return sorted(p.glob("*.db")) if p.exists() else []


@st.cache_data(show_spinner="Loading audit data…")
def load_all_rows(db_paths: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for p in db_paths:
        try:
            with sqlite3.connect(p) as conn:
                df = pd.read_sql_query(
                    """SELECT run_id, instance_id, axis, scorer_name, scorer_model,
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
    out["scorer_model"] = out["scorer_model"].fillna("")
    out["latency_ms"] = pd.to_numeric(out["latency_ms"], errors="coerce").fillna(0.0)
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0)
    out["disagreement"] = pd.to_numeric(out["disagreement"], errors="coerce").fillna(0.0)
    out["tier"] = out["tier"].astype(str)
    return out


def estimate_cost_usd(row) -> float:
    model_raw = row.get("scorer_model")
    if model_raw is None or (isinstance(model_raw, float) and pd.isna(model_raw)):
        return 0.0
    latency = row.get("latency_ms")
    if latency is None or (isinstance(latency, float) and pd.isna(latency)):
        return 0.0
    model = str(model_raw).lower()
    if not model:
        return 0.0
    per_sec = COST_PER_SEC.get(model, 0.00005)
    return float(latency) / 1000.0 * per_sec


def detect_run_badge(run_df: pd.DataFrame) -> str:
    actions = set(run_df["guardrail_action"].unique()) - {""}
    if not actions:
        return "baseline"
    tags = []
    if any("pattern_block" in a for a in actions):
        tags.append("pattern")
    if any("embedding_block" in a for a in actions):
        tags.append("embed")
    if any("classifier_block" in a for a in actions):
        tags.append("classifier")
    return f"guarded({'+'.join(tags)})" if tags else "guarded"


def runs_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for (run_id, mut), group in df.groupby(["run_id", "model_under_test"]):
        rows.append({
            "run_id": run_id,
            "model_under_test": mut,
            "model_short": short_model_name(mut),
            "short_id": short_run_id(run_id),
            "badge": detect_run_badge(group),
            "rows": len(group),
            "avg_score": float(group["value"].mean()),
            "tier1_pct": (group["tier"] == "1").mean() * 100.0,
            "tier3_pct": (group["tier"] == "3").mean() * 100.0,
            "p50_latency_s": float(group["latency_ms"].median() / 1000.0),
            "est_cost_usd": float(group.apply(estimate_cost_usd, axis=1).sum()),
            "first_ts": group["timestamp"].min(),
            "when": humanize_ts(group["timestamp"].min()),
        })
    return pd.DataFrame(rows).sort_values("first_ts", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------

def axis_color(value: float) -> str:
    if value >= AXIS_GREEN:
        return "#16a34a"
    if value >= AXIS_YELLOW:
        return "#ca8a04"
    return "#dc2626"


BADGE_COLORS = {
    "baseline": "#64748b",
    "guarded": "#0ea5e9",
}


def badge_color(badge: str) -> str:
    if badge == "baseline":
        return BADGE_COLORS["baseline"]
    # Any guarded variant uses the same blue family with intensity by tier count
    n_tiers = badge.count("+") + 1 if "guarded(" in badge else 1
    intensity = {1: "#0ea5e9", 2: "#0284c7", 3: "#0369a1"}.get(n_tiers, "#0ea5e9")
    return intensity


def model_chip(model: str, small: bool = False) -> str:
    color = color_for_model(model)
    name = short_model_name(model)
    size = "0.75em" if small else "0.85em"
    pad = "1px 6px" if small else "3px 10px"
    return (
        f'<span style="background:{color};color:white;padding:{pad};'
        f'border-radius:10px;font-size:{size};font-weight:600;'
        f'white-space:nowrap;">{name}</span>'
    )


def badge_chip(badge: str, small: bool = False) -> str:
    color = badge_color(badge)
    label = badge
    size = "0.75em" if small else "0.85em"
    pad = "1px 6px" if small else "2px 8px"
    return (
        f'<span style="background:{color};color:white;padding:{pad};'
        f'border-radius:10px;font-size:{size};font-weight:500;'
        f'white-space:nowrap;">{label}</span>'
    )


def run_label_chips(row) -> str:
    """Compact HTML: <model chip> <badge chip> <when> · <short_id>"""
    return (
        f'{model_chip(row["model_under_test"], small=True)} '
        f'{badge_chip(row["badge"], small=True)} '
        f'<span style="color:#64748b;font-size:0.85em;">{row["when"]} · '
        f'<code style="background:#f1f5f9;padding:1px 4px;border-radius:3px;'
        f'font-size:0.85em;">{row["short_id"]}</code></span>'
    )


def scorecard(label: str, value: float, max_value: float = 2.0):
    color = axis_color(value)
    fill_pct = (value / max_value) * 100.0
    st.markdown(
        f"""
        <div style="border:1px solid #e5e7eb;border-radius:8px;padding:12px;">
          <div style="font-size:0.75em;color:#6b7280;text-transform:uppercase;
                      letter-spacing:0.04em;">{label}</div>
          <div style="font-size:1.6em;font-weight:600;color:{color};margin-top:4px;">{value:.3f}</div>
          <div style="background:#f3f4f6;border-radius:4px;height:6px;margin-top:8px;
                      overflow:hidden;">
            <div style="background:{color};height:6px;width:{fill_pct:.1f}%;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Sidebar — minimal: just data source
# ----------------------------------------------------------------------------

st.sidebar.title("🔭 Argus")
audit_dir = st.sidebar.text_input("Audit directory", value=str(DEFAULT_AUDIT_DIR))
dbs = discover_dbs(audit_dir)
if not dbs:
    st.warning(
        f"No audit `.db` files found in `{audit_dir}`. "
        "Run `examples/kitchen_sink.py` first."
    )
    st.stop()
selected_db_names = st.sidebar.multiselect(
    "Audit stores",
    options=[p.name for p in dbs],
    default=[p.name for p in dbs],
    label_visibility="visible",
)
selected_dbs = tuple(str(p) for p in dbs if p.name in selected_db_names)
if not selected_dbs:
    st.info("Select at least one audit store in the sidebar.")
    st.stop()

df = load_all_rows(selected_dbs)
if df.empty:
    st.info("No audit rows in the selected stores.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption("Read-only viewer over JSONL + SQLite audit stores. "
                   "Add data: `uv run python examples/kitchen_sink.py`.")


# ----------------------------------------------------------------------------
# Top KPI strip — small + neutral
# ----------------------------------------------------------------------------

n_runs = df["run_id"].nunique()
n_models = df["model_under_test"].nunique()
n_rows = len(df)
avg_score = float(df["value"].mean())
tier1_pct = (df["tier"] == "1").mean() * 100.0
total_cost = float(df.apply(estimate_cost_usd, axis=1).sum())

st.markdown("## 🔭 Argus Audit Viewer")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Runs", n_runs)
k2.metric("Models", n_models)
k3.metric("Rows", f"{n_rows:,}")
k4.metric("Avg score", f"{avg_score:.3f}")
k5.metric("Tier-1 %", f"{tier1_pct:.1f}%")
k6.metric("Est. cost", f"${total_cost:.2f}")

st.divider()

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------

tab_runs, tab_detail, tab_compare, tab_board, tab_probe = st.tabs([
    "📋 Runs", "🔍 Detail", "⚖️ Compare", "🏆 Leaderboard", "🎯 Probe",
])


# ===========================================================================
# Tab 1 — Runs (grouped by model, card layout)
# ===========================================================================

with tab_runs:
    st.markdown("#### All audit runs")
    summary = runs_summary(df)
    if summary.empty:
        st.info("No runs.")
    else:
        # Group by model so users see "all Qwen runs" / "all GPT-4o-mini runs"
        # together — much easier to spot which-was-which than a chronological
        # blob.
        for model_id in summary["model_under_test"].unique():
            model_runs = summary[summary["model_under_test"] == model_id]
            color = color_for_model(model_id)
            name = short_model_name(model_id)
            n_runs_for_model = len(model_runs)
            total_rows = int(model_runs["rows"].sum())
            cost = float(model_runs["est_cost_usd"].sum())
            st.markdown(
                f"""
                <div style="margin-top:18px;margin-bottom:6px;">
                  <span style="background:{color};color:white;padding:4px 12px;
                               border-radius:12px;font-weight:600;font-size:0.95em;">{name}</span>
                  <span style="color:#64748b;margin-left:10px;font-size:0.85em;">
                    {n_runs_for_model} run{'s' if n_runs_for_model != 1 else ''} ·
                    {total_rows:,} rows · est. ${cost:.2f}
                  </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            display_df = model_runs[[
                "badge", "when", "short_id", "rows", "avg_score",
                "tier1_pct", "tier3_pct", "p50_latency_s", "est_cost_usd",
            ]].copy()
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "badge": "Config",
                    "when": "When",
                    "short_id": "ID",
                    "rows": st.column_config.NumberColumn("Rows", format="%d"),
                    "avg_score": st.column_config.NumberColumn("Avg score", format="%.3f"),
                    "tier1_pct": st.column_config.ProgressColumn(
                        "Tier-1 %", format="%.1f%%", min_value=0, max_value=100,
                    ),
                    "tier3_pct": st.column_config.NumberColumn("Tier-3 %", format="%.1f%%"),
                    "p50_latency_s": st.column_config.NumberColumn("p50 latency (s)", format="%.2f"),
                    "est_cost_usd": st.column_config.NumberColumn("Cost", format="$%.2f"),
                },
            )


# ===========================================================================
# Tab 2 — Run Detail
# ===========================================================================

with tab_detail:
    summary = runs_summary(df)
    if summary.empty:
        st.info("No runs.")
    else:
        # Picker with rich label
        options = list(range(len(summary)))
        chosen_idx = st.selectbox(
            "Pick a run",
            options=options,
            format_func=lambda i: (
                f"{summary.iloc[i]['model_short']} · "
                f"{summary.iloc[i]['badge']} · "
                f"{summary.iloc[i]['when']} · "
                f"{summary.iloc[i]['short_id']}"
            ),
            key="detail_pick",
        )
        row = summary.iloc[chosen_idx]
        run_id = row["run_id"]
        run_df = df[df["run_id"] == run_id]

        # Header strip with rich chips
        st.markdown(
            f'<div style="margin-top:8px;">'
            f'{model_chip(row["model_under_test"])} &nbsp; '
            f'{badge_chip(row["badge"])} &nbsp; '
            f'<span style="color:#64748b;font-size:0.9em;">'
            f'{row["when"]} · <code style="background:#f1f5f9;padding:2px 6px;border-radius:3px;">{row["short_id"]}</code>'
            f'</span></div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", int(row["rows"]))
        m2.metric("Avg score", f"{row['avg_score']:.3f}")
        m3.metric("Tier-1 %", f"{row['tier1_pct']:.1f}%")
        m4.metric("Est. cost", f"${row['est_cost_usd']:.2f}")

        st.markdown("---")
        st.markdown("**Axis scorecards**")
        means = run_df.groupby("axis")["value"].mean().sort_index()
        cols = st.columns(len(means))
        for col, (axis, val) in zip(cols, means.items()):
            with col:
                scorecard(axis, val)

        st.markdown("---")
        col_actions, col_tiers = st.columns(2)
        with col_actions:
            st.markdown("**Pre-flight guardrail actions**")
            actions = run_df["guardrail_action"]
            actions = actions[actions != ""].value_counts()
            if actions.empty:
                st.info("None — provider not wrapped, or no patterns matched.")
            else:
                st.dataframe(
                    actions.reset_index().rename(
                        columns={"index": "Action", actions.name: "Count"},
                    ),
                    use_container_width=True, hide_index=True,
                )
        with col_tiers:
            st.markdown("**Tier distribution**")
            tiers = run_df["tier"].value_counts().sort_index()
            st.dataframe(
                tiers.reset_index().rename(
                    columns={"index": "Tier", tiers.name: "Count"},
                ),
                use_container_width=True, hide_index=True,
            )

        st.markdown("---")
        with st.expander("Score distributions by axis"):
            for axis in sorted(run_df["axis"].unique()):
                axis_df = run_df[run_df["axis"] == axis]
                counts = axis_df["value"].round(1).value_counts().sort_index()
                st.caption(f"`{axis}` — {len(axis_df)} rows, mean {axis_df['value'].mean():.3f}")
                st.bar_chart(counts.rename("count"))

        st.markdown("---")
        st.markdown("**Worst-scoring probes**")
        threshold = st.slider("Show rows below score", 0.0, 2.0, 1.0, 0.1, key="worst_thresh")
        worst = (
            run_df[run_df["value"] < threshold]
            .sort_values("value")
            .head(100)
            [["instance_id", "axis", "value", "tier", "scorer_name",
              "attack_transform", "multi_turn", "rationale"]]
        )
        st.dataframe(worst, use_container_width=True, hide_index=True)


# ===========================================================================
# Tab 3 — Compare
# ===========================================================================

with tab_compare:
    summary = runs_summary(df)
    if len(summary) < 2:
        st.info("Need at least two runs to compare.")
    else:
        options = list(range(len(summary)))
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("**Baseline (before)**")
            a_idx = st.selectbox(
                "—",
                options=options,
                index=min(1, len(summary) - 1),
                format_func=lambda i: (
                    f"{summary.iloc[i]['model_short']} · "
                    f"{summary.iloc[i]['badge']} · "
                    f"{summary.iloc[i]['when']} · {summary.iloc[i]['short_id']}"
                ),
                key="cmp_a",
                label_visibility="collapsed",
            )
            row_a = summary.iloc[a_idx]
            st.markdown(
                f'<div>{model_chip(row_a["model_under_test"])} '
                f'{badge_chip(row_a["badge"])}</div>',
                unsafe_allow_html=True,
            )
        with c_right:
            st.markdown("**Treatment (after)**")
            b_idx = st.selectbox(
                "—",
                options=options,
                index=0,
                format_func=lambda i: (
                    f"{summary.iloc[i]['model_short']} · "
                    f"{summary.iloc[i]['badge']} · "
                    f"{summary.iloc[i]['when']} · {summary.iloc[i]['short_id']}"
                ),
                key="cmp_b",
                label_visibility="collapsed",
            )
            row_b = summary.iloc[b_idx]
            st.markdown(
                f'<div>{model_chip(row_b["model_under_test"])} '
                f'{badge_chip(row_b["badge"])}</div>',
                unsafe_allow_html=True,
            )

        run_a, run_b = row_a["run_id"], row_b["run_id"]
        means_a = df[df["run_id"] == run_a].groupby("axis")["value"].mean()
        means_b = df[df["run_id"] == run_b].groupby("axis")["value"].mean()
        axes_all = sorted(set(means_a.index) | set(means_b.index))
        lift = pd.DataFrame([{
            "axis": ax,
            "before": means_a.get(ax, 0.0),
            "after":  means_b.get(ax, 0.0),
            "delta":  means_b.get(ax, 0.0) - means_a.get(ax, 0.0),
            "pct_change": ((means_b.get(ax, 0.0) - means_a.get(ax, 0.0))
                           / means_a.get(ax, 0.0) * 100.0)
                          if means_a.get(ax, 0.0) else 0.0,
        } for ax in axes_all])

        st.markdown("---")
        st.markdown("#### Axis lift")
        st.dataframe(
            lift, use_container_width=True, hide_index=True,
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

        if row_a["model_under_test"] == row_b["model_under_test"]:
            st.markdown("---")
            st.markdown("#### Score-vs-score per probe")
            df_a = df[df["run_id"] == run_a][["instance_id", "axis", "attack_transform", "value"]]
            df_b = df[df["run_id"] == run_b][["instance_id", "axis", "attack_transform", "value"]]
            merged = df_a.merge(df_b, on=["instance_id", "axis", "attack_transform"],
                                suffixes=("_before", "_after"))
            if not merged.empty:
                st.scatter_chart(pd.DataFrame({
                    "before": merged["value_before"],
                    "after":  merged["value_after"],
                }), x="before", y="after")

            flips = merged[merged["value_before"] != merged["value_after"]]
            if not flips.empty:
                up = flips[flips["value_after"] > flips["value_before"]]
                down = flips[flips["value_after"] < flips["value_before"]]
                cu, cd = st.columns(2)
                cu.metric("Improvements ▲", len(up))
                cd.metric("Regressions ▼", len(down))

            st.markdown("---")
            st.markdown("#### Side-by-side response viewer")
            common = sorted(set(df_a["instance_id"]) & set(df_b["instance_id"]))
            if common:
                probe = st.selectbox("Probe", common, key="cmp_probe")
                ts = sorted(set(df[(df["run_id"] == run_a) & (df["instance_id"] == probe)]
                                ["attack_transform"]))
                tform = st.selectbox("Transform", ts, key="cmp_tform")
                ra = df[(df["run_id"] == run_a) & (df["instance_id"] == probe)
                        & (df["attack_transform"] == tform)].iloc[0]
                rb = df[(df["run_id"] == run_b) & (df["instance_id"] == probe)
                        & (df["attack_transform"] == tform)].iloc[0]
                st.code(ra["prompt"][:500], language=None)
                lcol, rcol = st.columns(2)
                with lcol:
                    st.caption(f"**Before** · score={ra['value']:.2f} · tier={ra['tier']}")
                    if ra["guardrail_action"]:
                        st.caption(f"⛔ `{ra['guardrail_action']}`")
                    st.markdown(ra["response"][:2000])
                with rcol:
                    st.caption(f"**After** · score={rb['value']:.2f} · tier={rb['tier']}")
                    if rb["guardrail_action"]:
                        st.caption(f"⛔ `{rb['guardrail_action']}`")
                    st.markdown(rb["response"][:2000])
        else:
            st.caption("Per-probe scatter and side-by-side only meaningful when "
                       "both runs target the same model.")


# ===========================================================================
# Tab 4 — Leaderboard
# ===========================================================================

with tab_board:
    st.markdown("#### Cross-model leaderboard")
    leaderboard = df.groupby(["model_under_test", "axis"])["value"].mean().unstack()
    if leaderboard.empty:
        st.info("Not enough data.")
    else:
        # Re-index with friendly model names for display
        leaderboard.index = [short_model_name(m) for m in leaderboard.index]
        st.dataframe(
            leaderboard.style.format("{:.3f}")
                .background_gradient(cmap="RdYlGn", axis=None, vmin=0.0, vmax=2.0),
            use_container_width=True,
        )

        st.markdown("---")
        col_t1, col_cost = st.columns(2)
        with col_t1:
            st.markdown("**Tier-1 % by model**")
            tier1 = (
                df.groupby("model_under_test")["tier"]
                  .apply(lambda s: (s == "1").mean() * 100.0)
                  .sort_values(ascending=False)
            )
            tier1.index = [short_model_name(m) for m in tier1.index]
            st.bar_chart(tier1)
        with col_cost:
            st.markdown("**Est. cost by model**")
            cost = (
                df.assign(c=df.apply(estimate_cost_usd, axis=1))
                  .groupby("model_under_test")["c"].sum()
                  .sort_values(ascending=False)
            )
            cost.index = [short_model_name(m) for m in cost.index]
            st.bar_chart(cost)

        if cost.sum() > 0:
            st.markdown("**Cost efficiency** — Tier-1 % per dollar")
            eff = (tier1 / cost.replace(0, np.nan)).dropna().sort_values(ascending=False)
            st.dataframe(
                eff.reset_index().rename(
                    columns={eff.name: "Tier-1 % per $", "model_under_test": "Model"},
                ),
                use_container_width=True, hide_index=True,
            )


# ===========================================================================
# Tab 5 — Probe drill-down
# ===========================================================================

with tab_probe:
    st.markdown("#### Probe drill-down")
    probe_ids = sorted(df["instance_id"].unique())
    probe = st.selectbox("Probe id", probe_ids, key="probe_pick")
    sub = df[df["instance_id"] == probe]
    if sub.empty:
        st.info("No data for this probe.")
    else:
        st.markdown("**Prompt**")
        st.code(sub["prompt"].iloc[0][:1000], language=None)

        st.markdown("**Per-model summary**")
        per_model = (
            sub.groupby("model_under_test")
            .agg(
                avg_score=("value", "mean"),
                worst_axis_score=("value", "min"),
                transforms=("attack_transform", lambda s: ", ".join(sorted(s.unique()))),
                guardrail_actions=("guardrail_action",
                                    lambda s: ", ".join(sorted([a for a in s.unique() if a])) or "(none)"),
            )
            .reset_index()
            .sort_values("avg_score", ascending=False)
        )
        per_model["Model"] = per_model["model_under_test"].apply(short_model_name)
        st.dataframe(
            per_model[["Model", "avg_score", "worst_axis_score", "transforms", "guardrail_actions"]],
            use_container_width=True, hide_index=True,
            column_config={
                "avg_score": st.column_config.NumberColumn("Avg score", format="%.3f"),
                "worst_axis_score": st.column_config.NumberColumn("Worst axis", format="%.3f"),
            },
        )

        st.markdown("---")
        st.markdown("**Responses**")
        for (mut, transform), gdf in sub.groupby(["model_under_test", "attack_transform"]):
            chip = model_chip(mut, small=True)
            guard = next((a for a in gdf["guardrail_action"] if a), "")
            guard_chip = f' <span style="color:#dc2626;font-size:0.8em;">⛔ {guard}</span>' if guard else ""
            header = f'{chip} <code style="background:#f1f5f9;padding:1px 6px;border-radius:3px;">{transform}</code>{guard_chip}'
            st.markdown(header, unsafe_allow_html=True)
            with st.expander("Show response + axis verdicts", expanded=False):
                st.markdown(gdf["response"].iloc[0][:2000])
                axis_table = gdf[["axis", "value", "tier", "scorer_name",
                                  "disagreement", "rationale"]].copy()
                st.dataframe(axis_table, use_container_width=True, hide_index=True)
                for _, row in gdf.iterrows():
                    extra = row.get("extra", "")
                    try:
                        extra = json.loads(extra) if isinstance(extra, str) else {}
                    except Exception:
                        continue
                    if extra and "per_judge" in extra:
                        st.caption(f"Per-judge breakdown — axis `{row['axis']}`")
                        st.dataframe(
                            pd.DataFrame.from_dict(extra["per_judge"], orient="index"),
                            use_container_width=True,
                        )
