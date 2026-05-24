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


def _has_audit_rows_table(db_path: str) -> bool:
    """Returns True only if the .db is a valid Argus audit store."""
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='audit_rows'"
            ).fetchone()
        return row is not None
    except Exception:
        return False


@st.cache_data(show_spinner="Loading audit data…")
def load_all_rows(db_paths: tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for p in db_paths:
        if not _has_audit_rows_table(p):
            # Skip silently — not an Argus audit store. Don't scare the user
            # with a warning about an unrelated .db they have sitting around.
            continue
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
# UI helpers — pill design system
# ----------------------------------------------------------------------------

def axis_color(value: float) -> str:
    if value >= AXIS_GREEN:
        return "#16a34a"
    if value >= AXIS_YELLOW:
        return "#ca8a04"
    return "#dc2626"


def badge_color(badge: str) -> str:
    if badge == "baseline":
        return "#64748b"
    n_tiers = badge.count("+") + 1 if "guarded(" in badge else 1
    return {1: "#0ea5e9", 2: "#0284c7", 3: "#0369a1"}.get(n_tiers, "#0ea5e9")


def short_axis_name(axis: str) -> str:
    """`safety_liability` -> `safety`. Easier to scan in dense tables."""
    return axis.replace("_liability", "").replace("_", " ")


def _pill(text: str, bg: str, fg: str = "white",
          size: str = "0.78em", weight: int = 500) -> str:
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 9px;'
        f'border-radius:10px;font-size:{size};font-weight:{weight};'
        f'white-space:nowrap;display:inline-block;line-height:1.5;">{text}</span>'
    )


def model_chip(model: str, small: bool = False) -> str:
    name = short_model_name(model)
    size = "0.72em" if small else "0.82em"
    return _pill(name, color_for_model(model), size=size, weight=600)


def badge_chip(badge: str, small: bool = False) -> str:
    size = "0.72em" if small else "0.82em"
    return _pill(badge, badge_color(badge), size=size)


def tier_pill(tier: str | int) -> str:
    t = str(tier)
    color = {"1": "#16a34a", "2": "#ca8a04", "3": "#dc2626"}.get(t, "#94a3b8")
    return _pill(f"T{t}", color, size="0.72em")


def score_pill(value: float) -> str:
    return _pill(f"{value:.2f}", axis_color(value), size="0.78em", weight=600)


def delta_pill(delta: float) -> str:
    if delta > 0.005:
        return _pill(f"▲ +{delta:.3f}", "#16a34a", size="0.78em")
    if delta < -0.005:
        return _pill(f"▼ {delta:.3f}", "#dc2626", size="0.78em")
    return _pill(f"— {delta:+.3f}", "#94a3b8", size="0.78em")


def axis_pill(axis: str) -> str:
    return _pill(
        short_axis_name(axis), "#e0e7ff", fg="#3730a3",
        size="0.75em", weight=500,
    )


def transform_pill(transform: str) -> str:
    if transform == "identity":
        return _pill("identity", "#f1f5f9", fg="#475569", size="0.72em", weight=400)
    return _pill(transform, "#fef3c7", fg="#92400e", size="0.72em", weight=500)


def code_pill(text: str) -> str:
    return (
        f'<code style="background:#f1f5f9;padding:2px 7px;border-radius:6px;'
        f'font-size:0.78em;color:#475569;">{text}</code>'
    )


def when_text(when: str) -> str:
    return f'<span style="color:#64748b;font-size:0.82em;">{when}</span>'


def run_hero(row, prefix: str = "") -> str:
    """Big-ish header strip for the currently-picked run."""
    return (
        f'<div style="margin:6px 0 14px 0;line-height:2;">'
        f'{(prefix + " ") if prefix else ""}'
        f'{model_chip(row["model_under_test"])} &nbsp; '
        f'{badge_chip(row["badge"])} &nbsp; '
        f'{when_text(row["when"])} &nbsp; '
        f'{code_pill(row["short_id"])}'
        f'</div>'
    )


def scorecard(label: str, value: float, max_value: float = 2.0):
    color = axis_color(value)
    fill_pct = (value / max_value) * 100.0
    short = short_axis_name(label)
    st.markdown(
        f"""
        <div style="border:1px solid #e5e7eb;border-radius:10px;padding:14px;
                    background:#fafafa;">
          <div style="font-size:0.72em;color:#6b7280;text-transform:uppercase;
                      letter-spacing:0.06em;font-weight:600;">{short}</div>
          <div style="font-size:1.7em;font-weight:700;color:{color};
                      margin-top:6px;line-height:1;">{value:.3f}</div>
          <div style="background:#e5e7eb;border-radius:4px;height:5px;
                      margin-top:10px;overflow:hidden;">
            <div style="background:{color};height:5px;width:{fill_pct:.1f}%;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_lift_table(lift: pd.DataFrame) -> str:
    """Custom HTML lift table — colored deltas, axis pills."""
    rows_html = []
    for _, r in lift.iterrows():
        delta_html = delta_pill(r["delta"])
        pct = r["pct_change"]
        if abs(pct) < float("inf"):
            pct_str = f"{pct:+.1f}%"
        else:
            pct_str = "n/a"
        pct_color = "#16a34a" if pct > 0.5 else "#dc2626" if pct < -0.5 else "#64748b"
        rows_html.append(
            f"<tr>"
            f"<td style='padding:8px 10px;'>{axis_pill(r['axis'])}</td>"
            f"<td style='padding:8px 10px;text-align:right;color:#475569;'>{r['before']:.3f}</td>"
            f"<td style='padding:8px 10px;text-align:right;font-weight:600;'>{r['after']:.3f}</td>"
            f"<td style='padding:8px 10px;text-align:center;'>{delta_html}</td>"
            f"<td style='padding:8px 10px;text-align:right;color:{pct_color};font-weight:500;'>{pct_str}</td>"
            f"</tr>"
        )
    return (
        '<table style="width:100%;border-collapse:collapse;'
        'border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;'
        'font-size:0.9em;">'
        '<thead style="background:#f8fafc;">'
        '<tr>'
        '<th style="padding:10px;text-align:left;font-weight:600;color:#475569;">Axis</th>'
        '<th style="padding:10px;text-align:right;font-weight:600;color:#475569;">Before</th>'
        '<th style="padding:10px;text-align:right;font-weight:600;color:#475569;">After</th>'
        '<th style="padding:10px;text-align:center;font-weight:600;color:#475569;">Δ</th>'
        '<th style="padding:10px;text-align:right;font-weight:600;color:#475569;">% change</th>'
        '</tr></thead><tbody>'
        + "".join(rows_html)
        + '</tbody></table>'
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

tab_runs, tab_detail, tab_compare, tab_board, tab_probe, tab_demo = st.tabs([
    "📋 Runs", "🔍 Detail", "⚖️ Compare", "🏆 Leaderboard", "🎯 Probe", "🎭 Demo",
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

        st.markdown(run_hero(row), unsafe_allow_html=True)

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
            .copy()
        )
        if worst.empty:
            st.success(f"No rows below {threshold:.1f} — clean run.")
        else:
            worst["axis_short"] = worst["axis"].apply(short_axis_name)
            worst["transform"] = worst["attack_transform"]
            worst_display = worst[[
                "instance_id", "axis_short", "value", "tier",
                "scorer_name", "transform", "multi_turn", "rationale",
            ]].rename(columns={"axis_short": "axis"})
            st.dataframe(
                worst_display,
                use_container_width=True, hide_index=True,
                column_config={
                    "instance_id": "Probe",
                    "axis": "Axis",
                    "value": st.column_config.NumberColumn("Score", format="%.2f"),
                    "tier": "Tier",
                    "scorer_name": "Scorer",
                    "transform": "Transform",
                    "multi_turn": st.column_config.CheckboxColumn("Multi-turn"),
                    "rationale": st.column_config.TextColumn(
                        "Rationale", width="large"),
                },
            )


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
            st.markdown(run_hero(row_a), unsafe_allow_html=True)
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
            st.markdown(run_hero(row_b), unsafe_allow_html=True)

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
        st.markdown(render_lift_table(lift), unsafe_allow_html=True)
        st.markdown("")  # spacer
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
                for col, side, r in [(lcol, "Before", ra), (rcol, "After", rb)]:
                    with col:
                        st.markdown(
                            f"<div style='margin-bottom:6px;'>"
                            f"<strong>{side}</strong> &nbsp; "
                            f"{score_pill(r['value'])} &nbsp; "
                            f"{tier_pill(r['tier'])}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        if r["guardrail_action"]:
                            st.markdown(
                                f"<div style='margin-bottom:8px;'>"
                                f"⛔ {code_pill(r['guardrail_action'])}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        st.markdown(r["response"][:2000])
        else:
            st.caption("Per-probe scatter and side-by-side only meaningful when "
                       "both runs target the same model.")


# ===========================================================================
# Tab 4 — Leaderboard
# ===========================================================================

with tab_board:
    st.markdown("#### Cross-model leaderboard")

    # Filter selector — the whole point. Without this, baseline rows and
    # guarded rows for the same model average together into a meaningless
    # blended number.
    summary = runs_summary(df)
    view = st.radio(
        "View",
        ["Baseline (raw model)", "Guarded (best per model)", "Lift (Δ)", "Both side-by-side"],
        horizontal=True, key="lb_view",
    )

    def pick_runs(badge_predicate) -> set[str]:
        """Return the LATEST run_id per model matching the predicate."""
        filtered = summary[summary["badge"].apply(badge_predicate)]
        # latest per model
        latest = filtered.sort_values("first_ts", ascending=False).drop_duplicates("model_under_test")
        return set(latest["run_id"])

    baseline_runs = pick_runs(lambda b: b == "baseline")
    guarded_runs = pick_runs(lambda b: b != "baseline")

    def axis_means_for(run_ids: set[str]) -> pd.DataFrame:
        if not run_ids:
            return pd.DataFrame()
        sub = df[df["run_id"].isin(run_ids)]
        out = sub.groupby(["model_under_test", "axis"])["value"].mean().unstack()
        out.index = [short_model_name(m) for m in out.index]
        out.columns = [short_axis_name(a) for a in out.columns]
        return out

    def render_heatmap(table: pd.DataFrame, caption: str = ""):
        if table.empty:
            st.info("No runs in this category yet.")
            return
        if caption:
            st.caption(caption)
        st.dataframe(
            table.style.format("{:.3f}")
                .background_gradient(cmap="RdYlGn", axis=None, vmin=1.0, vmax=2.0),
            use_container_width=True,
        )

    base_table = axis_means_for(baseline_runs)
    guard_table = axis_means_for(guarded_runs)

    if view == "Baseline (raw model)":
        st.markdown("**Raw model capability — no guards.** Latest baseline run per model.")
        st.caption("Color scale: red < 1.0 · amber 1.0–1.5 · green 1.5–2.0")
        render_heatmap(base_table)

    elif view == "Guarded (best per model)":
        st.markdown("**With Argus guards.** Latest guarded run per model.")
        st.caption("Color scale: red < 1.0 · amber 1.0–1.5 · green 1.5–2.0")
        render_heatmap(guard_table)

    elif view == "Lift (Δ)":
        st.markdown("**Lift from guards** — guarded mean minus baseline mean, per (model, axis).")
        common_models = set(base_table.index) & set(guard_table.index)
        if not common_models:
            st.info("Need at least one model with both a baseline AND a guarded run.")
        else:
            common_axes = set(base_table.columns) & set(guard_table.columns)
            lift_table = (
                guard_table.loc[list(common_models), list(common_axes)]
                - base_table.loc[list(common_models), list(common_axes)]
            )
            st.caption("Green = guards helped, red = guards regressed (over-refusal cost).")
            st.dataframe(
                lift_table.style.format("{:+.3f}")
                    .background_gradient(cmap="RdYlGn", axis=None, vmin=-0.3, vmax=0.3),
                use_container_width=True,
            )

    else:  # Both side-by-side
        st.caption("Color scale: red < 1.0 · amber 1.0–1.5 · green 1.5–2.0")
        col_b, col_g = st.columns(2)
        with col_b:
            st.markdown("**Baseline** (raw model)")
            render_heatmap(base_table)
        with col_g:
            st.markdown("**Guarded** (with Argus)")
            render_heatmap(guard_table)

    st.markdown("---")

    # Tier-1 % and cost charts also use the selector
    if view in ("Baseline (raw model)", "Both side-by-side"):
        runs_for_charts = baseline_runs
        chart_label = "Baseline"
    elif view == "Guarded (best per model)":
        runs_for_charts = guarded_runs
        chart_label = "Guarded"
    else:  # Lift
        runs_for_charts = baseline_runs | guarded_runs
        chart_label = "All"

    chart_df = df[df["run_id"].isin(runs_for_charts)] if runs_for_charts else df

    col_t1, col_cost = st.columns(2)
    with col_t1:
        st.markdown(f"**Tier-1 % by model** ({chart_label})")
        tier1 = (
            chart_df.groupby("model_under_test")["tier"]
              .apply(lambda s: (s == "1").mean() * 100.0)
              .sort_values(ascending=False)
        )
        tier1.index = [short_model_name(m) for m in tier1.index]
        st.bar_chart(tier1)
    with col_cost:
        st.markdown(f"**Est. cost by model** ({chart_label})")
        cost = (
            chart_df.assign(c=chart_df.apply(estimate_cost_usd, axis=1))
              .groupby("model_under_test")["c"].sum()
              .sort_values(ascending=False)
        )
        cost.index = [short_model_name(m) for m in cost.index]
        st.bar_chart(cost)

    if cost.sum() > 0:
        st.markdown(f"**Cost efficiency** — Tier-1 % per dollar ({chart_label})")
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

        st.markdown("**Per-model verdict**")
        # HTML mini-table with pills for an at-a-glance read
        per_model_rows = []
        per_model = (
            sub.groupby("model_under_test")
            .agg(
                avg_score=("value", "mean"),
                worst_axis_score=("value", "min"),
            )
            .reset_index()
            .sort_values("avg_score", ascending=False)
        )
        for _, r in per_model.iterrows():
            mut = r["model_under_test"]
            guards = sorted({a for a in sub[sub["model_under_test"] == mut]["guardrail_action"] if a})
            guard_html = " ".join(code_pill(g) for g in guards) if guards else (
                '<span style="color:#94a3b8;font-size:0.8em;">no guards fired</span>'
            )
            per_model_rows.append(
                f"<tr>"
                f"<td style='padding:8px 12px;'>{model_chip(mut)}</td>"
                f"<td style='padding:8px 12px;'>{score_pill(r['avg_score'])}</td>"
                f"<td style='padding:8px 12px;'>{score_pill(r['worst_axis_score'])}</td>"
                f"<td style='padding:8px 12px;'>{guard_html}</td>"
                f"</tr>"
            )
        st.markdown(
            '<table style="width:100%;border-collapse:collapse;'
            'border:1px solid #e5e7eb;border-radius:8px;overflow:hidden;'
            'font-size:0.9em;">'
            '<thead style="background:#f8fafc;"><tr>'
            '<th style="padding:10px;text-align:left;color:#475569;">Model</th>'
            '<th style="padding:10px;text-align:left;color:#475569;">Avg</th>'
            '<th style="padding:10px;text-align:left;color:#475569;">Worst axis</th>'
            '<th style="padding:10px;text-align:left;color:#475569;">Guard actions</th>'
            '</tr></thead><tbody>'
            + "".join(per_model_rows)
            + '</tbody></table>',
            unsafe_allow_html=True,
        )
        st.markdown("")  # spacer

        st.markdown("---")
        st.markdown("**Responses**")
        for (mut, transform), gdf in sub.groupby(["model_under_test", "attack_transform"]):
            avg = float(gdf["value"].mean())
            worst = float(gdf["value"].min())
            guard = next((a for a in gdf["guardrail_action"] if a), "")
            header = (
                f'{model_chip(mut, small=True)} &nbsp; '
                f'{transform_pill(transform)} &nbsp; '
                f'{score_pill(avg)} &nbsp; '
                f'<span style="color:#64748b;font-size:0.78em;">worst {worst:.2f}</span>'
            )
            if guard:
                header += f' &nbsp; ⛔ {code_pill(guard)}'
            st.markdown(header, unsafe_allow_html=True)
            with st.expander("Show response + axis verdicts", expanded=False):
                st.markdown(gdf["response"].iloc[0][:2000])
                axis_view = gdf[["axis", "value", "tier", "scorer_name",
                                 "disagreement", "rationale"]].copy()
                axis_view["axis"] = axis_view["axis"].apply(short_axis_name)
                st.dataframe(
                    axis_view, use_container_width=True, hide_index=True,
                    column_config={
                        "value": st.column_config.NumberColumn("Score", format="%.2f"),
                        "disagreement": st.column_config.NumberColumn(
                            "Disagree", format="%.3f"),
                        "rationale": st.column_config.TextColumn(
                            "Rationale", width="large"),
                    },
                )
                for _, row in gdf.iterrows():
                    extra = row.get("extra", "")
                    try:
                        extra = json.loads(extra) if isinstance(extra, str) else {}
                    except Exception:
                        continue
                    if extra and "per_judge" in extra:
                        st.caption(f"Per-judge breakdown · {axis_pill(row['axis'])}")
                        st.dataframe(
                            pd.DataFrame.from_dict(extra["per_judge"], orient="index"),
                            use_container_width=True,
                        )




# ===========================================================================
# Tab 6 — Demo: pull a real past exploit, replay it, watch guards block
# ===========================================================================

with tab_demo:
    st.markdown("#### Replay a real exploit — before vs after guards")
    st.caption(
        "Pick a prompt that broke a model in past audits. The **Before** "
        "column shows the model's actual past response (the failure that "
        "made it Tier-3). The **After** column re-runs the same prompt "
        "through the full guard cascade — usually short-circuits before "
        "the model is even called."
    )

    # ------------- Pick a model context ---------------------------------
    summary_all = runs_summary(df) if not df.empty else pd.DataFrame()
    if summary_all.empty:
        st.info("No audit data loaded. Run `examples/kitchen_sink.py` first.")
        st.stop()

    available_models = sorted(summary_all["model_under_test"].unique())
    chosen_mut = st.selectbox(
        "Audited model",
        options=available_models,
        format_func=short_model_name,
        key="demo_mut",
    )

    # Find baseline + guarded runs for this model
    model_runs = summary_all[summary_all["model_under_test"] == chosen_mut]
    baseline_run = model_runs[model_runs["badge"] == "baseline"]
    if baseline_run.empty:
        st.warning(
            f"No baseline (un-guarded) run found for {short_model_name(chosen_mut)}. "
            "Need a baseline audit so the Demo tab can find real failures."
        )
        st.stop()
    baseline_run_id = baseline_run.iloc[0]["run_id"]

    # ------------- Pick a Tier-3 failure from this baseline run -------
    fails = df[
        (df["run_id"] == baseline_run_id)
        & (df["value"] < 1.0)
    ].copy()
    if fails.empty:
        st.success(
            f"No Tier-3 failures in baseline `{baseline_run_id[-8:]}` — "
            "this model didn't break on any probe. Pick a different model "
            "or run a larger audit."
        )
        st.stop()

    # One row per unique (instance_id, attack_transform) — collapse rows that
    # the same prompt failed on multiple axes
    fails = fails.sort_values("value").drop_duplicates(
        subset=["instance_id", "attack_transform"], keep="first",
    )
    st.caption(
        f"Pulling Tier-3 failures from baseline run "
        f"{code_pill(baseline_run_id[-8:])} ({len(fails)} unique exploits)",
    )
    st.markdown("", unsafe_allow_html=True)

    # Display labels for the selectbox
    fails["__label"] = fails.apply(
        lambda r: (
            f"{r['instance_id']}  ·  "
            f"axis={short_axis_name(r['axis'])}  ·  "
            f"transform={r['attack_transform']}  ·  "
            f"score={r['value']:.2f}"
        ),
        axis=1,
    )
    chosen_label = st.selectbox(
        "Pick an exploit to replay",
        options=fails["__label"].tolist(),
        key="demo_exploit",
    )
    chosen_fail = fails[fails["__label"] == chosen_label].iloc[0]

    # ------------- Configure guard cascade for the After side ---------
    st.divider()
    st.markdown("**After-side cascade — which guards to enable for the replay:**")
    g_col1, g_col2, g_col3 = st.columns(3)
    with g_col1:
        use_pattern = st.checkbox("Pattern (regex)", value=True, key="demo_pattern")
    with g_col2:
        use_embedding = st.checkbox(
            "Embedding (fail-index)", value=True, key="demo_embedding",
            help="Loads examples/out/fail_index_<slug>.npz for this model",
        )
    with g_col3:
        use_classifier = st.checkbox(
            "Classifier ($0.0001/call)", value=False, key="demo_classifier",
        )

    rerun_before = st.checkbox(
        "Re-run Before live (instead of showing historical response)",
        value=False, key="demo_rerun_before",
        help="Off (default): show the model's past Tier-3 response from "
             "the audit log — instant, free, deterministic. "
             "On: call the model live with the same prompt.",
    )

    run = st.button("▶ Replay this exploit", type="primary", key="demo_run")

    # ------------- Pre-pick context preview -----------------------------
    st.markdown("**Exploit prompt:**")
    st.code(chosen_fail["prompt"][:1500], language=None)

    if run:
        import re as _re
        import time as _time
        try:
            from argus import (
                OpenRouterProvider, ModalProvider, GuardrailedProvider,
                PreFlightPatternGuard, PreFlightClassifierGuard,
                PreFlightEmbeddingGuard, LlamaPromptGuardScorer, FailIndex,
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to import argus: {e}")
            st.stop()

        # Build the inner provider from the audited model's identity
        # Format is "<provider_kind>:<model>" e.g. "openrouter:openai/gpt-4o-mini"
        provider_kind, _, model_id = chosen_mut.partition(":")
        provider_kind = provider_kind or "openrouter"
        try:
            if provider_kind == "modal":
                inner = ModalProvider()
            else:
                inner = OpenRouterProvider(model=model_id or "openai/gpt-4o-mini")
        except Exception as e:  # noqa: BLE001
            st.error(f"Provider init failed: {e}")
            st.stop()

        # Build After-side guarded provider
        slug = _re.sub(r"[^a-zA-Z0-9]+", "_", model_id or "default").strip("_").lower() or "default"
        guards = []
        cascade_info = []
        if use_pattern:
            guards.append(PreFlightPatternGuard())
            cascade_info.append(("Pattern (regex)", "armed"))
        if use_embedding:
            fail_index_path = Path(audit_dir) / f"fail_index_{slug}.npz"
            if fail_index_path.exists():
                idx = FailIndex(fail_index_path)
                guards.append(PreFlightEmbeddingGuard(idx))
                cascade_info.append(
                    (f"Embedding ({len(idx)} vectors in fail-index)", "armed"),
                )
            else:
                cascade_info.append(
                    (f"Embedding (no fail-index for {slug})", "skipped"),
                )
        if use_classifier:
            try:
                guards.append(PreFlightClassifierGuard(
                    scorer=LlamaPromptGuardScorer(), label="prompt_guard",
                ))
                cascade_info.append(("Classifier (Prompt-Guard-2)", "armed"))
            except Exception as e:  # noqa: BLE001
                cascade_info.append(("Classifier", f"failed: {e}"))
        guarded = GuardrailedProvider(inner=inner, pre_flight=guards) if guards else inner

        # -----------  BEFORE  -----------
        if rerun_before:
            t0 = _time.time()
            with st.spinner("Calling raw model (no guards)…"):
                try:
                    before_resp = inner.chat(
                        [{"role": "user", "content": chosen_fail["prompt"]}],
                        max_tokens=512, temperature=0.0,
                    )
                except Exception as e:  # noqa: BLE001
                    before_resp = f"[INFERENCE_BLOCKED: {type(e).__name__}: {str(e)[:200]}]"
            before_lat = _time.time() - t0
            before_score = "(live re-run)"
            before_rationale = "Re-ran live; original audit verdict was below."
        else:
            before_resp = chosen_fail["response"]
            before_lat = float(chosen_fail.get("latency_ms", 0) or 0) / 1000.0
            before_score = f"{chosen_fail['value']:.2f}"
            before_rationale = chosen_fail.get("rationale") or ""

        # -----------  AFTER  -----------
        t0 = _time.time()
        with st.spinner("Calling through guard cascade…"):
            try:
                after_resp = guarded.chat(
                    [{"role": "user", "content": chosen_fail["prompt"]}],
                    max_tokens=512, temperature=0.0,
                )
            except Exception as e:  # noqa: BLE001
                after_resp = f"[INFERENCE_BLOCKED: {type(e).__name__}: {str(e)[:200]}]"
        after_lat = _time.time() - t0
        after_actions = list(getattr(guarded, "last_actions", []) or [])

        # ----------- Display ------------
        st.divider()
        st.markdown("### Before vs After")
        bcol, acol = st.columns(2)

        with bcol:
            st.markdown(
                f'<div style="margin-bottom:8px;">'
                f'<strong>Before</strong> — raw {model_chip(chosen_mut, small=True)} '
                f'<span style="color:#dc2626;font-size:0.8em;font-weight:600;">EXPLOITED</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:#dc26261a;border-left:4px solid #dc2626;'
                f'padding:12px;border-radius:6px;margin-bottom:12px;">'
                f'<div style="font-size:0.85em;color:#6b7280;">Tier-3 verdict</div>'
                f'<div style="font-size:1.2em;font-weight:600;color:#dc2626;">'
                f'score = {before_score}</div>'
                f'<div style="font-size:0.85em;margin-top:6px;color:#475569;">'
                f'{before_rationale[:300]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Latency: {before_lat:.2f}s")
            st.markdown("**Model response:**")
            st.code(str(before_resp)[:2500], language=None)

        with acol:
            if after_actions:
                first_tier = after_actions[0].split(":")[0]
                color = {"pattern_block": "#3b82f6",
                         "embedding_block": "#6366f1",
                         "classifier_block": "#0ea5e9"}.get(first_tier, "#16a34a")
                st.markdown(
                    f'<div style="margin-bottom:8px;">'
                    f'<strong>After</strong> — guarded {model_chip(chosen_mut, small=True)} '
                    f'<span style="color:#16a34a;font-size:0.8em;font-weight:600;">PROTECTED</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="background:#16a34a1a;border-left:4px solid #16a34a;'
                    f'padding:12px;border-radius:6px;margin-bottom:12px;">'
                    f'<div style="font-size:0.85em;color:#6b7280;">Blocked by</div>'
                    f'<div style="font-family:monospace;font-size:1em;color:{color};'
                    f'font-weight:600;">{after_actions[0]}</div>'
                    f'<div style="font-size:0.85em;margin-top:6px;color:#475569;">'
                    f'Model was never called; canned refusal returned.'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="margin-bottom:8px;">'
                    f'<strong>After</strong> — guarded {model_chip(chosen_mut, small=True)} '
                    f'<span style="color:#ca8a04;font-size:0.8em;font-weight:600;">ALL GUARDS PASSED</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="background:#ca8a041a;border-left:4px solid #ca8a04;'
                    f'padding:12px;border-radius:6px;margin-bottom:12px;">'
                    f'<div style="font-size:0.85em;color:#475569;">'
                    f'No guard fired; model was called. Check the response — '
                    f'the model may now refuse on its own, or this exploit '
                    f'may evade the current cascade. </div></div>',
                    unsafe_allow_html=True,
                )
            st.caption(f"Latency: {after_lat:.2f}s")
            st.markdown("**Response:**")
            st.code(str(after_resp)[:2500], language=None)

        # Cascade trace beneath
        st.divider()
        st.markdown("**Cascade trace** (which tiers ran, which fired)")
        tier_to_prefix = {
            "Pattern": "pattern_block",
            "Embedding": "embedding_block",
            "Classifier": "classifier_block",
        }
        for tier_name, status in cascade_info:
            prefix = next(
                (v for k, v in tier_to_prefix.items() if k in tier_name), "",
            )
            hit = bool(prefix) and any(prefix in a for a in after_actions)
            icon = "⛔" if hit else ("○" if status == "armed" else "—")
            color = "#dc2626" if hit else ("#94a3b8" if status == "armed" else "#cbd5e1")
            st.markdown(
                f'<div style="font-family:monospace;font-size:0.85em;color:{color};">'
                f'{icon} {tier_name} <span style="color:#94a3b8;">({status})</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
