"""Streamlit observability dashboard. Reads evals/live_log.jsonl."""

import statistics
from collections import Counter
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import streamlit as st

from observability import read_log

st.set_page_config(page_title="Argus — Observability", layout="wide")

st.title("📊 Observability")
st.caption("Live log of every assistant call, scored by RiskScorer in real time.")

if st.button("🔄 Refresh"):
    st.rerun()

rows = read_log()

if not rows:
    st.info("No log entries yet. Send a few prompts on the Chat page to populate this dashboard.")
    st.stop()


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


tiered_rows = [r for r in rows if r.get("overall_tier") is not None]
guard_rows = [r for r in rows if r.get("guardrail_action")]
tool_rows = [r for r in rows if r.get("tool_calls_count", 0) > 0]
lat_values = [r["latency_s"] for r in rows if r.get("latency_s")]

# Top-level KPIs
cols = st.columns(5)
cols[0].metric("Total calls", len(rows))
if tiered_rows:
    tier1 = sum(1 for r in tiered_rows if r["overall_tier"] == 1)
    cols[1].metric("Tier 1 %", f"{tier1/len(tiered_rows)*100:.0f}%")
else:
    cols[1].metric("Tier 1 %", "—")
cols[2].metric("Guardrail trips", len(guard_rows))
cols[3].metric("Tool calls", sum(r.get("tool_calls_count", 0) for r in rows))
cols[4].metric("p95 latency", f"{percentile(lat_values, 0.95):.1f}s")

st.divider()

# Recent calls table — surfaced first since it's the most concrete signal
st.subheader("Recent calls")
recent = list(reversed(rows[-20:]))
table_data = []
for r in recent:
    ts = r.get("timestamp", "")
    try:
        ts_short = datetime.fromisoformat(ts).astimezone(timezone.utc).strftime("%H:%M:%S")
    except Exception:
        ts_short = ts[-8:] if ts else ""
    table_data.append({
        "time": ts_short,
        "provider": r.get("provider"),
        "prompt": (r.get("prompt") or "")[:60],
        "tier": r.get("overall_tier"),
        "guardrail": r.get("guardrail_action"),
        "tools": r.get("tool_calls_count", 0),
        "latency_s": r.get("latency_s"),
    })
st.dataframe(table_data, use_container_width=True, hide_index=True)

st.divider()

# Per-provider breakdown
st.subheader("Per-provider breakdown")
providers = sorted({r["provider"] for r in rows})
prov_cols = st.columns(len(providers) or 1)
for col, prov in zip(prov_cols, providers):
    prov_rows = [r for r in rows if r["provider"] == prov]
    with col:
        st.markdown(f"**{prov}** — {len(prov_rows)} calls")
        if any(r.get("overall_tier") for r in prov_rows):
            scores = {
                "Output": [r.get("output_liability") for r in prov_rows if r.get("output_liability") is not None],
                "Discrim": [r.get("discrimination_liability") for r in prov_rows if r.get("discrimination_liability") is not None],
                "Safety": [r.get("safety_liability") for r in prov_rows if r.get("safety_liability") is not None],
            }
            for axis, vals in scores.items():
                if vals:
                    st.write(f"{axis}: {statistics.mean(vals):.2f}")
        else:
            st.caption("(no risk-scored rows for this provider yet — enable guardrails in env)")

st.divider()

# Time-series of risk scores
scored_rows = [r for r in rows if r.get("overall_tier") is not None]
if scored_rows:
    st.subheader("Risk scores over time (rolling avg)")
    window = max(1, min(10, len(scored_rows) // 3))

    def rolling_avg(values, window):
        out = []
        for i in range(len(values)):
            chunk = values[max(0, i - window + 1) : i + 1]
            out.append(sum(chunk) / len(chunk))
        return out

    x = list(range(1, len(scored_rows) + 1))
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for axis, color in [
        ("output_liability", "tab:blue"),
        ("discrimination_liability", "tab:orange"),
        ("safety_liability", "tab:green"),
    ]:
        vals = [r.get(axis, 0) for r in scored_rows]
        ax.plot(x, rolling_avg(vals, window), label=axis.replace("_liability", "").title(), color=color, marker="o", markersize=4, linewidth=2)
    ax.set_ylim(-0.1, 2.15)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticklabels(["0\n(high risk)", "0.5", "1.0", "1.5", "2.0\n(low risk)"])
    ax.axhline(1.5, linestyle="--", color="#2ca02c", alpha=0.3, label="_Tier 1 floor")
    ax.axhline(1.0, linestyle="--", color="#ffbf00", alpha=0.3, label="_Tier 2 floor")
    if len(x) == 1:
        ax.set_xlim(0.5, 1.5)
    ax.set_xlabel("Call number")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)

# Tier distribution and guardrail breakdown
c1, c2 = st.columns(2)
with c1:
    if tiered_rows:
        st.subheader("Tier distribution")
        tier_counts = Counter(r["overall_tier"] for r in tiered_rows)
        fig, ax = plt.subplots(figsize=(5, 3))
        tiers = [1, 2, 3]
        colors = {1: "#2ca02c", 2: "#ffbf00", 3: "#d62728"}
        ax.bar(tiers, [tier_counts.get(t, 0) for t in tiers], color=[colors[t] for t in tiers])
        ax.set_xticks(tiers)
        ax.set_xticklabels(["Tier 1", "Tier 2", "Tier 3"])
        ax.set_ylabel("Count")
        fig.tight_layout()
        st.pyplot(fig)
with c2:
    if guard_rows:
        st.subheader("Guardrail interventions")
        actions = Counter(r["guardrail_action"] for r in guard_rows)
        for action, count in actions.most_common():
            st.write(f"`{action}` — **{count}**")
    else:
        st.subheader("Guardrail interventions")
        st.caption("None yet. Try a jailbreak prompt to trigger one.")
