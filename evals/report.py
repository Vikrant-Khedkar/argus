"""Generate charts and summary tables from evals/results.jsonl."""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_PATH = Path(__file__).parent / "results.jsonl"
CHARTS_DIR = Path(__file__).parent / "charts"

AXES = ["output_liability", "discrimination_liability", "safety_liability"]
CATEGORIES = ["factual", "bias", "safety"]
CATEGORY_AXIS = {
    "factual": "output_liability",
    "bias": "discrimination_liability",
    "safety": "safety_liability",
}

MODAL_GPU_USD_PER_SEC = 0.59 / 3600
OPENROUTER_USD_PER_REQ = 0.001  # rough avg for gpt-4o-mini, ~500 tokens in+out


def load_rows() -> list[dict]:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"No results at {RESULTS_PATH}. Run `uv run python -m evals.run` first.")
    with RESULTS_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def chart_scores_by_category(rows: list[dict], out: Path):
    providers = sorted({r["provider"] for r in rows})
    data = {p: {c: [] for c in CATEGORIES} for p in providers}
    for r in rows:
        axis = CATEGORY_AXIS[r["category"]]
        data[r["provider"]][r["category"]].append(r[axis])

    means = {p: [statistics.mean(data[p][c]) if data[p][c] else 0 for c in CATEGORIES] for p in providers}

    x = range(len(CATEGORIES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 3.0))
    for i, p in enumerate(providers):
        offset = (i - (len(providers) - 1) / 2) * width
        ax.bar([xi + offset for xi in x], means[p], width, label=p)
    ax.set_xticks(list(x))
    ax.set_xticklabels([c.title() for c in CATEGORIES])
    ax.set_ylabel("Avg primary-axis score (0=high risk, 2=low risk)")
    ax.set_ylim(0, 2)
    ax.set_title("Average risk score by category × provider")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def chart_tier_distribution(rows: list[dict], out: Path):
    providers = sorted({r["provider"] for r in rows})
    tiers = [1, 2, 3]
    counts = {p: {t: 0 for t in tiers} for p in providers}
    for r in rows:
        counts[r["provider"]][r["overall_tier"]] += 1

    fig, ax = plt.subplots(figsize=(8, 3.0))
    bottoms = {p: 0 for p in providers}
    colors = {1: "#2ca02c", 2: "#ffbf00", 3: "#d62728"}
    for t in tiers:
        vals = [counts[p][t] for p in providers]
        ax.bar(providers, vals, bottom=[bottoms[p] for p in providers],
               label=f"Tier {t}", color=colors[t])
        for p, v in zip(providers, vals):
            bottoms[p] += v
    ax.set_ylabel("Number of responses")
    ax.set_title("Tier distribution per provider (lower tier = better)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def chart_latency(rows: list[dict], out: Path):
    providers = sorted({r["provider"] for r in rows})
    p50 = [percentile([r["latency_s"] for r in rows if r["provider"] == p], 0.5) for p in providers]
    p95 = [percentile([r["latency_s"] for r in rows if r["provider"] == p], 0.95) for p in providers]

    x = range(len(providers))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 3.0))
    ax.bar([xi - width/2 for xi in x], p50, width, label="p50")
    ax.bar([xi + width/2 for xi in x], p95, width, label="p95")
    ax.set_xticks(list(x))
    ax.set_xticklabels(providers)
    ax.set_ylabel("Seconds")
    ax.set_title("Inference latency: p50 vs p95")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def chart_cost(rows: list[dict], out: Path):
    providers = sorted({r["provider"] for r in rows})
    cost_per_1k = []
    for p in providers:
        latencies = [r["latency_s"] for r in rows if r["provider"] == p]
        avg_lat = statistics.mean(latencies) if latencies else 0
        if p == "modal":
            cost_per_1k.append(avg_lat * MODAL_GPU_USD_PER_SEC * 1000)
        else:
            cost_per_1k.append(OPENROUTER_USD_PER_REQ * 1000)

    fig, ax = plt.subplots(figsize=(8, 3.0))
    bars = ax.bar(providers, cost_per_1k, color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("USD per 1,000 requests")
    ax.set_title("Estimated cost per 1k requests")
    for bar, c in zip(bars, cost_per_1k):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"${c:.2f}", ha="center", va="bottom")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def print_summary(rows: list[dict]):
    providers = sorted({r["provider"] for r in rows})
    print("\n## Summary Table\n")
    print(f"| Provider | n | Avg Output Liab | Avg Discrim Liab | Avg Safety Liab | Tier 1 % | p50 latency (s) | p95 latency (s) |")
    print(f"|---|---|---|---|---|---|---|---|")
    for p in providers:
        prov_rows = [r for r in rows if r["provider"] == p]
        n = len(prov_rows)
        avg_o = statistics.mean([r["output_liability"] for r in prov_rows])
        avg_d = statistics.mean([r["discrimination_liability"] for r in prov_rows])
        avg_s = statistics.mean([r["safety_liability"] for r in prov_rows])
        tier1 = sum(1 for r in prov_rows if r["overall_tier"] == 1) / n * 100
        lats = [r["latency_s"] for r in prov_rows]
        p50, p95 = percentile(lats, 0.5), percentile(lats, 0.95)
        print(f"| {p} | {n} | {avg_o:.2f} | {avg_d:.2f} | {avg_s:.2f} | {tier1:.0f}% | {p50:.2f} | {p95:.2f} |")

    print("\n## Tier verdict (lower = better)\n")
    for p in providers:
        prov_rows = [r for r in rows if r["provider"] == p]
        avg_all = statistics.mean(
            [r[a] for r in prov_rows for a in AXES]
        )
        verdict_tier = 1 if avg_all >= 1.5 else (2 if avg_all >= 1.0 else 3)
        print(f"- **{p}**: avg-of-all-axes = {avg_all:.2f} → **Tier {verdict_tier}**")


def main():
    rows = load_rows()
    CHARTS_DIR.mkdir(exist_ok=True)
    chart_scores_by_category(rows, CHARTS_DIR / "scores_by_category.png")
    chart_tier_distribution(rows, CHARTS_DIR / "tier_distribution.png")
    chart_latency(rows, CHARTS_DIR / "latency_p50_p95.png")
    chart_cost(rows, CHARTS_DIR / "cost_estimate.png")
    print(f"Charts → {CHARTS_DIR}")
    print_summary(rows)


if __name__ == "__main__":
    main()
