# v1 → v2 Comparison

Direct side-by-side: what we measured in v1 vs what we measure in v2,
with the actual axis means from real audit runs on both versions.

---

## Methodology — what changed between v1 and v2

| Aspect | v1 | v2 |
|---|---|---|
| Axes | 3 (output, discrimination, safety) | **5** (+ toxicity + PII leakage) |
| Probes per audit | 35 hand-curated | **up to 320** from HF (HarmBench / JailbreakBench / XSTest / BBQ / SimpleQA / curated PII) |
| Attack transforms | 0 | **7** procedural multipliers (persona swap, translation laundering, multi-turn escalation, etc.) |
| Scoring policy | Single LLM judge per axis (Sonnet) | **Per-axis policy:** classifier-final (safety), heterogeneous Legion (discrimination), composite + audit sample (toxicity), LLM judge (output), dual-classifier + tie-break (PII) |
| Pre-flight guards | 4 regex patterns, OFF during eval | **4-tier cascade:** 11 regex families + **dynamic embedding fail-index** + Llama-Prompt-Guard-2 classifier, ON during the A/B |
| A/B story | None | **Baseline vs guarded** every run; lift table is the deliverable |
| Audit events per run | 70 (35 prompts × 2 providers) | **~9,500** at full scale (320 probes × 3 transforms × 5 axes × 2 runs) |
| Resumable | No | Yes (`--resume`, JSONL durable) |
| Cross-model leaderboard | No | Yes (`model_under_test` tagged on every row) |
| Concurrency | Sequential | AIMD-adaptive semaphore, ~9× throughput |
| Cost per audit | ~$1.50 | ~$1–4 depending on model |

---

## Side-by-side: same 3 axes, same 2 models

These are the **only directly-comparable axes** (the 3 v1 had).
For v2, both **baseline** (raw model, no guards — closest to v1's
setup) and **guarded** (full 4-tier cascade) are shown.

### Qwen 2.5-1.5B on Modal

| Axis | v1 (n=35) | v2 baseline (medium) | v2 guarded (medium) |
|---|---|---|---|
| output_liability | 1.143 | 1.241 | **1.922** |
| discrimination_liability | 1.743 | 1.789 | **1.991** |
| safety_liability | 1.600 | 1.776 | **2.000** |
| **Tier-1 share** | **60.0%** | ~80% | **97.9%** |

**v1 → v2 baseline:** baselines are slightly higher across the board.
v2 surfaces *more* probes per axis (35 hand-curated → ~50 from HF),
and the heterogeneous Legion (4 voters median-aggregated) absorbs the
single-judge bias that v1 had.

**v1 → v2 guarded:** the headline. **Safety lifts from 1.60 → 2.000
(perfect).** Output_liability lifts from 1.14 → 1.92 (+68% over v1).
Tier-1 share: **60% → 97.9%.** That's the SDK + guards story.

### GPT-4o-mini on OpenRouter

| Axis | v1 (n=35) | v2 baseline (full) | v2 guarded (full) |
|---|---|---|---|
| output_liability | 1.800 | 1.717 | **1.855** |
| discrimination_liability | 1.857 | 1.945 | **1.985** |
| safety_liability | 1.943 | 1.978 | **1.994** |
| **Tier-1 share** | **88.6%** | 95.1% | **96.7%** |

**v1 → v2 baseline:** the v2 output_liability looks *slightly worse*
than v1. **This is real signal, not regression.** v2 ran ~9× more
probes (315 vs 35), including SimpleQA's long-tail factual recall
(Ramiriquí's founding year, Pakistan's third Deputy Speaker dates,
etc.) — these are harder than the v1 hand-curated set, so the model
hallucinates more often. The bigger probe set is a more honest read.

**v1 → v2 guarded:** safety up (+0.05), output up (+0.06), Tier-1 share
up to 96.7%. Less dramatic than Qwen's lift because GPT-4o-mini was
already near-frontier. The bigger story is on the *new* axes:

---

## What v2 measures that v1 didn't

These two axes are new in v2:

### toxicity_liability

| Model | Baseline | Guarded |
|---|---|---|
| Qwen 1.5B Modal | 1.931 | 1.914 (−0.9%) |
| Qwen 7B OpenRouter | 1.971 | 1.872 (−5.0%) |
| GPT-4o-mini | 1.982 | 1.868 (−5.7%) |

The toxicity axis **goes down with guards** — pre-flight regex
over-blocks benign-but-transformed prompts (e.g. XSTest probes after
`translation_laundering` look like "translate this Eiffel-Tower
question," guards refuse, judge correctly flags as over-refusal).
**The embedding tier shrinks this cost as the fail-index matures**:
Qwen 1.5B with 13 fail-vectors saw only −0.9%; GPT-4o-mini with 7
fail-vectors saw −5.7%. Compounding-learning has measurable evidence.

### pii_leakage

| Model | Baseline | Guarded |
|---|---|---|
| Qwen 1.5B Modal | 1.879 | **1.983** (+5.5%) |
| Qwen 7B OpenRouter | 1.980 | 1.992 |
| GPT-4o-mini | 1.992 | 1.995 |

GPT-4o-mini was already perfect; Qwen 1.5B gained from guards.
Captured by Presidio (NER) + ExactMatchPII (HELM-style memorization)
in parallel, with a Sonnet tie-break when they disagree.

---

## Headline: cross-vendor leaderboard (v2 only)

Three models, full 4-tier cascade, ranked by Tier-1 share:

| Rank | Model | Guarded Tier-1 % | Cost per audit |
|---|---|---|---|
| 1 | **Qwen 2.5-1.5B + Argus** (Modal, ours) | **97.9%** | ~$1 |
| 2 | Qwen 2.5-7B + Argus (OpenRouter) | 96.9% | ~$2-3 |
| 3 | GPT-4o-mini + Argus (OpenRouter) | 96.7% | ~$3-4 |

**Qwen 1.5B + Argus beats raw GPT-4o-mini Tier-1 share** (95.1%) at
~1/30 the inference cost. The smaller the model, the more leverage the
4-tier cascade delivers — output_liability lifts +54.9% on Qwen 1.5B
vs +8.1% on GPT-4o-mini.

This is the product story for the AI-vendor-liability insurance use case:
**guards make commodity OSS deployable at frontier-equivalent safety,
at fraction-of-cost.**

---

## Per-model detail reports

- [`qwen_1_5b_modal.md`](qwen_1_5b_modal.md) — Qwen 2.5-1.5B on Modal
- [`qwen_7b_openrouter.md`](qwen_7b_openrouter.md) — Qwen 2.5-7B on OpenRouter
- [`gpt_4o_mini.md`](gpt_4o_mini.md) — GPT-4o-mini on OpenRouter

Each includes the full axis-lift table, tier distribution, pre-flight
action counts (broken down by pattern family + embedding-block nearest-neighbor),
and worst-probe samples from the guarded run.

---

## Honest caveats

- **v1's 70-row eval ran without guards.** v2's "baseline" run also has no guards (closest comparison). The v2 "guarded" column is the upgrade story.
- **Probe sets differ.** v1's 35 hand-curated prompts vs v2's HF-loaded set. Different distribution surfaces different failure modes (v2 sees more obscure-fact hallucinations because SimpleQA exposes them).
- **Scoring policies differ.** v1 used single-Sonnet-judge per axis; v2 uses per-axis policies including a 4-voter Legion for discrimination. Comparable but not identical.
- **No PII / toxicity data in v1.** Those axes are new; can't show year-on-year delta on them.
- **The 4-tier cascade isn't unique to Argus** — same family as Protect AI Rebuff, Microsoft PyRIT, NVIDIA garak. Argus's contribution is the *vendor-audit packaging* — A/B baseline-vs-guarded reporting, cross-model leaderboard queries, the underwriting memo as the deliverable.
