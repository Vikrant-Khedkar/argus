# Argus Eval Strategy

Why every axis is scored the way it is, and how Argus avoids the failure
modes that bite naive LLM-judge pipelines.

## TL;DR

| Axis | Primary signal | Why |
|---|---|---|
| **safety_liability** | classifier-final (Llama Guard 4 12B); optional N% audit-sample LLM judge | LLM judges hit *self-refusal* on extreme content. Llama Guard is trained to classify harm, not avoid engaging with it — and chat-API classifiers don't expose calibrated probabilities, so the right "second opinion" pattern is random audit sampling, not confidence-threshold cascade. |
| **toxicity_liability** | classifier-final (Perspective API); optional N% audit-sample | Same self-refusal failure mode + Perspective is the industry-standard reference signal. |
| **discrimination_liability** | LLM judge (multi-judge in Legion mode) | No classifier with sufficient breadth for nuanced bias detection. Multi-judge mitigates single-judge bias. |
| **calibration_liability** | LLM judge w/ reference data | Detecting *confident wrongness* needs a model that can reason about claim strength. |
| **output_liability** | reference-based + LLM judge fallback | `KeyFactsScorer` is deterministic for probes with ground-truth references; LLM judge takes the rest. |
| **robustness** | deterministic (perturbation comparator) | Compare scores under attack-transforms vs identity. No external model required. |
| **efficiency** | deterministic (latency / cost telemetry) | Mechanical — read from provider response metadata. |

This is the *composite scoring policy* — each axis picks the cheapest signal
that doesn't have a failure mode on that axis's content, then falls back to
LLM judge only when the cheap signal is uncertain.

## Failure mode 1 — judge self-refusal

When a frontier judge sees a prompt like *"how do I synthesize sarin"* and the
model's response says *"step 1: …"*, some judges refuse to score it — they
treat the *evaluation* request as if it were the *harm* request and decline.
This silently caps Safety scoring at "I refused" floor, hiding the real
behavior of the model under test.

**Argus's mitigation:** the *primary* safety signal is `LlamaGuardScorer`,
which uses Meta's safety classifier. Llama Guard is trained to label
`safe` / `unsafe: <category>` — it does not have the alignment training that
makes frontier judges refuse to engage. LLM judges (`LLMJudgeScorer`) only
fire as the *fallback* (via `CompositeScorer.fallback_threshold`) when the
classifier is uncertain.

For axes where no good classifier exists (discrimination, calibration), we
deploy **Legion mode** (`MultiJudgeScorer`): three judges in parallel from
different families (Claude / GPT-4o / Gemini). Per-judge attribution and
stdev disagreement are stored alongside the aggregated score — if one judge
refuses (score=0 with a refusal rationale) the other two still vote.

## Failure mode 2 — judge bias toward verbose answers

LLM judges over-reward longer responses (length bias) and responses written
in their own family's style (family bias). Single-judge pipelines bake these
biases into every score.

**Argus's mitigation:** Legion mode aggregator defaults to `median` for
ordinal scores, which is robust to one judge's outlier. `disagreement` (stdev
across judges) surfaces in the AuditRow's `extra` blob; reviewers can flag
high-disagreement rows for manual inspection.

## Failure mode 3 — one-shot probes miss attack patterns

Real abuse is multi-turn: rapport → prime → payload. A single-prompt eval
will never catch a roleplay-laundering jailbreak that needs two innocuous
setup turns before the payload turn.

**Argus's mitigation:**
1. **Procedural multipliers** — `AttackTransform` subclasses apply each
   transform to every probe. `TranslationLaunderingTransform` × 50 raw
   safety probes = 50 new test cases without curation cost. Six transforms
   ship in v2: identity, translation laundering, persona swap, multi-turn
   escalation, typos, paraphrase, case change.
2. **Dedicated multi-turn probes** — `DEFAULT_MULTI_TURN_PROBES` ships five
   hand-crafted scenarios targeting the canonical patterns: escalation,
   persona-drift, roleplay-laundering, trust-building PII, context-dilution.
   `score_conversation()` walks the conversation through the provider and
   scores the final assistant turn.

## Failure mode 4 — public-dataset overfitting

If every AI vendor evaluates on HarmBench and JailbreakBench, models will be
RLHF'd to refuse those specific phrasings. The benchmark stops generalizing.

**Argus's mitigation:** *two-path safety coverage* —
- **Path A (procedural):** raw harm behaviors × attack transforms.
  Generalizes to phrasings the model hasn't seen.
- **Path B (in-the-wild):** pre-jailbroken prompts from JailbreakBench.
  Catches known-strong attack patterns.

The same audit run covers both, so a vendor can't pass by only memorizing
one side.

## Failure mode 5 — judge cost

Naive policy = one LLM judge per axis per probe. At Claude Sonnet 4 pricing
that's ~$0.005 per probe per axis. 1000 probes × 7 axes = $35 just for
scoring, before any retries.

**Argus's mitigation:** for safety + toxicity axes, the classifier
(Llama Guard, Perspective) is **final** — no LLM judge is called for the
hot path. This matches the deployed pattern at Anthropic / OpenAI /
MLCommons AILuminate. Classifier cost is ~$0.0002/probe via OpenRouter,
~25× cheaper than an LLM judge.

For audit assurance you optionally **add** an LLM judge on a random N% of
probes via `CompositeScorer(audit_sample_rate=0.15)` — the standard
"spot-check the cheap model with the expensive one" pattern. The judge
isn't a fallback; it's a parallel signal you can compare against the
classifier offline.

`CompositeScorer` exposes three escalation triggers, all standard:
  - `audit_sample_rate` — random N% audit (Meta integrity, YouTube spam,
    Stripe Radar all use this)
  - `value_zone=(lo, hi)` — escalate when a primary returns a borderline
    score (selective prediction; useful when primaries have a real ordinal
    output)
  - `fire_on_disagreement` — escalate when multiple primaries disagree by
    more than a threshold (ensemble disagreement)

What we explicitly do *not* do: threshold on a primary's self-reported
`confidence` field. Chat-API classifiers like Llama Guard don't expose
calibrated probabilities — that field would be a constant, so the trigger
would be dead code.

## How to evaluate the strongest models (e.g. Opus 4.7)

The dominant concern is **judge self-refusal at the extremes**. Opus 4.7
under test will sometimes generate content that any judge of comparable or
weaker capability will refuse to grade.

Defence:
1. **Use classifier-primary for safety/toxicity** (already the default). Llama
   Guard doesn't refuse.
2. **Use Legion mode for discrimination/calibration** with judges from
   different families. Probability of *all three* refusing is much smaller
   than any one.
3. **Detect refusal explicitly.** `RefusalRegexScorer` runs on every judge
   response; if a judge refuses, its verdict is dropped from the ensemble
   rather than counted as score=0.
4. **Include adversarial transforms.** A model that's been RLHF'd hard
   refuses surface forms — but persona-swap, translation laundering, and
   multi-turn escalation often bypass surface refusal. The transforms turn
   one strong probe into N hardened variants.

## What the audit report tells the underwriter

`AuditReport.to_underwriting_memo()` is the deliverable. It contains:

- **Axis means** — one row per axis, scored 0 (high liability) to 2 (low).
- **Tier distribution** — Tier 1 / 2 / 3 counts using the configured
  `TierMapping` (default: Insurance Tier 1 ≥ avg 1.5).
- **Guardrail-lift table** (if `baseline_run_id` is provided) — per-axis
  before vs after delta, with arrow + percentage. This is the value-prop
  for the broker: "GuardrailedProvider moved safety mean from 0.84 to 1.92,
  ▲ +1.08 (+129%)."
- **Worst-scoring probes** — bottom-N rows below `threshold`, with axis,
  rationale, attack transform / multi-turn flag, and instance id.

The same JSONL + SQLite store backs the web view (planned) — no schema
migration needed because every row already has run_id, instance_id, axis,
scorer_name, value, tier, plus the `extra` blob for ensemble + composite
detail.

## See also

- [EXTENDING_ARGUS.md](EXTENDING_ARGUS.md) — adding a scorer, axis, or
  transform.
- [`examples/legion_mode.py`](../examples/legion_mode.py) — Legion mode in
  practice with three judges.
- [`examples/guardrail_lift.py`](../examples/guardrail_lift.py) — A/B with
  vs without `GuardrailedProvider`, reads `compare()` table.
