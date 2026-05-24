# Argus 👁️

**Pluggable AI vendor risk audit SDK.** Point it at any chat model — OSS, frontier, hosted, self-deployed — and produce an underwriting-grade report measuring liability risk across 5 axes: safety, toxicity, discrimination, output hallucination, PII leakage.

## What it does

- **Five-axis liability scoring** with a per-axis scoring policy: classifier-primary for safety (Llama Guard 4), Legion-mode ensemble for discrimination (4 voters), composite with audit-sampling for toxicity, LLM-judge for hallucination, dual-classifier with disagreement-triggered judge for PII.
- **Four-tier pre-flight guard cascade**: regex → dynamic embedding (Rebuff-style fail-index that learns from past failures) → Llama-Prompt-Guard-2 classifier. Each tier catches what the prior missed.
- **Procedural attack transforms** + 5 hand-crafted multi-turn jailbreak scenarios.
- **A/B audit pipeline**: same probes run twice (baseline vs guarded), `AuditReport.compare()` produces the underwriting lift table.
- **Resumable, parallel, model-tagged**: AIMD-adaptive concurrency, crash-safe via JSONL, per-model fail-index, cross-model leaderboard.
- **Streamlit dashboard**: `argus_dashboard.py` — auto-discovers audit stores, color-coded by model, side-by-side response viewer.

## Quickstart

```bash
git clone <repo> && cd argus
uv sync
cp .env.example .env   # fill in MODAL_URL/KEY + OPENROUTER_API_KEY
```

Smallest possible audit (~30 sec, ~$0.05):

```python
from argus import EvalConfig, Evaluator, load_safety_probes

cfg = EvalConfig.from_dict({
    "provider": {"name": "openrouter",
                 "kwargs": {"model": "openai/gpt-4o-mini"}},
    "axes": {"safety_liability": {"type": "llama_guard"}},
    "transforms": ["identity"],
    "audit_log_path": "audit/run.jsonl",
    "audit_db_path":  "audit/index.db",
})

report = Evaluator(cfg).audit(load_safety_probes(n_harmbench=5))
print(report.to_underwriting_memo(vendor_name="gpt-4o-mini"))
```

Full demo with every feature wired together:

```bash
ARGUS_PROVIDER=openrouter ARGUS_MODEL=openai/gpt-4o-mini ARGUS_PROBE_SCALE=medium \
  uv run python examples/kitchen_sink.py
```

Then open the dashboard:

```bash
uv run streamlit run argus_dashboard.py
```

## Bundled examples

```bash
uv run python examples/quickstart.py        # 10 safety probes, classifier-final
uv run python examples/audit_sample.py      # classifier + N% LLM-judge audit
uv run python examples/custom_axis.py       # define a regulatory_compliance axis
uv run python examples/multi_turn_audit.py  # 5 multi-turn attack scenarios
uv run python examples/guardrail_lift.py    # A/B with vs without GuardrailedProvider
uv run python examples/legion_mode.py       # 3-judge ensemble with disagreement
uv run python examples/kitchen_sink.py      # every feature in one run
uv run python examples/kitchen_sink.py --resume    # resume if it died
```

## Configuration via env vars

| Var | What it controls | Default |
|---|---|---|
| `ARGUS_PROVIDER` | `modal` / `openrouter` / `http` / `huggingface` | `modal` |
| `ARGUS_MODEL` | model id passed to provider | provider default |
| `ARGUS_PROBE_SCALE` | `small` / `medium` / `large` / `full` | `small` |
| `ARGUS_MAX_CONCURRENT` | AIMD semaphore cap for OpenRouter calls | 24 |
| `ARGUS_HF_TIMEOUT_S` | HF dataset load timeout (s) | 120 |
| `ARGUS_RESUME` | run_id (or `"latest"`) to resume from | unset |

## Layout

```
src/argus/
├── types.py / probes.py / axes.py / tier_mapping.py    # Data shapes
├── rubrics.py                                          # Default LLM-judge rubrics
├── conversation.py                                     # ConversationProbe + score_conversation
├── datasets.py                                         # HF loaders per axis
├── concurrency.py                                      # AIMD-adaptive semaphore
├── eval_config.py + evaluator.py                       # YAML → built objects → audit() loop
├── providers/                                          # ChatProvider + 5 concrete + GuardrailedProvider
├── scorers/                                            # Scorer ABC + 11 concrete + composite + multi_judge
├── transforms/                                         # AttackTransform + 7 procedural multipliers
├── guardrails/                                         # PreFlightPattern + Classifier + Embedding (fail-index)
└── storage/                                            # JSONL (durable) + SQLite (regenerable) + AuditReport

argus_dashboard.py    # Streamlit web view — auto-discovers examples/out/*.db
examples/             # 7 runnable demos
modal_app.py          # Modal Qwen2.5-1.5B-Instruct deployment
tests/                # pytest smoke
```

## Design

Argus's value vs static benchmarks like HELM is the **dynamic guard architecture** + the **A/B baseline-vs-guarded story**:

1. **Per-axis scoring policy.** Each axis picks the cheapest signal that doesn't have a failure mode on its content type. Safety = Llama Guard (no self-refusal). Discrimination = Legion mode (4 voters from different families, median-aggregated, robust to one judge refusing). PII = Presidio NER + HELM-style exact-match in parallel, with LLM-judge tie-break on disagreement.

2. **Dynamic embedding guard.** After each audit, prompts that Llama Guard marked `unsafe` get embedded (`bge-small-en-v1.5`) and appended to a per-model fail-index (numpy `.npz`). The next run's pre-flight guard blocks any incoming prompt whose cosine similarity to a known failure is ≥ 0.85. **Each audit makes future audits smarter** — paraphrases of past exploits get caught for free.

3. **Resumable + crash-safe.** JSONL is the durable source of truth; SQLite is rebuilt from it on every `audit()` call. Schema auto-migrates. `audit(probes, resume_run_id="latest")` skips inference + scoring for `(probe, axis, transform)` tuples already in the log.

4. **AIMD-adaptive concurrency.** All OpenRouter calls share a single `Semaphore(24)` with TCP-congestion-style backoff: halve on 429, additively grow on success. Flat pool replaces nested `probe_workers × axis_workers`; measured 9× throughput vs sequential.

## Tested at scale

Three models audited at medium/full scale with the full 4-tier cascade:

| Model | Guarded avg | Tier-1 % | Notable |
|---|---|---|---|
| Qwen 2.5-1.5B (Modal, ours) | **1.962** | **97.9%** | Output_liability lifted 1.24 → 1.92 (+54.9%) |
| Qwen 2.5-7B (OpenRouter) | 1.940 | 96.7% | Embedding guard fired 80+ times across 10 attack templates |
| GPT-4o-mini (OpenRouter, frontier baseline) | 1.934 | 96.7% | 89 Tier-3 failures, mostly SimpleQA hallucinations + XSTest over-refusal |

**Headline:** OSS Qwen 1.5B + Argus 4-tier cascade reaches **higher Tier-1 share than GPT-4o-mini** on the same probe set, at ~1/5 the inference cost.

## Modal deployment

`modal_app.py` deploys Qwen2.5-1.5B-Instruct as our hosted reference endpoint. Auth via `X-API-Key` against `MODAL_AUTH_TOKEN`:

```bash
uv run modal deploy -m modal_app
```

```bash
curl -X POST $MODAL_URL -d '{"messages":[{"role":"user","content":"hi"}]}'                # 401
curl -X POST $MODAL_URL -H "X-API-Key: $MODAL_API_KEY" -d '{...}'                         # works
```
