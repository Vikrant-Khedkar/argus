"""A/B: same probes, same model, run twice — without and with a
GuardrailedProvider — then report per-axis lift."""

from argus import EvalConfig, Evaluator, load_safety_probes

LOG = "examples/out/lift.jsonl"
DB  = "examples/out/lift.db"

base = {
    "provider": {
        "name": "openrouter",
        "kwargs": {"model": "openai/gpt-4o-mini"},
    },
    "axes": {
        "safety_liability": {
            "type": "llama_guard",
        },
    },
    "transforms": ["identity", "persona_swap"],
    "audit_log_path": LOG,
    "audit_db_path": DB,
}

probes = load_safety_probes(n_harmbench=10, n_jailbreakbench=6, n_xstest=4)

# Run 1 — baseline, no guardrails
print("=== Run 1: baseline (no guardrails) ===")
cfg_base = EvalConfig.from_dict(base)
report_base = Evaluator(cfg_base).audit(probes, vendor_name="baseline")
baseline_run = report_base.run_id

# Run 2 — guardrailed
print("=== Run 2: with GuardrailedProvider pre-flight pattern guard ===")
cfg_guarded = EvalConfig.from_dict({
    **base,
    "provider": {
        "name": "openrouter",
        "kwargs": {"model": "openai/gpt-4o-mini"},
        "guardrail": {"pre_flight": ["pattern"]},
    },
})
report_guarded = Evaluator(cfg_guarded).audit(probes, vendor_name="guarded")

print()
print(report_guarded.to_underwriting_memo(
    baseline_run_id=baseline_run,
    vendor_name="gpt-4o-mini + Argus guardrails",
    title="Guardrail Lift Audit",
))
