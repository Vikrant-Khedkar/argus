"""Audit-sample pattern — classifier-primary with N% LLM-judge sampling.

The standard industry pattern: classifier is the source of truth for every
probe, but a random sample of probes ALSO gets the LLM judge for audit. This
is what Meta integrity, YouTube spam, and Stripe Radar deploy in production —
not "fallback on low confidence" (Llama Guard has no real confidence signal),
but "spot-check the cheap model with the expensive one".

Two other triggers are available — value_zone (escalate borderline scores)
and fire_on_disagreement (escalate when multiple primaries disagree). All
three can be combined; here we use audit_sample_rate alone.
"""
from argus import EvalConfig, Evaluator, load_safety_probes

cfg = EvalConfig.from_dict({
    "provider": {
        "name": "openrouter",
        "kwargs": {"model": "openai/gpt-4o-mini"},
    },
    "axes": {
        "safety_liability": {
            "type": "composite",
            "primary": [{"type": "llama_guard"}],
            "llm_fallback": {
                "type": "llm_judge",
                "model": "anthropic/claude-sonnet-4",
                "rubric_axis": "safety_liability",
            },
            "audit_sample_rate": 0.15,   # judge audits 15% of probes
        },
    },
    "transforms": ["identity"],
    "audit_log_path": "examples/out/audit_sample.jsonl",
    "audit_db_path":  "examples/out/audit_sample.db",
})

probes = load_safety_probes(n_harmbench=10, n_jailbreakbench=6, n_xstest=4)
report = Evaluator(cfg).audit(probes, vendor_name="gpt-4o-mini")
print(report.to_underwriting_memo(
    vendor_name="gpt-4o-mini",
    title="Classifier-Primary + 15% Judge Audit",
))

# Show how often the judge actually fired
import sqlite3
conn = sqlite3.connect("examples/out/audit_sample.db")
n_fired = conn.execute(
    "SELECT COUNT(*) FROM audit_rows WHERE run_id=? AND fallback_fired=1",
    (report.run_id,),
).fetchone()[0]
n_total = conn.execute(
    "SELECT COUNT(*) FROM audit_rows WHERE run_id=?", (report.run_id,),
).fetchone()[0]
print(f"\nJudge audited {n_fired}/{n_total} probes ({100 * n_fired / n_total:.0f}%)")
