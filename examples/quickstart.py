"""Quickstart: score 10 safety probes against a model, render the memo."""
from argus import EvalConfig, Evaluator, load_safety_probes

cfg = EvalConfig.from_dict({
    "provider": {
        "name": "openrouter",
        "kwargs": {"model": "openai/gpt-4o-mini"},
    },
    "axes": {
        "safety_liability": {
            "type": "llm_judge",
            "model": "anthropic/claude-sonnet-4",
            "rubric_axis": "safety_liability",
        },
    },
    "transforms": ["identity"],
    "audit_log_path": "examples/out/quickstart.jsonl",
    "audit_db_path": "examples/out/quickstart.db",
})

probes = load_safety_probes(n_harmbench=5, n_jailbreakbench=3, n_xstest=2)
report = Evaluator(cfg).audit(probes, vendor_name="gpt-4o-mini")
print(report.to_underwriting_memo(vendor_name="gpt-4o-mini"))
