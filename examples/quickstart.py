"""Quickstart: score safety probes with classifier-final policy.

Llama Guard 4 12B is the source of truth — no LLM-judge fallback. This
matches the deployed pattern at Anthropic / OpenAI / MLCommons AILuminate:
the safety classifier is final, because it (a) doesn't self-refuse on
extreme content and (b) doesn't expose calibrated probabilities for
threshold-based escalation. See `audit_sample.py` for the audit-sample
pattern that adds an LLM judge on a random N% of probes.
"""
from argus import EvalConfig, Evaluator, load_safety_probes

cfg = EvalConfig.from_dict({
    "provider": {
        "name": "openrouter",
        "kwargs": {"model": "openai/gpt-4o-mini"},
    },
    "axes": {
        "safety_liability": {
            "type": "llama_guard",   # meta-llama/llama-guard-4-12b
        },
    },
    "transforms": ["identity"],
    "audit_log_path": "examples/out/quickstart.jsonl",
    "audit_db_path": "examples/out/quickstart.db",
})

probes = load_safety_probes(n_harmbench=5, n_jailbreakbench=3, n_xstest=2)
report = Evaluator(cfg).audit(probes, vendor_name="gpt-4o-mini")
print(report.to_underwriting_memo(vendor_name="gpt-4o-mini"))
