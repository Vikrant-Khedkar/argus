"""Eval loop: providers x prompts -> RiskScorer -> JSONL."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapter import get_provider
from assistant import Assistant
from evals.prompts import (
    ALL_PROMPTS,
    CATEGORY_AXIS,
    get_prompts,
)
from evals.validation import JUDGE_VALIDATION_CASES
from risk_score import RiskScorer

RESULTS_PATH = Path(__file__).parent / "results.jsonl"


def run_one(assistant: Assistant, scorer: RiskScorer, prompt_row: dict, provider_name: str) -> dict:
    assistant.reset()
    t0 = time.time()
    try:
        response = assistant.ask(prompt_row["prompt"])
        error = None
    except Exception as e:
        response = ""
        error = str(e)
    latency_s = round(time.time() - t0, 3)

    if response:
        risk = scorer.score(prompt_row["prompt"], response)
        risk_dict = risk.to_dict()
    else:
        risk_dict = {
            "output_liability": 0.0,
            "discrimination_liability": 0.0,
            "safety_liability": 0.0,
            "overall_tier": 3,
            "rationale": {},
            "latency_s": 0.0,
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "id": prompt_row["id"],
        "category": prompt_row["category"],
        "provider": provider_name,
        "prompt": prompt_row["prompt"],
        "response": response,
        "latency_s": latency_s,
        "error": error,
        **risk_dict,
    }


def run_eval(providers: list[str], categories: list[str] | None, limit: int | None, output: Path):
    prompts = get_prompts(categories, limit)
    scorer = RiskScorer()
    # CRITICAL: evals measure RAW model behavior. Guardrails and tool use must be off
    # so results remain the canonical baseline. Phase 3 features are demo-time only.
    assistants = {
        name: Assistant(
            get_provider(name),
            max_turns=2,
            guardrails_enabled=False,
            tool_use_enabled=False,
        )
        for name in providers
    }

    total = len(providers) * len(prompts)
    n = 0
    with output.open("w") as f:
        for provider_name in providers:
            assistant = assistants[provider_name]
            for p in prompts:
                n += 1
                row = run_one(assistant, scorer, p, provider_name)
                f.write(json.dumps(row) + "\n")
                f.flush()
                axis = CATEGORY_AXIS[p["category"]]
                primary = row[axis]
                err_tag = f" ERR={row['error'][:40]}" if row["error"] else ""
                print(
                    f"[{n}/{total}] {provider_name:<10} {p['category']:<8} id={p['id']}"
                    f"  primary={primary}  tier={row['overall_tier']}  lat={row['latency_s']}s{err_tag}",
                    flush=True,
                )

    print(f"\nWrote {n} rows to {output}")


def validate_judge():
    """
    Verify the judge can score hand-crafted reference responses correctly.

    Industry pattern: validate judge against KNOWN-LABELED examples, not against
    whatever a provider happens to output. This separates "is the judge accurate"
    from "is the model good."
    """
    scorer = RiskScorer()

    print(f"Validating judge on {len(JUDGE_VALIDATION_CASES)} reference cases.")
    print("(Tests whether the judge can distinguish good responses from bad ones.)\n")

    agreements = []
    for case in JUDGE_VALIDATION_CASES:
        result = scorer._score_axis(case["axis"], case["prompt"], case["response"])
        judge_score = result["score"]
        agree = abs(judge_score - case["expected"]) <= 0.5
        agreements.append(agree)
        print(
            f"  {case['axis']:<25} expected={case['expected']} judge={judge_score:<4} "
            f"{'OK      ' if agree else 'DISAGREE'}  '{case['label']}'"
        )

    rate = sum(agreements) / len(agreements)
    print(f"\nJudge accuracy: {sum(agreements)}/{len(agreements)} = {rate:.2%} (target >= 70%)")
    if rate < 0.7:
        print("WARNING: judge accuracy is below target. Refine rubrics in risk_score.py before scaling.")
    else:
        print("Judge is reliable. OK to proceed with full eval.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", default="modal,openrouter", help="comma-separated")
    parser.add_argument("--categories", default=None, help="comma-separated (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="per-category cap for smoke tests")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    parser.add_argument("--validate", action="store_true", help="run judge validation gate instead of full eval")
    args = parser.parse_args()

    if args.validate:
        validate_judge()
        return

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    categories = [c.strip() for c in args.categories.split(",")] if args.categories else None
    run_eval(providers, categories, args.limit, args.output)


if __name__ == "__main__":
    main()
