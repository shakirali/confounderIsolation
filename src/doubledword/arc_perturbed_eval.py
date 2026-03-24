"""
ARC-Challenge perturbed eval via Doubleword batch (writes under experiments/doubleword_batches/arc/).

Default eval model: Nemotron 120B (`ARC_EVAL_MODEL`). User messages are the raw task text unless you pass `--no-think`
(Qwen-style `/no_think` line; Nemotron never receives that prefix even with `--no-think`).

Usage (repo root):
    PYTHONPATH=src/doubledword uv run python src/doubledword/arc_perturbed_eval.py --n 10
    PYTHONPATH=src/doubledword uv run python src/doubledword/arc_perturbed_eval.py --batch-id <uuid>
"""

import argparse

from dotenv import load_dotenv

from doubleword_client import ARC_BATCH_ROOT, ARC_EVAL_MODEL, DEFAULT_COMPLETION_WINDOW
from perturbed_eval_smoke_test import run_perturbed_smoke_test

load_dotenv()


def main():
    p = argparse.ArgumentParser(description="ARC-Challenge perturbed Doubleword batch eval")
    p.add_argument("--n", type=int, default=10, help="Number of unique question_ids (default 10 smoke)")
    p.add_argument("--window", default=DEFAULT_COMPLETION_WINDOW, choices=["24h", "1h"])
    p.add_argument("--batch-id", default=None, help="Download existing batch instead of submitting")
    p.add_argument("--eval-model", default=ARC_EVAL_MODEL)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument(
        "--no-think",
        action="store_true",
        help="Prepend /no_think for Qwen-style models only (opt-in; default is raw prompt)",
    )
    args = p.parse_args()

    run_perturbed_smoke_test(
        eval_model=args.eval_model,
        n=args.n,
        input_path="data/perturbations/arc_challenge_test_perturbed.csv",
        prompt_column="prompt_sent",
        completion_window=args.window,
        eval_batch_id=args.batch_id,
        max_tokens=args.max_tokens,
        batch_root=ARC_BATCH_ROOT,
        no_think_prefix=args.no_think,
        label="arc_perturbed_eval",
    )


if __name__ == "__main__":
    main()
