"""
MATH-500 perturbed eval via Doubleword batch (writes under experiments/doubleword_batches/math500/).

Default eval model: Nemotron 120B (`MATH500_EVAL_MODEL`). No /no_think prefix (Nemotron never uses it).

Usage (repo root):
    .venv/bin/python src/doubledword/math500_perturbed_eval.py --n 10 --window 1h
    .venv/bin/python src/doubledword/math500_perturbed_eval.py --n 500 --window 24h
    .venv/bin/python src/doubledword/math500_perturbed_eval.py --batch-id <uuid>
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from doubleword_client import MATH500_BATCH_ROOT, MATH500_EVAL_MODEL, DEFAULT_COMPLETION_WINDOW
from perturbed_eval_smoke_test import run_perturbed_smoke_test

load_dotenv()


def main():
    p = argparse.ArgumentParser(description="MATH-500 perturbed Doubleword batch eval")
    p.add_argument("--n", type=int, default=10, help="Number of unique problems (default 10 smoke → 50 rows)")
    p.add_argument("--window", default=DEFAULT_COMPLETION_WINDOW, choices=["24h", "1h"])
    p.add_argument("--batch-id", default=None, help="Download existing batch instead of submitting")
    p.add_argument("--eval-model", default=MATH500_EVAL_MODEL)
    p.add_argument("--max-tokens", type=int, default=4096)
    args = p.parse_args()

    run_perturbed_smoke_test(
        eval_model=args.eval_model,
        n=args.n,
        input_path="data/perturbations/math500_perturbed.csv",
        prompt_column="prompt_sent",
        completion_window=args.window,
        eval_batch_id=args.batch_id,
        max_tokens=args.max_tokens,
        batch_root=MATH500_BATCH_ROOT,
        no_think_prefix=False,
        label="math500_perturbed_eval",
        id_column="unique_id",
    )


if __name__ == "__main__":
    main()
