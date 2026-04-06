"""
MultiFin baseline eval via Doubleword batch.

Writes under experiments/doubleword_batches/multifin/.
Eval model: Nemotron 120B. No /no_think prefix.
n=546 (all English test rows).

Usage (repo root):
    .venv/bin/python src/doubledword/multifin_baseline_eval.py --n 10 --window 1h
    .venv/bin/python src/doubledword/multifin_baseline_eval.py --window 24h
    .venv/bin/python src/doubledword/multifin_baseline_eval.py --batch-id <uuid>
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_eval_smoke_test_doubleword import run_smoke_test
from doubleword_client import MATH500_EVAL_MODEL, DEFAULT_COMPLETION_WINDOW

load_dotenv()

MULTIFIN_BATCH_ROOT = "experiments/doubleword_batches/multifin"


def main():
    p = argparse.ArgumentParser(description="MultiFin baseline Doubleword batch eval")
    p.add_argument("--n", type=int, default=10, help="Number of examples to eval (default 10 smoke)")
    p.add_argument("--window", default=DEFAULT_COMPLETION_WINDOW, choices=["24h", "1h"])
    p.add_argument("--batch-id", default=None, help="Download existing batch instead of submitting")
    p.add_argument("--eval-model", default=MATH500_EVAL_MODEL)
    p.add_argument("--max-tokens", type=int, default=4096)
    args = p.parse_args()

    run_smoke_test(
        eval_model=args.eval_model,
        n=args.n,
        input_path="data/baseline/multifin_baseline.csv",
        prompt_column="prompt",
        completion_window=args.window,
        eval_batch_id=args.batch_id,
        max_tokens=args.max_tokens,
        batch_root=MULTIFIN_BATCH_ROOT,
        no_think_prefix=False,
        label="multifin_baseline_eval",
    )


if __name__ == "__main__":
    main()
