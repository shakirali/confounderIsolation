"""
MATH-500 perturbed judge via Doubleword Batch API.

Scores perturbed eval responses as correct (1) or incorrect (0) using Nemotron as judge.
Batch input/output saved to experiments/doubleword_batches/math500/<batch_id>_math500_perturbed_judge/.

Usage:
    .venv/bin/python src/doubledword/math500_perturbed_judge.py --eval-batch-id <id> --n-questions 10
    .venv/bin/python src/doubledword/math500_perturbed_judge.py --eval-batch-id <id> --n-questions 500 --judge-batch-id <id>
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from doubleword_client import batch_dir, MATH500_BATCH_ROOT
from math500_judge_core import score_math_jsonl, MATH_JUDGE_MODEL

load_dotenv()


def main():
    p = argparse.ArgumentParser(description="Score MATH-500 perturbed eval via Nemotron judge")
    p.add_argument("--eval-batch-id", required=True, help="Completed perturbed eval batch ID")
    p.add_argument("--n-questions", type=int, required=True,
                   help="Number of unique problems in the eval run (must match eval --n)")
    p.add_argument("--judge-batch-id", default=None, help="Resume from a completed judge batch ID")
    p.add_argument("--judge-model", default=MATH_JUDGE_MODEL)
    p.add_argument("--window", default="24h", choices=["24h", "1h"])
    args = p.parse_args()

    bdir = batch_dir(args.eval_batch_id, "math500_perturbed_eval", batch_root=MATH500_BATCH_ROOT)
    eval_input_jsonl = os.path.join(bdir, "input.jsonl")
    eval_output_jsonl = os.path.join(bdir, "output.jsonl")

    score_math_jsonl(
        eval_input_jsonl=eval_input_jsonl,
        eval_output_jsonl=eval_output_jsonl,
        baseline=False,
        n_questions=args.n_questions,
        judge_model=args.judge_model,
        completion_window=args.window,
        judge_batch_id=args.judge_batch_id,
        label="math500_perturbed_judge",
    )


if __name__ == "__main__":
    main()
