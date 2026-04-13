"""
MATH-500 baseline judge for Qwen3.5-397B eval responses.

Uses Nemotron as judge. Reads eval batch from math500_qwen batch root.
Judge batch saved under experiments/doubleword_batches/math500_qwen/.

Usage:
    .venv/bin/python src/doubledword/math500_qwen_baseline_judge.py --eval-batch-id <id>
    .venv/bin/python src/doubledword/math500_qwen_baseline_judge.py --eval-batch-id <id> --judge-batch-id <id>
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from doubleword_client import batch_dir, MATH500_QWEN_BATCH_ROOT
from math500_judge_core import score_math_jsonl, MATH_JUDGE_MODEL

load_dotenv()


def main():
    p = argparse.ArgumentParser(description="Score MATH-500 Qwen baseline eval via Nemotron judge")
    p.add_argument("--eval-batch-id", required=True, help="Completed Qwen baseline eval batch ID")
    p.add_argument("--judge-batch-id", default=None, help="Resume from a completed judge batch ID")
    p.add_argument("--judge-model", default=MATH_JUDGE_MODEL, help="Judge model (default: Nemotron)")
    p.add_argument("--window", default="24h", choices=["24h", "1h"])
    args = p.parse_args()

    bdir = batch_dir(args.eval_batch_id, "math500_baseline_eval", batch_root=MATH500_QWEN_BATCH_ROOT)
    eval_input_jsonl = os.path.join(bdir, "input.jsonl")
    eval_output_jsonl = os.path.join(bdir, "output.jsonl")

    score_math_jsonl(
        eval_input_jsonl=eval_input_jsonl,
        eval_output_jsonl=eval_output_jsonl,
        baseline=True,
        judge_model=args.judge_model,
        completion_window=args.window,
        judge_batch_id=args.judge_batch_id,
        label="math500_baseline_judge",
        batch_root=MATH500_QWEN_BATCH_ROOT,
    )


if __name__ == "__main__":
    main()
