"""
Baseline judge via the Doubleword Batch API.

Scores baseline eval responses as truthful (1) or not (0) using a judge model.
Batch input/output saved to experiments/doubleword_batches/<batch_id>_baseline_judge/.

Usage:
    python src/doubledword/baseline_judge_doubleword.py --eval-batch-id <id>
    python src/doubledword/baseline_judge_doubleword.py --eval-batch-id <id> --judge-batch-id <id>
    python src/doubledword/baseline_judge_doubleword.py --eval-batch-id <id> --build-only experiments/judge_input_baseline.jsonl
"""

import argparse
import os

from dotenv import load_dotenv

from doubleword_client import DEFAULT_COMPLETION_WINDOW, batch_dir
from judge_core import DEFAULT_JUDGE_MODEL, score_jsonl

load_dotenv()


def run_baseline_judge(
    eval_batch_id: str,
    output_path: str = "experiments/results/raw/baseline_scored.csv",
    judge_model: str = DEFAULT_JUDGE_MODEL,
    completion_window: str = DEFAULT_COMPLETION_WINDOW,
    judge_batch_id: str | None = None,
    build_only: str | None = None,
):
    """
    Score baseline eval responses via Doubleword judge batch.

    Args:
        eval_batch_id: Batch ID of the completed baseline eval run.
        output_path: Output scored CSV path.
        judge_model: Model to use as truthfulness judge.
        completion_window: "24h" or "1h".
        judge_batch_id: If provided, skip submission and download from this completed batch ID.
        build_only: If set, write judge input JSONL to this path without submitting.
    """
    bdir = batch_dir(eval_batch_id, "baseline_eval")
    eval_input_jsonl = os.path.join(bdir, "input.jsonl")
    eval_output_jsonl = os.path.join(bdir, "output.jsonl")

    score_jsonl(
        eval_input_jsonl=eval_input_jsonl,
        eval_output_jsonl=eval_output_jsonl,
        output_path=output_path,
        judge_model=judge_model,
        completion_window=completion_window,
        judge_batch_id=judge_batch_id,
        build_only=build_only,
        label="baseline_judge",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score baseline eval responses via Doubleword judge batch")
    parser.add_argument("--eval-batch-id", required=True, help="Completed baseline eval batch ID")
    parser.add_argument("--output", default="experiments/results/raw/baseline_scored.csv", help="Output scored CSV path")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--window", default=DEFAULT_COMPLETION_WINDOW, choices=["24h", "1h"])
    parser.add_argument("--judge-batch-id", default=None, help="Resume from a completed judge batch ID")
    parser.add_argument("--build-only", default=None, metavar="PATH", help="Build judge input JSONL to PATH without submitting")
    args = parser.parse_args()

    run_baseline_judge(
        eval_batch_id=args.eval_batch_id,
        output_path=args.output,
        judge_model=args.judge_model,
        completion_window=args.window,
        judge_batch_id=args.judge_batch_id,
        build_only=args.build_only,
    )
