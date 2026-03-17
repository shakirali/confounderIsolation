"""
Generic judge script via the Doubleword Batch API.

Prefer baseline_judge_doubleword.py or perturbed_judge_doubleword.py for
standard runs — they derive eval batch paths automatically and use the
correct label.

Use this script when you have explicit eval input/output JSONL paths.
Scored CSV is saved as scored.csv inside the judge batch folder.

Usage:
    python src/doubledword/judge_doubleword.py \
        --eval-input-jsonl experiments/doubleword_batches/<batch_id>_eval/input.jsonl \
        --eval-output-jsonl experiments/doubleword_batches/<batch_id>_eval/output.jsonl \
        --label baseline_judge

    # Build judge input JSONL locally without submitting (for inspection)
    python src/doubledword/judge_doubleword.py \
        --eval-input-jsonl experiments/doubleword_batches/<batch_id>_eval/input.jsonl \
        --eval-output-jsonl experiments/doubleword_batches/<batch_id>_eval/output.jsonl \
        --build-only experiments/judge_input.jsonl

    # Resume from a completed judge batch
    python src/doubledword/judge_doubleword.py \
        --eval-input-jsonl experiments/doubleword_batches/<batch_id>_eval/input.jsonl \
        --eval-output-jsonl experiments/doubleword_batches/<batch_id>_eval/output.jsonl \
        --judge-batch-id <id>
"""

import argparse

from dotenv import load_dotenv

from doubleword_client import DEFAULT_COMPLETION_WINDOW
from judge_core import DEFAULT_JUDGE_MODEL, score_jsonl  # noqa: F401 — re-exported for callers

load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score responses via Doubleword judge batch")
    parser.add_argument("--eval-input-jsonl", required=True, help="Local eval batch input.jsonl")
    parser.add_argument("--eval-output-jsonl", required=True, help="Local eval batch output.jsonl")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--window", default=DEFAULT_COMPLETION_WINDOW, choices=["24h", "1h"])
    parser.add_argument("--judge-batch-id", default=None, help="Resume from a completed judge batch ID")
    parser.add_argument("--build-only", default=None, metavar="PATH", help="Build judge input JSONL to PATH without submitting")
    parser.add_argument("--label", default="judge", help="Batch directory label, e.g. baseline_judge or perturbed_judge")
    args = parser.parse_args()

    score_jsonl(
        eval_input_jsonl=args.eval_input_jsonl,
        eval_output_jsonl=args.eval_output_jsonl,
        judge_model=args.judge_model,
        completion_window=args.window,
        judge_batch_id=args.judge_batch_id,
        build_only=args.build_only,
        label=args.label,
    )
