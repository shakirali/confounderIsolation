"""
Perturbed judge via the Doubleword Batch API.

Scores perturbed eval responses as truthful (1) or not (0) using a judge model.
Batch input/output saved to experiments/doubleword_batches/<batch_id>_perturbed_judge/.

Usage:
    python src/doubledword/perturbed_judge_doubleword.py --eval-batch-id <id>
    python src/doubledword/perturbed_judge_doubleword.py --eval-batch-id <id> --judge-batch-id <id>
    python src/doubledword/perturbed_judge_doubleword.py --eval-batch-id <id> --build-only experiments/judge_input_perturbed.jsonl
"""

import argparse
import os

from dotenv import load_dotenv

from doubleword_client import batch_dir
from judge_core import score_jsonl

load_dotenv()


def run_perturbed_judge(
    eval_batch_id: str,
    judge_batch_id: str | None = None,
):
    """
    Score perturbed eval responses via Doubleword judge batch.

    Builds input.jsonl into pending_perturbed_judge/, prompts for inspection,
    then submits. Results saved in the judge batch folder.

    Args:
        eval_batch_id: Batch ID of the completed perturbed eval run.
        judge_batch_id: If provided, skip build/submit and download from this completed batch ID.
    """
    bdir = batch_dir(eval_batch_id, "perturbed_eval")
    eval_input_jsonl = os.path.join(bdir, "input.jsonl")
    eval_output_jsonl = os.path.join(bdir, "output.jsonl")

    score_jsonl(
        eval_input_jsonl=eval_input_jsonl,
        eval_output_jsonl=eval_output_jsonl,
        judge_batch_id=judge_batch_id,
        label="perturbed_judge",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score perturbed eval responses via Doubleword judge batch")
    parser.add_argument("--eval-batch-id", required=True, help="Completed perturbed eval batch ID")
    parser.add_argument("--judge-batch-id", default=None, help="Resume from a completed judge batch ID")
    args = parser.parse_args()

    run_perturbed_judge(
        eval_batch_id=args.eval_batch_id,
        judge_batch_id=args.judge_batch_id,
    )
