"""
Judge the stable eval dataset.

Reads data/stable_eval.csv (2,806 stable question/response pairs from run 1),
strips perturbation artifacts from questions, builds judge input JSONL, and
submits as a 24h batch.

Usage:
    PYTHONPATH=src/doubledword python scripts/judge_stable_eval.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'doubledword'))

import pandas as pd
from judge_core import build_judge_input, DEFAULT_JUDGE_MODEL
from doubleword_client import submit_batch_from_file, batch_dir

# Original eval had 4085 pairs (817 × 5); size results array by this so
# custom_ids from stable_eval.csv align correctly with eval rows.
TOTAL_EVAL_PAIRS = 4085


def strip_question(prompt_sent: str, perturbation_type: str) -> str:
    """Strip perturbation artifacts to get the plain question for the judge."""
    q = prompt_sent
    if perturbation_type == 'p1_format_soft':
        suffix = '\nAnswer in JSON format: {"answer": "your answer here"}'
        if q.endswith(suffix):
            q = q[:-len(suffix)]
    elif perturbation_type == 'p2_complexity':
        marker = "The question is: "
        if marker in q:
            q = q.split(marker, 1)[-1]
    elif perturbation_type == 'p5_fewshot':
        q = q.rsplit("\nQ: ", 1)[-1]
    # p1_format: plain question, no stripping needed
    # p4_role: system prompt is separate, user prompt is plain question
    return q


def main():
    df = pd.read_csv("data/stable_eval.csv")
    print(f"Loaded {len(df)} stable pairs from data/stable_eval.csv")
    print(f"Perturbation type counts:\n{df['perturbation_type'].value_counts().to_string()}\n")

    questions = [strip_question(row.prompt_sent, row.perturbation_type) for row in df.itertuples()]
    responses = df["response"].tolist()
    custom_ids = df["custom_id"].tolist()

    pending_dir = os.path.join("experiments", "doubleword_batches", "pending_stable_judge")
    input_path = os.path.join(pending_dir, "input.jsonl")
    build_judge_input(questions, responses, input_path, DEFAULT_JUDGE_MODEL, custom_ids=custom_ids)

    raw_results, submitted_batch_id = submit_batch_from_file(
        input_jsonl_path=input_path,
        num_requests=TOTAL_EVAL_PAIRS,
        completion_window="24h",
        content_only=True,
        label="stable_judge",
    )

    bdir = batch_dir(submitted_batch_id, "stable_judge")
    print(f"\nBatch complete. Results saved → {bdir}")
    print(f"Judge batch ID: {submitted_batch_id}")


if __name__ == "__main__":
    main()
