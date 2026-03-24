"""
Judge the full 817-question baseline eval.

Reads baseline eval input/output JSONL, filters to non-empty responses,
builds judge input, and submits as a 24h batch.

Usage:
    PYTHONPATH=src/doubledword python scripts/judge_baseline_full.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'doubledword'))

from judge_core import build_judge_input, DEFAULT_JUDGE_MODEL
from doubleword_client import submit_batch_from_file, batch_dir

EVAL_BATCH_ID = "ddbdabb1-97c3-49dc-b1fc-702d77175ef0"
TOTAL_QUESTIONS = 817


def main():
    input_jsonl = f"experiments/doubleword_batches/{EVAL_BATCH_ID}_baseline_eval/input.jsonl"
    output_jsonl = f"experiments/doubleword_batches/{EVAL_BATCH_ID}_baseline_eval/output.jsonl"

    # Load questions
    questions = {}
    with open(input_jsonl) as f:
        for line in f:
            r = json.loads(line)
            cid = int(r["custom_id"])
            user_msg = next(m["content"] for m in r["body"]["messages"] if m["role"] == "user")
            questions[cid] = user_msg

    # Load responses, skip empty
    responses = {}
    empty = []
    with open(output_jsonl) as f:
        for line in f:
            r = json.loads(line)
            cid = int(r["custom_id"])
            content = r["response"]["body"]["choices"][0]["message"].get("content") or ""
            if content:
                responses[cid] = content
            else:
                empty.append(cid)

    print(f"Total questions:  {TOTAL_QUESTIONS}")
    print(f"Valid responses:  {len(responses)}")
    print(f"Empty (skipped):  {len(empty)}")

    valid_cids = sorted(responses.keys())
    valid_questions = [questions[cid] for cid in valid_cids]
    valid_responses = [responses[cid] for cid in valid_cids]

    pending_dir = os.path.join("experiments", "doubleword_batches", "pending_baseline_full_judge")
    input_path = os.path.join(pending_dir, "input.jsonl")
    build_judge_input(valid_questions, valid_responses, input_path, DEFAULT_JUDGE_MODEL, custom_ids=valid_cids)

    raw_results, submitted_batch_id = submit_batch_from_file(
        input_jsonl_path=input_path,
        num_requests=TOTAL_QUESTIONS,
        completion_window="24h",
        content_only=True,
        label="baseline_full_judge",
    )

    bdir = batch_dir(submitted_batch_id, "baseline_full_judge")
    print(f"\nBatch complete. Results saved → {bdir}")
    print(f"Judge batch ID: {submitted_batch_id}")


if __name__ == "__main__":
    main()
