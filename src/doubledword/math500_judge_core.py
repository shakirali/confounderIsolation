"""
Shared judge logic for MATH-500 Doubleword batch scoring.

Mirrors judge_core.py but uses a math-specific prompt that includes the gold answer.
The judge model (Nemotron) decides whether the student's final answer is mathematically
correct given the problem and gold answer.

Judge scores:
  1  — correct
  0  — incorrect
  -1 — parse error / empty response (excluded from mean)
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

_src = Path(__file__).resolve().parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from doubleword_client import (
    MATH500_EVAL_MODEL,
    MATH500_BATCH_ROOT,
    DEFAULT_COMPLETION_WINDOW,
    batch_dir,
    submit_batch_from_file,
    download_results,
    model_uses_no_think_user_prefix,
)

MATH_JUDGE_MODEL = MATH500_EVAL_MODEL  # Nemotron — same model as eval

MATH_JUDGE_PROMPT = """You are a math grader. Given a problem, the gold answer, and a student's response, determine whether the student's final answer is mathematically equivalent to the gold answer.

Problem: {problem}

Gold answer: {gold_answer}

Student response: {response}

Does the student's final answer equal the gold answer (mathematically)? Answer in JSON format: {{"correct": 1}} if correct or {{"correct": 0}} if not."""

_REPO = Path(__file__).resolve().parent.parent.parent
_BASELINE_CSV = _REPO / "data" / "baseline" / "math500_baseline.csv"
_PERTURBED_CSV = _REPO / "data" / "perturbations" / "math500_perturbed.csv"


def parse_scores(raw_results: list[str]) -> list[int]:
    """Parse judge responses into scores (1=correct, 0=incorrect, -1=parse error)."""
    scores = []
    for result in raw_results:
        if not result or result == "[ERROR]":
            scores.append(-1)
            continue
        try:
            val = int(json.loads(result)["correct"])
            scores.append(val if val in (0, 1) else -1)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            scores.append(-1)
    return scores


def load_math_jsonl_pairs(
    input_jsonl: str,
    output_jsonl: str,
    baseline: bool = True,
    n_questions: int | None = None,
) -> pd.DataFrame:
    """Load eval input/output JSONL and join with gold answers from CSV.

    Returns DataFrame with columns: custom_id, problem, gold_answer, response, model.
    """
    # Load eval inputs (custom_id → prompt sent)
    inputs = {}
    with open(input_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            inputs[int(rec["custom_id"])] = {
                "model": rec["body"]["model"],
            }

    # Load eval outputs (custom_id → response content)
    outputs = {}
    with open(output_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            msg = rec["response"]["body"]["choices"][0]["message"]
            response = (msg.get("content") or "").strip() or "[ERROR]"
            outputs[int(rec["custom_id"])] = response

    # Load gold answers — align by custom_id (= row position in eval CSV slice)
    if baseline:
        gold_df = pd.read_csv(_BASELINE_CSV).head(len(inputs)).reset_index(drop=True)
        gold_df["custom_id"] = range(len(gold_df))
    else:
        full = pd.read_csv(_PERTURBED_CSV)
        uids = full["unique_id"].unique()[:n_questions]
        gold_df = full[full["unique_id"].isin(uids)].reset_index(drop=True)
        gold_df["custom_id"] = range(len(gold_df))

    rows = []
    for _, gr in gold_df.iterrows():
        cid = int(gr["custom_id"])
        rows.append({
            "custom_id": cid,
            "problem": gr.get("problem", gr.get("prompt_sent", "")),
            "gold_answer": str(gr["answer"]).strip(),
            "model": inputs.get(cid, {}).get("model", ""),
            "response": outputs.get(cid, "[ERROR]"),
        })

    return pd.DataFrame(rows)


def build_judge_input(
    df: pd.DataFrame,
    output_path: str,
    judge_model: str = MATH_JUDGE_MODEL,
) -> None:
    """Build judge batch input JSONL. Skips [ERROR] eval responses."""
    lines = []
    error_count = 0
    for _, row in df.iterrows():
        if row["response"] == "[ERROR]":
            error_count += 1
            continue
        prompt = MATH_JUDGE_PROMPT.format(
            problem=row["problem"],
            gold_answer=row["gold_answer"],
            response=row["response"],
        )
        if model_uses_no_think_user_prefix(judge_model):
            prompt = f"/no_think\n{prompt}"
        lines.append(json.dumps({
            "custom_id": str(row["custom_id"]),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": judge_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 4096,
            },
        }))

    if error_count:
        print(f"Skipping {error_count} [ERROR] eval responses — will be scored -1.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {len(lines)} judge prompts → {output_path}")


def run_judge(
    df: pd.DataFrame,
    judge_model: str = MATH_JUDGE_MODEL,
    completion_window: str = DEFAULT_COMPLETION_WINDOW,
    judge_batch_id: str | None = None,
    label: str = "math500_baseline_judge",
) -> tuple[list[int], str]:
    """Score eval responses using Nemotron judge via Doubleword batch.

    Returns (scores, judge_batch_dir). scores indexed by custom_id (len = max_custom_id + 1).
    """
    num_requests = len(df)

    if judge_batch_id:
        print(f"Downloading judge results from existing batch: {judge_batch_id}")
        raw_results = download_results(judge_batch_id, num_requests, label=label)
        bdir = batch_dir(judge_batch_id, label)
        return parse_scores(raw_results), bdir

    pending_dir = os.path.join(MATH500_BATCH_ROOT, f"pending_{label}")
    input_path = os.path.join(pending_dir, "input.jsonl")
    build_judge_input(df, input_path, judge_model)

    print(f"\nInspect input.jsonl at: {input_path}")
    input("Press Enter to submit the judge batch (Ctrl+C to cancel)...")

    raw_results, submitted_batch_id = submit_batch_from_file(
        input_jsonl_path=input_path,
        num_requests=num_requests,
        completion_window=completion_window,
        content_only=True,
        label=label,
        batch_root=MATH500_BATCH_ROOT,
    )
    bdir = batch_dir(submitted_batch_id, label, batch_root=MATH500_BATCH_ROOT)
    return parse_scores(raw_results), bdir


def score_math_jsonl(
    eval_input_jsonl: str,
    eval_output_jsonl: str,
    baseline: bool = True,
    n_questions: int | None = None,
    judge_model: str = MATH_JUDGE_MODEL,
    completion_window: str = DEFAULT_COMPLETION_WINDOW,
    judge_batch_id: str | None = None,
    label: str = "math500_baseline_judge",
) -> None:
    """Score MATH-500 eval responses via Nemotron judge batch.

    Loads (problem, gold_answer, response) triples, submits judge batch,
    prints score summary.
    """
    df = load_math_jsonl_pairs(
        eval_input_jsonl, eval_output_jsonl,
        baseline=baseline, n_questions=n_questions,
    )
    print(f"Loaded {len(df)} problem/response pairs.")

    scores, bdir = run_judge(
        df,
        judge_model=judge_model,
        completion_window=completion_window,
        judge_batch_id=judge_batch_id,
        label=label,
    )
    df["score"] = scores

    valid = df[df["score"] != -1]
    print(f"\nScore distribution:\n{df['score'].value_counts().sort_index().to_string()}")
    if len(valid):
        print(f"Accuracy: {valid['score'].mean():.3f} ({valid['score'].sum()}/{len(valid)}, excluding parse errors)")
    print(f"\nResults saved → {bdir}")
