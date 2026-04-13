#!/usr/bin/env python3
"""
Score Last Letter Concatenation Doubleword batch output vs gold answer (exact match, no judge).

Joins output.jsonl rows (by custom_id) to the same CSV rows used when the batch was built:
  - Baseline: first N rows of data/baseline/last_letter_baseline.csv (custom_id = row index).
  - Perturbed: first K unique_ids × 5 types from data/perturbations/last_letter_perturbed.csv.

Predicted answer extraction:
  - p1_format / p1_format_soft: try JSON {"answer": "..."} first, else use full content.
  - Other types: use full stripped content.

Scoring: correct=1 if predicted.lower() == gold.lower(), 0 if wrong, -1 if empty/parse failure.

Usage (repo root):
  .venv/bin/python scripts/score_last_letter.py \\
    --output-jsonl experiments/doubleword_batches/last_letter/<batch_id>_last_letter_baseline_eval/output.jsonl \\
    --baseline

  .venv/bin/python scripts/score_last_letter.py \\
    --output-jsonl experiments/doubleword_batches/last_letter/<batch_id>_last_letter_perturbed_eval/output.jsonl \\
    --perturbed --n-questions 1000

  .venv/bin/python scripts/score_last_letter.py ... --out-csv results/scored_last_letter.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
_BASELINE_CSV = _REPO / "data" / "baseline" / "last_letter_baseline.csv"
_PERTURBED_CSV = _REPO / "data" / "perturbations" / "last_letter_perturbed.csv"

JSON_TYPES = {"p1_format", "p1_format_soft"}


def strip_thinking(content: str) -> str:
    """Strip Qwen-style <think>...</think> reasoning block, return text after it."""
    if "</think>" in content:
        return content.split("</think>", 1)[1].strip()
    # Also handle plain "Thinking Process:\n...\n\n" preamble (no tags)
    if "Thinking Process:" in content:
        # Take everything after the last blank line following the thinking block
        parts = content.rsplit("\n\n", 1)
        if len(parts) == 2:
            return parts[1].strip()
    return content


def extract_answer(content: str, perturbation_type: str) -> str:
    """Extract the answer string from model response."""
    content = content.strip()
    if not content:
        return ""
    content = strip_thinking(content)
    if perturbation_type in JSON_TYPES:
        # Strip markdown code fences before JSON parsing
        stripped = content.strip("`").strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
        try:
            return json.loads(stripped).get("answer", content).strip()
        except (json.JSONDecodeError, AttributeError):
            pass
    # Strip "Answer: " prefix (from p5_fewshot few-shot format)
    if content.lower().startswith("answer:"):
        return content[len("answer:"):].strip()
    # Extract last **answer** bold pattern (from p2_complexity verbose explanations)
    bold_matches = re.findall(r"\*\*([^*]+)\*\*", content)
    if bold_matches:
        return bold_matches[-1].strip().rstrip(".")
    return content


def score_record(content: str, gold: str, perturbation_type: str) -> int:
    """Return 1 (correct), 0 (wrong), or -1 (empty/parse failure)."""
    if not content:
        return -1
    predicted = extract_answer(content, perturbation_type)
    if not predicted:
        return -1
    return 1 if predicted.lower() == gold.lower() else 0


def load_output(output_jsonl: str) -> dict[int, dict]:
    records = {}
    with open(output_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            cid = int(rec["custom_id"])
            choices = rec.get("response", {}).get("body", {}).get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = (msg.get("content") or "").strip()
                finish = choices[0].get("finish_reason")
            else:
                content = ""
                finish = "error"
            records[cid] = {"content": content, "finish_reason": finish}
    return records


def score_baseline(output_jsonl: str, n: int | None, out_csv: str | None) -> None:
    baseline_df = pd.read_csv(_BASELINE_CSV)
    if n:
        baseline_df = baseline_df.head(n)
    outputs = load_output(output_jsonl)

    rows = []
    for i, row in baseline_df.iterrows():
        cid = i
        out = outputs.get(cid, {"content": "", "finish_reason": "missing"})
        score = score_record(out["content"], row["answer"], "baseline")
        rows.append({
            "custom_id": cid,
            "unique_id": row["unique_id"],
            "full_name": row["full_name"],
            "gold": row["answer"],
            "predicted": extract_answer(out["content"], "baseline"),
            "finish_reason": out["finish_reason"],
            "correct": score,
        })

    df = pd.DataFrame(rows)
    valid = df[df["correct"] != -1]
    correct = (df["correct"] == 1).sum()
    wrong = (df["correct"] == 0).sum()
    errors = (df["correct"] == -1).sum()

    print(f"Baseline scoring — {len(df)} questions")
    print(f"  Correct:  {correct}")
    print(f"  Wrong:    {wrong}")
    print(f"  Errors:   {errors}")
    if len(valid):
        print(f"  Accuracy: {correct}/{len(valid)} = {correct/len(valid):.3f} (excluding errors)")

    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"  Saved → {out_csv}")


def score_perturbed(output_jsonl: str, n_questions: int, out_csv: str | None) -> None:
    full_df = pd.read_csv(_PERTURBED_CSV)
    uids = full_df["unique_id"].unique()[:n_questions]
    perturbed_df = full_df[full_df["unique_id"].isin(uids)].reset_index(drop=True)
    outputs = load_output(output_jsonl)

    rows = []
    for i, row in perturbed_df.iterrows():
        out = outputs.get(i, {"content": "", "finish_reason": "missing"})
        score = score_record(out["content"], row["answer"], row["perturbation_type"])
        rows.append({
            "custom_id": i,
            "unique_id": row["unique_id"],
            "full_name": row["full_name"],
            "perturbation_type": row["perturbation_type"],
            "gold": row["answer"],
            "predicted": extract_answer(out["content"], row["perturbation_type"]),
            "finish_reason": out["finish_reason"],
            "correct": score,
        })

    df = pd.DataFrame(rows)

    print(f"Perturbed scoring — {n_questions} questions × 5 types = {len(df)} rows")
    print(f"\n{'Perturbation':<18} {'Correct':>8} {'Wrong':>6} {'Errors':>7} {'Accuracy':>9}")
    print("-" * 52)

    for ptype in ["p1_format", "p1_format_soft", "p2_complexity", "p4_role", "p5_fewshot"]:
        sub = df[df["perturbation_type"] == ptype]
        correct = (sub["correct"] == 1).sum()
        wrong = (sub["correct"] == 0).sum()
        errors = (sub["correct"] == -1).sum()
        valid = correct + wrong
        acc = f"{correct/valid:.3f}" if valid else "N/A"
        print(f"{ptype:<18} {correct:>8} {wrong:>6} {errors:>7} {acc:>9}")

    valid_all = df[df["correct"] != -1]
    overall = (df["correct"] == 1).sum()
    print(f"\nOverall accuracy: {overall}/{len(valid_all)} = {overall/len(valid_all):.3f}")

    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"Saved → {out_csv}")


def main():
    p = argparse.ArgumentParser(description="Score Last Letter Concatenation eval output (exact match)")
    p.add_argument("--output-jsonl", required=True, help="Path to output.jsonl from eval batch")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baseline", action="store_true")
    mode.add_argument("--perturbed", action="store_true")
    p.add_argument("--n", type=int, default=None, help="Baseline: limit to first N rows")
    p.add_argument("--n-questions", type=int, default=1000, help="Perturbed: number of unique questions")
    p.add_argument("--out-csv", default=None, help="Optional path to save scored CSV")
    args = p.parse_args()

    if args.baseline:
        score_baseline(args.output_jsonl, args.n, args.out_csv)
    else:
        score_perturbed(args.output_jsonl, args.n_questions, args.out_csv)


if __name__ == "__main__":
    main()
