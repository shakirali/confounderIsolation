#!/usr/bin/env python3
"""
Score MultiFin Doubleword batch output vs gold answer_key (exact match, no judge).

Joins output.jsonl rows (by custom_id) to the same CSV rows used when the batch was built:
  - Baseline: first N rows of data/baseline/multifin_baseline.csv (custom_id = row index).
  - Perturbed: first K unique_ids × 5 types from data/perturbations/multifin_perturbed.csv.

Predicted letter extraction:
  - p1_format / p1_format_soft: try JSON {"answer": "A"} first, else fall back to letter extraction.
  - Other types: extract first standalone letter A–F from content.

Scoring: correct=1 if predicted==gold, 0 if wrong, -1 if empty/parse failure.

Usage (repo root):
  .venv/bin/python scripts/score_multifin.py \\
    --output-jsonl experiments/doubleword_batches/multifin/<batch_id>_multifin_baseline_eval/output.jsonl \\
    --baseline

  .venv/bin/python scripts/score_multifin.py \\
    --output-jsonl experiments/doubleword_batches/multifin/<batch_id>_multifin_perturbed_eval/output.jsonl \\
    --perturbed --n-questions 546

  .venv/bin/python scripts/score_multifin.py ... --out-csv results/scored_multifin.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
_BASELINE_CSV = _REPO / "data" / "baseline" / "multifin_baseline.csv"
_PERTURBED_CSV = _REPO / "data" / "perturbations" / "multifin_perturbed.csv"

# A–F: MultiFin has 6 fixed options
LETTER_RE = re.compile(r"\b([A-F])\b", re.IGNORECASE)
JSON_ANSWER_RE = re.compile(r'"answer"\s*:\s*"([A-Fa-f])"', re.IGNORECASE)

JSON_TYPES = {"p1_format", "p1_format_soft"}


def strip_thinking(content: str) -> str:
    """Strip Qwen-style <think>...</think> reasoning block, return text after it."""
    if "</think>" in content:
        return content.split("</think>", 1)[1].strip()
    if "Thinking Process:" in content:
        parts = content.rsplit("\n\n", 1)
        if len(parts) == 2:
            return parts[1].strip()
    return content


def extract_letter(content: str, perturbation_type: str) -> str:
    """Extract predicted letter (A–F) from model response."""
    content = content.strip()
    if not content:
        return ""
    content = strip_thinking(content)

    if perturbation_type in JSON_TYPES:
        m = JSON_ANSWER_RE.search(content)
        if m:
            return m.group(1).upper()
        # Try full JSON parse
        try:
            obj = json.loads(content)
            ans = str(obj.get("answer", "")).strip().upper()
            if ans in set("ABCDEF"):
                return ans
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fall back: last standalone letter A–F in content
    matches = LETTER_RE.findall(content)
    if matches:
        return matches[-1].upper()
    return ""


def score_record(content: str, gold: str, perturbation_type: str) -> int:
    if not content:
        return -1
    predicted = extract_letter(content, perturbation_type)
    if not predicted:
        return -1
    return 1 if predicted == gold.upper() else 0


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
        out = outputs.get(i, {"content": "", "finish_reason": "missing"})
        score = score_record(out["content"], row["answer_key"], "baseline")
        rows.append({
            "custom_id": i,
            "unique_id": row["unique_id"],
            "text": row["text"],
            "label": row["label"],
            "gold": row["answer_key"],
            "predicted": extract_letter(out["content"], "baseline"),
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
        score = score_record(out["content"], row["answer_key"], row["perturbation_type"])
        rows.append({
            "custom_id": i,
            "unique_id": row["unique_id"],
            "text": row["text"],
            "label": row["label"],
            "perturbation_type": row["perturbation_type"],
            "gold": row["answer_key"],
            "predicted": extract_letter(out["content"], row["perturbation_type"]),
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
    p = argparse.ArgumentParser(description="Score MultiFin eval output (MCQ letter match)")
    p.add_argument("--output-jsonl", required=True, help="Path to output.jsonl from eval batch")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baseline", action="store_true")
    mode.add_argument("--perturbed", action="store_true")
    p.add_argument("--n", type=int, default=None, help="Baseline: limit to first N rows")
    p.add_argument("--n-questions", type=int, default=546, help="Perturbed: number of unique questions")
    p.add_argument("--out-csv", default=None, help="Optional path to save scored CSV")
    args = p.parse_args()

    if args.baseline:
        score_baseline(args.output_jsonl, args.n, args.out_csv)
    else:
        score_perturbed(args.output_jsonl, args.n_questions, args.out_csv)


if __name__ == "__main__":
    main()
