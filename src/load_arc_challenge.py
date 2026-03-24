"""
Load ARC-Challenge (test split) from HuggingFace and save baseline CSV.

Usage (from repo root):
    uv run python src/load_arc_challenge.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

_src = Path(__file__).resolve().parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from arc_prompts import choices_to_json, format_baseline_mcq


def load_arc_challenge_test() -> pd.DataFrame:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rows = []
    for i, ex in enumerate(ds):
        q = ex["question"]
        ch = {"label": list(ex["choices"]["label"]), "text": list(ex["choices"]["text"])}
        rows.append({
            "question_id": i,
            "arc_id": ex["id"],
            "question": q,
            "choices_json": choices_to_json(ch),
            "answerKey": ex["answerKey"],
            "prompt": format_baseline_mcq(q, ch),
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs("data/baseline", exist_ok=True)
    df = load_arc_challenge_test()
    out_path = "data/baseline/arc_challenge_test_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df.head(1).to_string())


if __name__ == "__main__":
    main()
