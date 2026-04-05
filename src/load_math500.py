"""
Load MATH-500 (test split) from HuggingFace and save baseline CSV.

Usage (from repo root):
    uv run python src/load_math500.py
"""

import os

import pandas as pd
from datasets import load_dataset

PROMPT_TEMPLATE = (
    "Solve the following math problem. "
    "Show your reasoning, then state your final answer clearly.\n\n{problem}"
)


def load_math500() -> pd.DataFrame:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = []
    for ex in ds:
        rows.append({
            "unique_id": ex["unique_id"],
            "problem": ex["problem"],
            "answer": ex["answer"],
            "subject": ex["subject"],
            "level": ex["level"],
            "prompt": PROMPT_TEMPLATE.format(problem=ex["problem"]),
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs("data/baseline", exist_ok=True)
    df = load_math500()
    out_path = "data/baseline/math500_baseline.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    assert len(df) == 500, f"Expected 500 rows, got {len(df)}"
    assert df["answer"].notna().all(), "Some rows have missing answers"
    assert df["unique_id"].nunique() == len(df), "unique_id values are not unique"
    print(df.head(2).to_string())


if __name__ == "__main__":
    main()
