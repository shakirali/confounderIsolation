"""
Load Last Letter Concatenation dataset from HuggingFace and save baseline CSV.

Dataset: yoonholee/last-letter-concatenation (train split, 1000 examples)
Each example: take the last letter of each word in a full name and concatenate them.

Usage (from repo root):
    uv run python src/load_last_letter.py
"""

import os

import pandas as pd
from datasets import load_dataset


def load_last_letter() -> pd.DataFrame:
    ds = load_dataset("yoonholee/last-letter-concatenation", split="train")
    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "unique_id": f"ll_{i:04d}",
            "full_name": ex["full_name"],
            "answer": ex["answer"],
            "prompt": ex["question"],
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs("data/baseline", exist_ok=True)
    df = load_last_letter()
    out_path = "data/baseline/last_letter_baseline.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    assert len(df) == 1000, f"Expected 1000 rows, got {len(df)}"
    assert df["answer"].notna().all(), "Some rows have missing answers"
    assert df["unique_id"].nunique() == len(df), "unique_id values are not unique"
    print(df.head(5).to_string())


if __name__ == "__main__":
    main()
