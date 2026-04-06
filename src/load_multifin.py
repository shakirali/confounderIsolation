"""
Load MultiFin (English test rows) from HuggingFace and save baseline CSV.

Dataset: awinml/MultiFin, config=all_languages_highlevel, split=test, lang=English
Task: 6-class MCQ — classify financial article headline into one of 6 topics.
Labels are fixed across all examples (A–F), so answer_key is deterministic.

n=546 (all English rows in the test split).

Usage (from repo root):
    uv run python src/load_multifin.py
"""

import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset

_REPO = Path(__file__).resolve().parent.parent

# Fixed label set — alphabetically sorted, assigned letters A–F
LABELS = [
    "Business & Management",
    "Finance",
    "Government & Controls",
    "Industry",
    "Tax & Accounting",
    "Technology",
]
LETTERS = list("ABCDEF")
LABEL_TO_LETTER = {label: letter for label, letter in zip(LABELS, LETTERS)}
LETTER_TO_LABEL = {v: k for k, v in LABEL_TO_LETTER.items()}

_OPTIONS_BLOCK = "\n".join(f"{l}. {lab}" for l, lab in zip(LETTERS, LABELS))
_LETTERS_STR = ", ".join(LETTERS)


def build_prompt(headline: str) -> str:
    return (
        "Classify the following financial article headline into one of the categories below.\n"
        "\n"
        f'Headline: "{headline}"\n'
        "\n"
        "Categories:\n"
        f"{_OPTIONS_BLOCK}\n"
        "\n"
        f"Respond with only the letter ({_LETTERS_STR})."
    )


def load_multifin() -> pd.DataFrame:
    ds = load_dataset("awinml/MultiFin", "all_languages_highlevel", split="test")
    df = ds.to_pandas()
    eng = df[df["lang"] == "English"].reset_index(drop=True)

    rows = []
    for i, row in eng.iterrows():
        label = row["label"]
        rows.append({
            "unique_id": f"mf_{i:04d}",
            "source_id": row["id"],
            "text": row["text"],
            "label": label,
            "answer_key": LABEL_TO_LETTER[label],
            "prompt": build_prompt(row["text"]),
        })

    return pd.DataFrame(rows)


def main():
    os.makedirs("data/baseline", exist_ok=True)
    df = load_multifin()
    out_path = "data/baseline/multifin_baseline.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

    # Verify
    assert df["unique_id"].nunique() == len(df), "Duplicate unique_ids"
    assert df["answer_key"].notna().all(), "Missing answer_keys"
    assert set(df["label"].unique()) == set(LABELS), f"Unexpected labels: {df['label'].unique()}"
    print("Verification passed.")

    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())
    print("\nSample rows:")
    print(df[["unique_id", "text", "label", "answer_key"]].head(5).to_string())
    print("\nSample prompt:\n")
    print(df["prompt"].iloc[0])


if __name__ == "__main__":
    main()
