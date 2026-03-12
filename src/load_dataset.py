import os
import pandas as pd
from datasets import load_dataset


def load_truthfulqa() -> pd.DataFrame:
    dataset = load_dataset("truthful_qa", "generation")
    split = dataset["validation"]
    return pd.DataFrame({
        "question_id": range(len(split)),
        "question": split["question"],
        "best_answer": split["best_answer"],
        "correct_answers": ["; ".join(a) for a in split["correct_answers"]],
        "incorrect_answers": ["; ".join(a) for a in split["incorrect_answers"]],
    })


def main():
    os.makedirs("data/baseline", exist_ok=True)
    df = load_truthfulqa()
    out_path = "data/baseline/truthfulqa_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} questions to {out_path}")


if __name__ == "__main__":
    main()
