"""
Baseline smoke test via the Doubleword Batch API.

Evaluates the first N questions from TruthfulQA (no perturbations). Results are
saved to results/raw/. Score responses separately using judge_doubleword.py.

Usage:
    python src/doubledword/baseline_eval_smoke_test_doubleword.py
    python src/doubledword/baseline_eval_smoke_test_doubleword.py --batch-id <id>
"""

import argparse
import os

import pandas as pd
from dotenv import load_dotenv

from doubleword_client import DEFAULT_COMPLETION_WINDOW, DEFAULT_MODEL, download_results, submit_batch

load_dotenv()


def run_smoke_test(
    eval_model: str = DEFAULT_MODEL,
    n: int = 100,
    input_path: str = "data/baseline/truthfulqa_raw.csv",
    output_path: str = "results/raw/responses_smoke_test_doubleword.csv",
    completion_window: str = DEFAULT_COMPLETION_WINDOW,
    eval_batch_id: str | None = None,
):
    """
    Smoke test: evaluate the first N questions from TruthfulQA via Doubleword batch.

    Args:
        eval_model: Model to evaluate.
        n: Number of questions to evaluate (default 100).
        input_path: Raw TruthfulQA CSV.
        output_path: Output CSV path.
        completion_window: "24h" or "1h".
        eval_batch_id: If provided, skip eval submission and download from this completed batch ID.
    """
    df = pd.read_csv(input_path).head(n)
    print(f"Smoke test: {len(df)} questions from {input_path}")

    if eval_batch_id:
        print(f"Downloading eval results from existing batch: {eval_batch_id}")
        responses = download_results(eval_batch_id, len(df), label="eval")
    else:
        print(f"Querying eval model: {eval_model}")
        responses = submit_batch(df["question"].tolist(), model=eval_model, completion_window=completion_window, label="eval")
    df["perturbation_type"] = "baseline"
    df["prompt_sent"] = df["question"]
    df["system_prompt"] = None
    df["model"] = eval_model
    df["response"] = responses

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} responses → {output_path}")
    print("Score with: python src/doubledword/judge_doubleword.py --input <output_path> --output <scored_path>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline smoke test via Doubleword batch")
    parser.add_argument("--eval-model", default=DEFAULT_MODEL)
    parser.add_argument("--window", default=DEFAULT_COMPLETION_WINDOW, choices=["24h", "1h"])
    parser.add_argument("--n", type=int, default=100, help="Number of questions")
    parser.add_argument("--batch-id", default=None, help="Resume from a completed eval batch ID")
    args = parser.parse_args()

    run_smoke_test(
        eval_model=args.eval_model,
        n=args.n,
        completion_window=args.window,
        eval_batch_id=args.batch_id,
    )
