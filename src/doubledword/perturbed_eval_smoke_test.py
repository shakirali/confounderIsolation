"""
Perturbed smoke test via the Doubleword Batch API.

Evaluates the first N questions across all perturbation types. Results are saved
to results/raw/. Score responses separately using judge_doubleword.py.

Usage:
    python src/doubledword/perturbed_smoke_test.py
    python src/doubledword/perturbed_smoke_test.py --n 50 --batch-id <id>
"""

import argparse
import os

import pandas as pd
from dotenv import load_dotenv

from doubleword_client import DEFAULT_COMPLETION_WINDOW, DEFAULT_MODEL, download_results, submit_batch

load_dotenv()


def run_perturbed_smoke_test(
    eval_model: str = DEFAULT_MODEL,
    n: int = 100,
    input_path: str = "data/perturbations/truthfulqa_perturbed.csv",
    output_path: str = "results/raw/responses_perturbed_smoke_test_doubleword.csv",
    completion_window: str = DEFAULT_COMPLETION_WINDOW,
    eval_batch_id: str | None = None,
):
    """
    Smoke test: evaluate the first N questions (all perturbation types) via Doubleword batch.

    Args:
        eval_model: Model to evaluate.
        n: Number of unique questions (default 100). Total rows = n × perturbation types.
        input_path: Perturbed prompts CSV.
        output_path: Output CSV path.
        completion_window: "24h" or "1h".
        eval_batch_id: If provided, skip eval submission and download from this completed batch ID.
    """
    full_df = pd.read_csv(input_path)
    question_ids = full_df["question_id"].unique()[:n]
    df = full_df[full_df["question_id"].isin(question_ids)].reset_index(drop=True)
    print(f"Perturbed smoke test: {len(df)} rows ({n} questions × {df['perturbation_type'].nunique()} perturbation types)")

    prompts = df["prompt_sent"].tolist()
    system_prompts = df["system_prompt"].where(df["system_prompt"].notna(), None).tolist()

    if eval_batch_id:
        print(f"Downloading eval results from existing batch: {eval_batch_id}")
        responses = download_results(eval_batch_id, len(df), label="eval")
    else:
        print(f"Querying eval model: {eval_model}")
        responses = submit_batch(
            prompts,
            model=eval_model,
            system_prompts=system_prompts,
            completion_window=completion_window,
            label="eval",
        )
    df["model"] = eval_model
    df["response"] = responses

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} responses → {output_path}")
    print("Score with: python src/doubledword/judge_doubleword.py --input <output_path> --output <scored_path>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perturbed smoke test via Doubleword batch")
    parser.add_argument("--eval-model", default=DEFAULT_MODEL)
    parser.add_argument("--window", default=DEFAULT_COMPLETION_WINDOW, choices=["24h", "1h"])
    parser.add_argument("--n", type=int, default=100, help="Number of unique questions")
    parser.add_argument("--batch-id", default=None, help="Resume from a completed eval batch ID")
    args = parser.parse_args()

    run_perturbed_smoke_test(
        eval_model=args.eval_model,
        n=args.n,
        completion_window=args.window,
        eval_batch_id=args.batch_id,
    )
