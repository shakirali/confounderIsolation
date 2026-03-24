"""
Baseline smoke test via the Doubleword Batch API.

Evaluates the first N questions from TruthfulQA (no perturbations). Batch input
and output are saved to experiments/doubleword_batches/<batch_id>_eval/.
Score responses separately using judge_doubleword.py.

Usage:
    python src/doubledword/baseline_eval_smoke_test_doubleword.py
    python src/doubledword/baseline_eval_smoke_test_doubleword.py --n 100 --batch-id <id>
"""

import argparse

import pandas as pd
from dotenv import load_dotenv

from doubleword_client import (
    ARC_BATCH_ROOT,
    DEFAULT_BATCH_ROOT,
    DEFAULT_COMPLETION_WINDOW,
    DEFAULT_MODEL,
    download_results,
    submit_batch,
)

load_dotenv()


def run_smoke_test(
    eval_model: str = DEFAULT_MODEL,
    n: int = 100,
    input_path: str = "data/baseline/truthfulqa_raw.csv",
    prompt_column: str = "question",
    completion_window: str = DEFAULT_COMPLETION_WINDOW,
    eval_batch_id: str | None = None,
    max_tokens: int = 4096,
    batch_root: str | None = None,
    no_think_prefix: bool = False,
    label: str = "baseline_eval",
):
    """
    Smoke test: evaluate the first N rows from a baseline CSV via Doubleword batch.

    Batch input/output JSONL saved under <batch_root>/<batch_id>_<label>/.

    Args:
        eval_model: Model to evaluate.
        n: Number of rows (default 100).
        input_path: Baseline CSV (TruthfulQA: `question`; ARC: `prompt`).
        prompt_column: Column to send as the user message.
        completion_window: "24h" or "1h".
        eval_batch_id: If set, skip submit and download this completed batch.
        max_tokens: Generation cap (ARC entrypoints default 4096; override with --max-tokens).
        batch_root: Parent of batch folders; default `experiments/doubleword_batches`.
        no_think_prefix: If True, prefixes `/no_think` for models that use it (e.g. Qwen); Nemotron never gets the line.
        label: Subfolder suffix after batch UUID.
    """
    root = batch_root if batch_root is not None else DEFAULT_BATCH_ROOT
    df = pd.read_csv(input_path).head(n)
    if prompt_column not in df.columns:
        raise ValueError(f"Column {prompt_column!r} not in {input_path}; columns: {list(df.columns)}")
    print(f"Smoke test: {len(df)} rows from {input_path} (column {prompt_column!r}, batch_root={root})")

    if eval_batch_id:
        print(f"Downloading eval results from existing batch: {eval_batch_id}")
        download_results(eval_batch_id, len(df), label=label, batch_root=batch_root)
        batch_id = eval_batch_id
    else:
        print(f"Querying eval model: {eval_model}")
        _, batch_id = submit_batch(
            df[prompt_column].tolist(),
            model=eval_model,
            completion_window=completion_window,
            max_tokens=max_tokens,
            no_think_prefix=no_think_prefix,
            label=label,
            batch_root=batch_root,
        )

    batch_rel = f"{root}/{batch_id}_{label}"
    print(f"\nBatch ID: {batch_id}")
    print(f"JSONL: {batch_rel}/input.jsonl , {batch_rel}/output.jsonl")
    print("\nTruthfulQA: score with judge_doubleword.py. ARC: use scripts/score_arc_mcq.py when available.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline smoke test via Doubleword batch")
    parser.add_argument("--eval-model", default=DEFAULT_MODEL)
    parser.add_argument("--window", default=DEFAULT_COMPLETION_WINDOW, choices=["24h", "1h"])
    parser.add_argument("--n", type=int, default=100, help="Number of questions")
    parser.add_argument("--batch-id", default=None, help="Resume from a completed eval batch ID")
    parser.add_argument("--input-csv", default=None, help="Override baseline CSV path")
    parser.add_argument("--prompt-column", default="question", help="CSV column for user message (ARC: prompt)")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--batch-root",
        default=None,
        help=f"Batch parent dir (ARC: {ARC_BATCH_ROOT})",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Prepend /no_think to the user message for Qwen-style models only (opt-in; default is raw prompt)",
    )
    parser.add_argument("--label", default="baseline_eval", help="Batch folder label suffix")
    args = parser.parse_args()

    run_smoke_test(
        eval_model=args.eval_model,
        n=args.n,
        input_path=args.input_csv or "data/baseline/truthfulqa_raw.csv",
        prompt_column=args.prompt_column,
        completion_window=args.window,
        eval_batch_id=args.batch_id,
        max_tokens=args.max_tokens,
        batch_root=args.batch_root,
        no_think_prefix=args.no_think,
        label=args.label,
    )
