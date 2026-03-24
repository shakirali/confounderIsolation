"""
Perturbed smoke test via the Doubleword Batch API.

Evaluates the first N questions across all perturbation types. Batch input and
output are saved to experiments/doubleword_batches/<batch_id>_eval/.
Score responses separately using judge_doubleword.py.

Usage:
    python src/doubledword/perturbed_eval_smoke_test.py
    python src/doubledword/perturbed_eval_smoke_test.py --n 100 --batch-id <id>
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


def run_perturbed_smoke_test(
    eval_model: str = DEFAULT_MODEL,
    n: int = 100,
    input_path: str = "data/perturbations/truthfulqa_perturbed.csv",
    prompt_column: str = "prompt_sent",
    completion_window: str = DEFAULT_COMPLETION_WINDOW,
    eval_batch_id: str | None = None,
    max_tokens: int = 4096,
    batch_root: str | None = None,
    no_think_prefix: bool = False,
    label: str = "perturbed_eval",
):
    """
    Smoke test: evaluate the first N question_ids (all perturbation types) via Doubleword batch.

    Batch input/output JSONL saved under <batch_root>/<batch_id>_<label>/.

    Args:
        prompt_column: Column with the user message (default `prompt_sent`).
        batch_root: Parent of batch folders (ARC: `ARC_BATCH_ROOT`).
        no_think_prefix: Opt-in `/no_think` line for Qwen-style models only (default False).
        label: Subfolder suffix after batch UUID.
    """
    root = batch_root if batch_root is not None else DEFAULT_BATCH_ROOT
    full_df = pd.read_csv(input_path)
    if prompt_column not in full_df.columns:
        raise ValueError(f"Column {prompt_column!r} not in {input_path}; columns: {list(full_df.columns)}")
    question_ids = full_df["question_id"].unique()[:n]
    df = full_df[full_df["question_id"].isin(question_ids)].reset_index(drop=True)
    print(
        f"Perturbed smoke test: {len(df)} rows ({n} questions × {df['perturbation_type'].nunique()} types) "
        f"from {input_path} (batch_root={root})"
    )

    prompts = df[prompt_column].tolist()
    system_prompts = [s if pd.notna(s) else None for s in df["system_prompt"].tolist()]
    response_formats = [
        {"type": "json_object"} if t == "p1_format" else None
        for t in df["perturbation_type"].tolist()
    ]

    if eval_batch_id:
        print(f"Downloading eval results from existing batch: {eval_batch_id}")
        download_results(eval_batch_id, len(df), label=label, batch_root=batch_root)
        batch_id = eval_batch_id
    else:
        print(f"Querying eval model: {eval_model}")
        _, batch_id = submit_batch(
            prompts,
            model=eval_model,
            system_prompts=system_prompts,
            response_formats=response_formats,
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
    parser = argparse.ArgumentParser(description="Perturbed smoke test via Doubleword batch")
    parser.add_argument("--eval-model", default=DEFAULT_MODEL)
    parser.add_argument("--window", default=DEFAULT_COMPLETION_WINDOW, choices=["24h", "1h"])
    parser.add_argument("--n", type=int, default=100, help="Number of unique questions")
    parser.add_argument("--batch-id", default=None, help="Resume from a completed eval batch ID")
    parser.add_argument("--input-csv", default=None, help="Override perturbed CSV path")
    parser.add_argument("--prompt-column", default="prompt_sent", help="Column for user message")
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
    parser.add_argument("--label", default="perturbed_eval", help="Batch folder label suffix")
    args = parser.parse_args()

    run_perturbed_smoke_test(
        eval_model=args.eval_model,
        n=args.n,
        input_path=args.input_csv or "data/perturbations/truthfulqa_perturbed.csv",
        prompt_column=args.prompt_column,
        completion_window=args.window,
        eval_batch_id=args.batch_id,
        max_tokens=args.max_tokens,
        batch_root=args.batch_root,
        no_think_prefix=args.no_think,
        label=args.label,
    )
