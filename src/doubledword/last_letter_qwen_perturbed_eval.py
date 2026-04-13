"""
Last Letter Concatenation perturbed eval via Doubleword batch — Qwen3.5-397B.

Writes under experiments/doubleword_batches/last_letter_qwen/.
No /no_think prefix (thinking on by default).

Usage (repo root):
    .venv/bin/python src/doubledword/last_letter_qwen_perturbed_eval.py --n 10 --window 1h
    .venv/bin/python src/doubledword/last_letter_qwen_perturbed_eval.py --window 24h
    .venv/bin/python src/doubledword/last_letter_qwen_perturbed_eval.py --batch-id <uuid>
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from doubleword_client import QWEN_EVAL_MODEL, LAST_LETTER_QWEN_BATCH_ROOT, DEFAULT_COMPLETION_WINDOW
from perturbed_eval_smoke_test import run_perturbed_smoke_test

load_dotenv()


def main():
    p = argparse.ArgumentParser(description="Last Letter perturbed eval — Qwen3.5-397B")
    p.add_argument("--n", type=int, default=10, help="Number of unique examples (default 10 smoke → 50 rows)")
    p.add_argument("--window", default=DEFAULT_COMPLETION_WINDOW, choices=["24h", "1h"])
    p.add_argument("--batch-id", default=None, help="Download existing batch instead of submitting")
    p.add_argument("--eval-model", default=QWEN_EVAL_MODEL)
    p.add_argument("--max-tokens", type=int, default=8192)
    args = p.parse_args()

    run_perturbed_smoke_test(
        eval_model=args.eval_model,
        n=args.n,
        input_path="data/perturbations/last_letter_perturbed.csv",
        prompt_column="prompt_sent",
        completion_window=args.window,
        eval_batch_id=args.batch_id,
        max_tokens=args.max_tokens,
        batch_root=LAST_LETTER_QWEN_BATCH_ROOT,
        no_think_prefix=False,
        label="last_letter_perturbed_eval",
        id_column="unique_id",
    )


if __name__ == "__main__":
    main()
