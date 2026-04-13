"""
MATH-500 baseline eval via Doubleword batch — Qwen3.5-397B.

Writes under experiments/doubleword_batches/math500_qwen/.
No /no_think prefix (thinking on by default).
Score responses with math500_qwen_baseline_judge.py (Nemotron judge).

Usage (repo root):
    .venv/bin/python src/doubledword/math500_qwen_baseline_eval.py --n 10 --window 1h
    .venv/bin/python src/doubledword/math500_qwen_baseline_eval.py --window 24h
    .venv/bin/python src/doubledword/math500_qwen_baseline_eval.py --batch-id <uuid>
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_eval_smoke_test_doubleword import run_smoke_test
from doubleword_client import QWEN_EVAL_MODEL, MATH500_QWEN_BATCH_ROOT, DEFAULT_COMPLETION_WINDOW

load_dotenv()


def main():
    p = argparse.ArgumentParser(description="MATH-500 baseline eval — Qwen3.5-397B")
    p.add_argument("--n", type=int, default=10, help="Number of problems (default 10 smoke)")
    p.add_argument("--window", default=DEFAULT_COMPLETION_WINDOW, choices=["24h", "1h"])
    p.add_argument("--batch-id", default=None, help="Download existing batch instead of submitting")
    p.add_argument("--eval-model", default=QWEN_EVAL_MODEL)
    p.add_argument("--max-tokens", type=int, default=8192)
    args = p.parse_args()

    run_smoke_test(
        eval_model=args.eval_model,
        n=args.n,
        input_path="data/baseline/math500_baseline.csv",
        prompt_column="prompt",
        completion_window=args.window,
        eval_batch_id=args.batch_id,
        max_tokens=args.max_tokens,
        batch_root=MATH500_QWEN_BATCH_ROOT,
        no_think_prefix=False,
        label="math500_baseline_eval",
    )


if __name__ == "__main__":
    main()
