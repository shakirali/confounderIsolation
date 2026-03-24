# ARC-Challenge — Doubleword batches

ARC-Challenge eval batch folders (`<batch_id>_<label>/` with `input.jsonl` and `output.jsonl`) are stored here.

Code constant: `ARC_BATCH_ROOT` in `src/doubledword/doubleword_client.py` (path `experiments/doubleword_batches/arc` relative to the repo root).

## Run eval (from repo root)

Requires `DOUBLEWORD_API_KEY` in `.env`.

**Protocol:** Default user text is the **raw prompt** from the CSV. Pass **`--no-think`** on `arc_*_eval.py` or the generic smoke scripts to prepend `/no_think` for **Qwen-style** models only (Nemotron never gets that line, even with `--no_think`).

**Baseline** (first `n` rows of `data/baseline/arc_challenge_test_raw.csv`, column `prompt`):

```bash
PYTHONPATH=src/doubledword uv run python src/doubledword/arc_baseline_eval.py --n 10
```

**Perturbed** (first `n` question_ids × 5 types from `data/perturbations/arc_challenge_test_perturbed.csv`):

```bash
PYTHONPATH=src/doubledword uv run python src/doubledword/arc_perturbed_eval.py --n 10
```

Resume after completion: `--batch-id <uuid>` (same flags as the run you are downloading).

Default **`max_tokens` is 4096** on `arc_baseline_eval.py` / `arc_perturbed_eval.py` (override with `--max-tokens`).

Default **eval model** is **Nemotron 120B** (`ARC_EVAL_MODEL` in `doubleword_client.py`). Override with `--eval-model`.

**TruthfulQA-style flags** (any CSV / batch root): use `baseline_eval_smoke_test_doubleword.py` and `perturbed_eval_smoke_test.py` with `--input-csv`, `--prompt-column`, `--batch-root`, `--max-tokens`, `--no-think`, `--label`.
