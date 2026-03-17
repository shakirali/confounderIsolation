# Confounder Isolation in LLM Safety Benchmarks

An empirical study investigating whether LLM safety benchmark scores reflect genuine model safety properties, or are significantly influenced by surface-level confounders such as output formatting, instruction complexity, and prompt phrasing.

## Research Question

> Do safety benchmark scores reflect genuine model safety properties, or are they significantly influenced by surface-level factors like output formatting, instruction complexity, and prompt phrasing?

**Motivation:** Bean et al. (2025) identified that most LLM benchmarks suffer from "confounding subtasks" — but no empirical evidence existed quantifying how much this matters for safety evaluations. This project fills that gap by systematically testing [TruthfulQA](https://huggingface.co/datasets/truthful_qa) (817 questions) across multiple surface-level perturbations and model scales.

## Methodology

We apply 5 perturbation types to TruthfulQA prompts — changes that are logically irrelevant to truthfulness — and measure how much model scores shift:

| ID | Type | Description |
|----|------|-------------|
| P1 | Output Format Constraint | Requires JSON response format |
| P2 | Instruction Complexity | Adds verbose preamble around the question |
| P3 | Prompt Phrasing Variants | Rephrases questions (2 variants) |
| P4 | Role Framing | Adds system prompt assigning a persona (medical expert) |
| P5 | Few-Shot Examples | Prepends example Q&A pairs |

**Models evaluated:**
- `meta-llama/Llama-3.1-8B-Instruct-Turbo` — full evaluation (all 4,902 variants via Doubleword batch)
- `meta-llama/Llama-3.1-70B-Instruct-Turbo` — stratified evaluation (1,000 variants via Doubleword batch)

**Statistical analysis:** score shift per perturbation type, two-way ANOVA, Kendall's tau ranking stability.

## Pipeline

```
TruthfulQA (HuggingFace)
    ↓ load_dataset.py
data/baseline/truthfulqa_raw.csv  (817 questions)
    ↓ generate_perturbations.py
data/perturbations/truthfulqa_perturbed.csv  (817 × 4 perturbed prompts)
    ↓ src/doubledword/baseline_eval_smoke_test_doubleword.py
    ↓ src/doubledword/perturbed_eval_smoke_test.py
experiments/doubleword_batches/<eval_batch_id>_{baseline,perturbed}_eval/  (input.jsonl + output.jsonl)
    ↓ src/doubledword/baseline_judge_doubleword.py
    ↓ src/doubledword/perturbed_judge_doubleword.py
experiments/doubleword_batches/<judge_batch_id>_{baseline,perturbed}_judge/  (input.jsonl + output.jsonl with scores)
```

## Setup

**Requirements:** Python 3.10+, [uv](https://github.com/astral-sh/uv)

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

```
DOUBLEWORD_API_KEY=<your_key>   # Doubleword Batch API key
```

## Usage

**Step 1 — Load dataset:**

```bash
python src/load_dataset.py   # Downloads TruthfulQA → data/baseline/truthfulqa_raw.csv
```

**Step 2 — Generate perturbations:**

```bash
python src/generate_perturbations.py   # Produces data/perturbations/truthfulqa_perturbed.csv
```

**Step 3 — Run evaluations (Doubleword batch):**

```bash
# Baseline (unperturbed)
python src/doubledword/baseline_eval_smoke_test_doubleword.py

# Perturbed
python src/doubledword/perturbed_eval_smoke_test.py
```

**Step 4 — Score responses:**

```bash
# Baseline
PYTHONPATH=src/doubledword python src/doubledword/baseline_judge_doubleword.py --eval-batch-id <baseline_eval_batch_id>

# Perturbed
PYTHONPATH=src/doubledword python src/doubledword/perturbed_judge_doubleword.py --eval-batch-id <perturbed_eval_batch_id>
```

Scores are saved in `output.jsonl` inside the judge batch folder under `experiments/doubleword_batches/`.

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Dataset loading + Doubleword batch pipeline | Done |
| 2 | Perturbation generation (P1, P2, P4, P5) | Done |
| 2 | P3 prompt rephrasing | Not implemented |
| 3 | Full evaluation across all models | Pending |
| 4 | Statistical analysis | Pending |
| 5 | Technical report | Pending |

## Repository Structure

```
src/
  load_dataset.py              # Fetch TruthfulQA from HuggingFace
  generate_perturbations.py    # Apply perturbations (P1, P2, P4, P5)
  doubledword/
    doubleword_client.py       # Doubleword Batch API client
    baseline_eval_smoke_test_doubleword.py  # Baseline evaluation via batch
    perturbed_eval_smoke_test.py            # Perturbed evaluation via batch
    judge_doubleword.py        # Judge scoring via batch
data/
  baseline/                    # Raw TruthfulQA CSV
  perturbations/               # Perturbed prompt CSV
results/
  raw/                         # Model responses (+ scored variants)
  figures/                     # Charts and tables (Phase 4)
agents/
  SPEC.md                      # Research specification
  PLAN.md                      # Implementation plan
  TECH.md                      # Tech stack and environment details
documents/                     # Reference papers (Bean et al. 2025)
```

## Reference

Bean, A. et al. (2025). *Benchmarks as Confounders: Disentangling What Language Models Actually Learn.* [Link to paper in `documents/`]
