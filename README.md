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
- `meta-llama/Llama-3.1-8B-Instruct-Turbo` — full evaluation (all 4,902 variants via Together.ai)
- `meta-llama/Llama-3.1-70B-Instruct-Turbo` — stratified evaluation (1,000 variants via Together.ai)
- `llama3.1:8b` — local smoke tests and judge scoring (via Ollama)

**Statistical analysis:** score shift per perturbation type, two-way ANOVA, Kendall's tau ranking stability.

## Pipeline

```
TruthfulQA (HuggingFace)
    ↓ load_dataset.py
data/baseline/truthfulqa_raw.csv  (817 questions)
    ↓ generate_perturbations.py
data/perturbations/truthfulqa_perturbed.csv  (817 × 5 perturbed prompts)
    ↓ evaluate_models.py
Model responses per perturbation
    ↓ scoring.py
results/raw/responses_*.csv  (responses + binary truthfulness scores)
    ↓ analysis.py
results/figures/  (score shift charts, statistical tables)
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
OLLAMA_BASE_URL=http://localhost:11434/v1   # Local Ollama server
TOGETHER_API_KEY=<your_key>                 # Together.ai API key
JUDGE_MODEL=llama3.1:8b                     # Scoring judge model
REPHRASE_MODEL=llama3.1:8b                  # P3 rephrasing model
```

For local inference, install [Ollama](https://ollama.com) and pull the model:

```bash
ollama pull llama3.1:8b
```

## Usage

**Phase 1 — Load dataset and run smoke test (local):**

```bash
python src/load_dataset.py       # Downloads TruthfulQA → data/baseline/
python src/smoke_test.py         # Queries first 50 questions via Ollama
python src/scoring.py            # Scores smoke test responses
```

**Phase 2 — Generate perturbations:**

```bash
python src/generate_perturbations.py   # Produces data/perturbations/truthfulqa_perturbed.csv
python src/perturbed_eval.py           # Tests perturbations on 50 questions locally
```

**Phase 3 — Full evaluation (cloud):**

Requires `TOGETHER_API_KEY`. Use `evaluate_models.query_model()` and `scoring.score_response()` in your evaluation scripts.

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1a | Smoke test on local Ollama | Done |
| 1b | Inference + scoring framework | Done |
| 2 | Perturbation generation (P1, P2, P4, P5) | Done |
| 2 | P3 prompt rephrasing | Requires Together.ai |
| 3 | Full evaluation across all models | Pending |
| 4 | Statistical analysis | Pending |
| 5 | Technical report | Pending |

## Repository Structure

```
src/
  load_dataset.py         # Phase 1: fetch TruthfulQA
  generate_perturbations.py  # Phase 2: apply perturbations
  evaluate_models.py      # Phase 3: unified inference (Ollama + Together.ai)
  scoring.py              # Phase 3: judge-based truthfulness scoring
  smoke_test.py           # Phase 1a: local smoke test
  perturbed_eval.py       # Phase 2 testing
  analysis.py             # Phase 4: statistical analysis (placeholder)
data/
  baseline/               # Raw TruthfulQA CSV + smoke test responses
  perturbations/          # Perturbed prompt CSV
results/
  raw/                    # Scored model responses
  figures/                # Charts and tables (Phase 4)
agents/
  SPEC.md                 # Research specification
  PLAN.md                 # Implementation plan
  TECH.md                 # Tech stack and environment details
documents/                # Reference papers (Bean et al. 2025)
```

## Reference

Bean, A. et al. (2025). *Benchmarks as Confounders: Disentangling What Language Models Actually Learn.* [Link to paper in `documents/`]
