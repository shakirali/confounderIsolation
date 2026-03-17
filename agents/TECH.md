# Tech Stack

## Environment

| Tool | Purpose |
|---|---|
| Python 3.11+ | Primary language |
| uv | Virtual environment and package management |

---

## Dependencies

| Package | Purpose |
|---|---|
| `datasets` | Load TruthfulQA from HuggingFace |
| `openai` | Doubleword client (OpenAI-compatible API) |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scipy` | Statistical tests (ANOVA, Kendall's tau) |
| `scikit-learn` | Supporting ML utilities |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualisation |
| `jupyter` | Exploratory notebooks |
| `python-dotenv` | Load credentials from `.env` |
| `tqdm` | Progress bars for long-running loops |

---

## Model Inference

### Cloud (full runs via Doubleword Batch API)

| Tool | Details |
|---|---|
| Doubleword | Hosted batch inference API — OpenAI-compatible at `https://api.doubleword.ai/v1` |
| Eval model | `Qwen/Qwen3.5-35B-A3B-FP8` |
| Judge model | `Qwen/Qwen3.5-397B-A17B-FP8` |
| Client | `openai` Python library with `base_url=DOUBLEWORD_BASE_URL` |

---

## Configuration

All runtime configuration is via environment variables loaded from `.env`:

| Variable | Default | Description |
|---|---|---|
| `DOUBLEWORD_API_KEY` | — | Doubleword Batch API key |

---

## Data

| Format | Usage |
|---|---|
| JSONL | Batch input/output stored per batch in `experiments/doubleword_batches/` |
| CSV | Source datasets (`data/`) |
| HuggingFace `datasets` | Source for TruthfulQA |

---

## Pre-requisites

- Doubleword account and API key
