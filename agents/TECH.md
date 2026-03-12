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
| `openai` | Ollama client (OpenAI-compatible local API) |
| `together` | Together.ai cloud inference client |
| `pandas` | Data manipulation and CSV I/O |
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

### Local (smoke tests)

| Tool | Details |
|---|---|
| Ollama | Local inference server — exposes an OpenAI-compatible REST API at `http://localhost:11434/v1` |
| Model | `llama3.1:8b` |
| Client | `openai` Python library with `base_url=OLLAMA_BASE_URL` |

### Cloud (full runs)

| Tool | Details |
|---|---|
| Together.ai | Hosted inference API |
| Models | `meta-llama/Llama-3.1-8B-Instruct-Turbo`, `meta-llama/Llama-3.1-70B-Instruct-Turbo` |
| Client | `together` Python library |

---

## Configuration

All runtime configuration is via environment variables loaded from `.env`:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama local server address |
| `TOGETHER_API_KEY` | — | Together.ai API key |
| `JUDGE_MODEL` | `llama3.1:8b` | Model used for response scoring |
| `REPHRASE_MODEL` | `llama3.1:8b` | Model used for P3 prompt rephrasing |

---

## Data

| Format | Usage |
|---|---|
| CSV | All intermediate and final datasets |
| HuggingFace `datasets` | Source for TruthfulQA |

---

## Pre-requisites

- Ollama installed and running (`ollama serve`)
- `llama3.1:8b` pulled locally (`ollama pull llama3.1:8b`)
- Together.ai account and API key for full runs
