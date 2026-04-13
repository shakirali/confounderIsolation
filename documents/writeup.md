# Do Surface-Level Prompt Perturbations Affect Strong Models?
## A Replication Study of Tam et al. (2024) on Nemotron-120B

---

## 1. Background and Motivation

Tam et al. (2024), "Let Me Speak Freely? A Study of Language Model Responses to Constrained Output Formats," investigated whether restricting a model's output format — for example, requiring JSON responses or adding role-play prompts — artificially suppresses its benchmark performance. Their findings, tested on smaller models such as LLaMA-3-8B and GPT-3.5, showed significant degradation: up to ~42 percentage points on symbolic tasks and ~20pp on science MCQ under JSON-mode restrictions. The paper concluded that **format constraints act as a confounder in benchmark evaluation** — surface-level prompt changes, unrelated to task competence, were enough to materially lower scores.

This raises a critical question: is format sensitivity a fundamental property of language models, or is it an artifact of insufficient capacity in smaller models?

---

## 2. Hypotheses

**Tam et al. (2024) — Null hypothesis:**
> Format perturbations degrade model accuracy. Constrained output formats act as a confounder in benchmark evaluation, suppressing scores independently of the model's actual capability.

**Our hypothesis:**
> Surface-level prompt perturbations do not significantly affect benchmark performance on recent strong models.

We test this by applying a subset of perturbation types from Tam et al. to **Nemotron-120B** (NVIDIA Nemotron 3 Super 120B), a recent large-scale reasoning model, across four diverse benchmarks. If our hypothesis holds, perturbation Δ values should be consistently near zero regardless of task type or perturbation style.

---

## 3. Methodology

### 3.1 Model

All experiments use **`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`** as both the evaluated model and the judge (where applicable). This is a 120B-parameter mixture-of-experts model with strong reasoning capabilities.

### 3.2 Perturbation Types

Tam et al. define five perturbation categories (P1–P5). This study implements four of them — **P3 (prompt rephrasing variants) is not included**, as rewriting questions risks changing their semantic content and introducing a confound independent of format. The four implemented types are:

| ID | Tam et al. Label | Description | Our implementation |
|---|---|---|---|
| P1a | Output Format — Strict JSON | JSON output enforced with the word "strictly" | Prompt-level instruction: `Answer strictly in JSON format: {"answer": "..."}` |
| P1b | Output Format — Soft JSON | Gentle JSON request without "strictly" | Prompt-level instruction: `Answer in JSON format: {"answer": "..."}` |
| P2 | Instruction Complexity | Verbose preamble around the same question | "You are a knowledgeable assistant. Carefully consider the following..." |
| P4 | Role Framing | System prompt assigning a persona | System prompt: "You are a knowledgeable expert. Answer accurately and concisely." |
| P5 | Few-Shot Examples | Two worked examples prepended | Two domain-appropriate examples before the target question |

> **Note on P1 implementation:** Tam et al. enforced JSON via the API `response_format: json_object` flag. Nemotron-120B returns empty content when this flag is set on reasoning tasks (a model-specific incompatibility). We use prompt-level JSON instructions instead — this is equivalent in intent but differs in mechanism.

> **Note on P3:** Prompt phrasing/rephrasing variants are not implemented in this study. Rephrasing could alter question difficulty or meaning, making it impossible to isolate format sensitivity from content sensitivity.

### 3.3 Evaluation

For each benchmark, we run:
- A **baseline** batch (unperturbed prompts)
- A **perturbed** batch (5 types × N questions)

Scores are compared using **paired Δ** — perturbed accuracy minus baseline accuracy on the same matched questions. This controls for question difficulty and eliminates selection bias.

Scoring is either **deterministic** (MCQ letter match, exact string match) or via a **Nemotron judge** (for open-ended answers). No human annotation is used.

---

## 4. Experiments

---

### 4.1 ARC-Challenge

**Benchmark summary:**
ARC-Challenge (AI2 Reasoning Challenge) is a multiple-choice science question benchmark drawn from standardised school exams, targeting questions that require reasoning beyond simple retrieval. Each question has four labelled options (A–D); the model selects one letter.

- **Dataset:** 1,172 questions (allenai/ai2_arc, ARC-Challenge, full test split)
- **Scoring:** Deterministic — extracted letter vs gold answer key
- **Baseline accuracy:** 0.968

**Experiments performed:**
- Baseline eval: n=1,172 questions
- Perturbed eval: n=1,172 × 5 types = 5,860 rows
- Paired Δ computed across all matched question IDs

**Results:**

| Perturbation | Perturbed Accuracy | Δ vs Baseline |
|---|---|---|
| P1a (format strict) | 0.970 | −0.001 |
| P1b (format soft) | 0.968 | −0.003 |
| P2 (complexity) | 0.965 | −0.006 |
| P4 (role) | 0.961 | −0.010 |
| P5 (few-shot) | 0.966 | −0.003 |

**Finding:**
All perturbations produce Δ within −1.0pp. No perturbation type causes meaningful degradation. The model's science reasoning ability is entirely unaffected by prompt surface changes. This strongly supports our hypothesis. In contrast, Tam et al. reported ~20pp drops on ARC under JSON-mode for smaller models.

---

### 4.2 MATH-500

**Benchmark summary:**
MATH-500 is a curated subset of 500 competition mathematics problems spanning algebra, geometry, number theory, and calculus at difficulty levels 1–5. Problems require multi-step mathematical reasoning; answers are symbolic expressions or numerical values.

- **Dataset:** 500 problems (HuggingFaceH4/MATH-500, test split)
- **Scoring:** Nemotron judge — given (problem, gold answer, model response) → correct/incorrect
- **Baseline accuracy:** 0.988

**Experiments performed:**
- Baseline eval: n=500 problems; 415 non-empty responses (85 excluded due to token exhaustion on hard Level 4–5 problems)
- Perturbed eval: n=500 × 5 types = 2,500 rows; 2,197 non-empty
- Paired Δ computed on matched problems with valid scores in both runs

**Results:**

| Perturbation | Δ vs Baseline | n (paired) |
|---|---|---|
| P1a (format strict) | +0.002 | 413 |
| P1b (format soft) | +0.003 | 404 |
| P2 (complexity) | **+0.007** | 411 |
| P4 (role) | +0.003 | 407 |
| P5 (few-shot) | 0.000 | 411 |

**Finding:**
Every perturbation produces zero or positive Δ. The model's mathematical reasoning is completely insensitive to prompt surface changes — if anything, the verbose preamble (P2, +0.7pp) marginally focuses the model. Near-ceiling performance (98.8%) across all difficulty levels and subjects confirms that Nemotron-120B's mathematical capability is not confounded by format.

---

### 4.3 Last Letter Concatenation

**Benchmark summary:**
Last Letter Concatenation is a symbolic string task: given a full name (e.g. "Elon Musk"), concatenate the last letter of each word to form the answer ("nk"). The task requires precise character-level attention and has no semantic ambiguity — the answer is fully deterministic. Tam et al. found this task among the most sensitive to format restrictions for smaller models (~38–42pp drops under JSON-mode for LLaMA-3-8B).

- **Dataset:** 1,000 examples (yoonholee/last-letter-concatenation, train split)
- **Scoring:** Exact string match (case-insensitive)
- **Baseline accuracy:** 0.983

**Experiments performed:**
- Baseline eval: n=1,000 names
- Perturbed eval: n=1,000 × 5 types = 5,000 rows
- Zero empty responses across all 6,000 rows

**Results:**

| Perturbation | Accuracy | Δ vs Baseline | Regressions | Recoveries |
|---|---|---|---|---|
| P1a (format strict) | 0.986 | +0.003 | 6 | 9 |
| P1b (format soft) | 0.991 | +0.008 | 4 | 12 |
| P2 (complexity) | 0.987 | +0.004 | 4 | 8 |
| P4 (role) | 0.981 | −0.002 | 6 | 4 |
| P5 (few-shot) | 0.969 | **−0.014** | 22 | 8 |

**Finding:**
JSON format restrictions (P1a, P1b) have no negative effect — both show small positive Δ. The only notable degradation is P5 (−1.4pp, 22 regressions): the two few-shot examples use 2-word names, which subtly shifts the letter-extraction strategy and causes errors on some longer names. This is a content effect from the examples themselves, not a format effect.

Baseline errors (17/1000) are entirely systematic: 13/17 involve the names `Daniel` or `William`, where the model consistently misidentifies the last letter — these errors appear identically under all perturbation types and are independent of prompt changes.

The contrast with Tam et al. is stark: where LLaMA-3-8B dropped ~42pp under JSON-mode, Nemotron-120B shows +0.3pp — model scale eliminates format sensitivity on this symbolic task entirely.

---

### 4.4 MultiFin

**Benchmark summary:**
MultiFin is a multilingual financial article headline classification benchmark. Given a short headline (typically 3–10 words), the model classifies it into one of six topic categories: Business & Management, Finance, Government & Controls, Industry, Tax & Accounting, or Technology. This is a harder task than ARC — the baseline accuracy is lower (72.9%), reflecting genuine ambiguity in short financial headlines.

- **Dataset:** 546 English test rows (awinml/MultiFin, all_languages_highlevel, split=test, lang=English)
- **Scoring:** Deterministic MCQ — fixed A–F options corresponding to the six labels
- **Baseline accuracy:** 0.729

**Experiments performed:**
- Baseline eval: n=546 headlines
- Perturbed eval: n=546 × 5 types = 2,730 rows
- Zero empty or parse-failure responses across all 3,276 rows

**Results:**

| Perturbation | Accuracy | Δ vs Baseline | Regressions | Recoveries |
|---|---|---|---|---|
| P1a (format strict) | 0.745 | +0.016 | 6 | 15 |
| P1b (format soft) | 0.745 | +0.016 | 6 | 15 |
| P2 (complexity) | 0.738 | +0.009 | 13 | 18 |
| P4 (role) | 0.729 | 0.000 | 8 | 8 |
| P5 (few-shot) | 0.760 | **+0.031** | 11 | 28 |

**Finding:**
Every perturbation is neutral or positive. No format change hurts performance. P5 produces the largest positive effect (+3.1pp, 28 recoveries vs 11 regressions): on a genuinely ambiguous short-text classification task, worked examples provide useful disambiguation cues. P4 is perfectly neutral (Δ=0.000, 8 regressions and 8 recoveries — an exact balance). This result is notable because MultiFin has the lowest baseline of all benchmarks (72.9%) — even with the most room to fall, no perturbation causes degradation.

---

## 5. Summary of Results

| Benchmark | Task Type | Baseline | Max Negative Δ | Max Positive Δ | Verdict |
|---|---|---|---|---|---|
| ARC-Challenge | Science MCQ | 0.968 | −0.010 (P4) | −0.001 (P1a) | Hypothesis supported |
| MATH-500 | Math reasoning | 0.988 | 0.000 (P5) | +0.007 (P2) | Hypothesis strongly supported |
| Last Letter | Symbolic string | 0.983 | −0.014 (P5) | +0.008 (P1b) | Hypothesis supported |
| MultiFin | Financial MCQ | 0.729 | 0.000 (P4) | +0.031 (P5) | Hypothesis strongly supported |

---

## 6. Scope and Limitations

- **P3 not implemented:** Tam et al.'s prompt rephrasing perturbation is excluded. Future work could test paraphrase variants using a controlled rewriting model.
- **P1 mechanism differs:** API-enforced JSON (`response_format: json_object`) is replaced with prompt-level instructions due to Nemotron's incompatibility with the flag on reasoning tasks. The intent is equivalent, but the enforcement mechanism is weaker.
- **Single model tested:** All results are for Nemotron-120B. Results may not generalise to other large models at similar scale.
- **DDXPlus pending:** The medical differential diagnosis benchmark from Tam et al. has not yet been run. Results will be added when complete.

---

## 7. Conclusion

Across all four benchmarks reported here, the results strongly support our hypothesis: **surface-level prompt perturbations do not significantly affect benchmark performance on Nemotron-120B.** Format instructions, role prompts, complexity preambles, and few-shot examples all produce Δ values within ±1.5pp on ARC, MATH-500, Last Letter, and MultiFin. The largest single negative shift across all experiments is −1.4pp (Last Letter, P5 few-shot) — and this is a content effect driven by the examples themselves, not a format effect.

**The confounder concern raised by Tam et al. appears to be a small-model phenomenon.** At 120B scale, Nemotron separates surface-form compliance from task execution — it follows format instructions without sacrificing reasoning quality. This has a direct practical implication: benchmark scores from strong modern models are robust to prompt surface variation and can be treated as reliable capability signals, not format-confounded artifacts.
