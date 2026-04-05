# Do AI Safety Benchmarks Measure What They Claim To? An Empirical Investigation

AI safety research relies on benchmarks to determine whether models are safe to deploy. If benchmark scores are driven by surface-level factors — output formatting, instruction complexity, prompt phrasing — rather than the underlying safety property being assessed, then safety evaluations may be systematically misleading.

Bean et al. (2025) identified this problem structurally, reviewing 445 LLM benchmarks and finding widespread **confounding subtasks**: auxiliary skills such as instruction following and output formatting silently inflate or deflate scores, independent of the target property. Their analysis was theoretical. No empirical evidence existed quantifying how much this matters in practice.

This post reports on an empirical study we ran to fill that gap, using TruthfulQA and ARC-Challenge as test cases across two models.

---

## Research Question

Do benchmark scores reflect a model's underlying capability, or are they significantly influenced by surface-level factors that should be orthogonal to the target property?

---

## Method

We applied five perturbation types to every question in each benchmark. Each perturbation modifies only surface-level properties of the prompt:

| ID | Perturbation | What changes |
|---|---|---|
| p1_format | Strict JSON output | Response format enforced via API `response_format` parameter |
| p1_format_soft | Soft JSON output | Same constraint expressed as a prompt instruction only |
| p2_complexity | Instruction complexity | Verbose "knowledgeable assistant" preamble prepended to the question |
| p4_role | Role framing | System prompt assigns a "knowledgeable expert" persona |
| p5_fewshot | Few-shot examples | Two Q&A examples prepended before the question |

None of these perturbations should logically affect whether a model knows the correct answer to a factual question. They constitute plausible confounders — variables that might influence measured scores without reflecting the target property.

### Experiment 1: TruthfulQA × Qwen3.5-35B

The full TruthfulQA validation set (817 questions) was evaluated using `Qwen3.5-35B-A3B-FP8`, producing 4,085 (question, perturbation) pairs. Responses were scored by `Qwen3.5-397B-A17B-FP8` acting as an LLM judge. All inference was run via batch API.

### Experiment 2: ARC-Challenge × Nemotron-120B

The full ARC-Challenge test split (1,172 questions) was evaluated using `NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`, producing 5,860 (question, perturbation) pairs. ARC-Challenge is a multiple-choice science benchmark, so scoring is deterministic — no LLM judge is required. Correct answer extraction from model output was fully automated.

---

## Experiment 1: TruthfulQA

### A Methodological Challenge: Reasoning Loop Failures

The Qwen eval model uses internal chain-of-thought reasoning. On a subset of questions, it enters a reasoning loop — exhausting its full token budget on thinking without producing a final answer. At `max_tokens=4096`, empty response rates ranged from 17% (p1_format) to 60% (p5_fewshot) across perturbation types.

Increasing `max_tokens` reduced but did not eliminate failures. More critically, simply discarding empty responses introduces selection bias: surviving questions over-represent easy, high-confidence cases. Any paired comparison between baseline and perturbed conditions would then be comparing different question populations.

We addressed this with a **triple-run methodology**: each (question, perturbation) pair was submitted three times independently at `max_tokens=4096`. Only pairs where all three runs returned non-empty content were retained as the *stable dataset*. This yields a conservative, unbiased sample for scoring, while the instability rate itself is treated as a separate measurement.

**Stable dataset:** 2,806 of 4,085 pairs (68.7%) were stable across all three runs. 259 of 817 questions were fully stable across all five perturbation types.

### Results: Truthfulness scores

For all matched (baseline, perturbation) pairs sharing the same question ID, we computed mean truthfulness scores:

| Perturbation | Pairs | Baseline mean | Perturbed mean | Δ |
|---|---|---|---|---|
| p1_format | 652 | 0.982 | 0.983 | +0.002 |
| p1_format_soft | 546 | 0.982 | 0.976 | −0.005 |
| p2_complexity | 619 | 0.985 | 0.994 | +0.008 |
| p4_role | 568 | 0.991 | 0.996 | +0.005 |
| p5_fewshot | 315 | 0.990 | 0.994 | +0.003 |
| **Overall** | **2,700** | **0.986** | **0.988** | **+0.003** |

All deltas fall within ±0.01. For questions where the model responds consistently, the perturbations tested here have essentially no effect on truthfulness scores.

### Results: Response stability

The instability rate — the fraction of (question, perturbation) pairs where the model fails to produce a stable response across three runs — varies substantially by perturbation type:

| Perturbation | Unstable pairs | Rate |
|---|---|---|
| p1_format | 141 / 817 | 17.3% |
| p2_complexity | 176 / 817 | 21.5% |
| p4_role | 227 / 817 | 27.8% |
| p1_format_soft | 243 / 817 | 29.7% |
| p5_fewshot | 492 / 817 | 60.2% |

Few-shot prompting caused the model to produce no stable answer on 60% of questions — more than triple the rate of the API-enforced JSON condition. This is an operationally significant finding: an evaluation pipeline that does not account for empty outputs would count all non-responses as incorrect, producing a large spurious score drop under few-shot conditions that has nothing to do with truthfulness.

### Selection bias and the limits of the stable dataset

The stable filter creates a non-representative sample. The 255 questions with valid scores across all six conditions (baseline + five perturbations) are the easiest, most consistently-answered questions in TruthfulQA. On this fully-paired subset, scores approach ceiling across all conditions (0.988–1.000), with p1_format_soft showing the only notable delta (−0.008 vs baseline).

The hard, ambiguous questions — where confounding effects are most plausible — are precisely the ones the model fails to answer consistently, and are therefore excluded from the paired comparison. The current analysis can establish that perturbations do not affect truthfulness scores on questions the model is confident about. Whether they affect scores on harder questions remains an open question.

---

## Experiment 2: ARC-Challenge

### No Stability Issues with Nemotron

Unlike Qwen on TruthfulQA, Nemotron-120B produced stable outputs across the full ARC test split with no triple-run methodology required. Parse failure rate was approximately 0.3% (4 out of 1,168 baseline responses, all due to `finish_reason: length` on a small number of long-reasoning questions). This allowed a clean single-run evaluation over all 1,172 questions.

### Results: Accuracy scores

**Full test split (1,172 questions), Nemotron-120B:**

| | Baseline | Perturbed (aggregate) |
|---|---|---|
| Correct | 1,131 | 5,665 |
| Wrong | 37 | 185 |
| Parse fail | 4 | 10 |
| **Accuracy (parsed)** | **96.83%** | **96.84%** |

By perturbation type:

| Perturbation | Correct | Total (parsed) | Accuracy | Δ vs Baseline |
|---|---|---|---|---|
| p1_format | 1,137 | 1,168 | 97.0% | +0.2pp |
| p1_format_soft | 1,135 | 1,170 | 97.0% | +0.2pp |
| p2_complexity | 1,132 | 1,170 | 96.8% | 0.0pp |
| p4_role | 1,128 | 1,162 | 97.1% | +0.3pp |
| p5_fewshot | 1,133 | 1,170 | 96.8% | 0.0pp |

All deltas are within ±0.3 percentage points of baseline. The perturbations have no meaningful effect on ARC-Challenge accuracy.

### A sharper null result

The ARC result is cleaner than the TruthfulQA result for two reasons. First, scoring is deterministic — there is no LLM judge introducing its own variability. Second, Nemotron does not suffer from the reasoning-loop instability that plagued Qwen on TruthfulQA, so the result covers the full question set without any stability-based filtering. The null result here is unambiguous: prompt surface variations do not affect multiple-choice science accuracy.

---

## Discussion

### The main finding holds across benchmarks and models

Both experiments converge on the same conclusion: **surface-level prompt perturbations have negligible effects on scores for questions the model answers at all**. On TruthfulQA, all truthfulness score deltas fall within ±0.01. On ARC-Challenge, all accuracy deltas fall within ±0.3pp. This holds across two different models, two different benchmarks, and two different scoring mechanisms (LLM judge vs. deterministic MCQ matching).

This partially addresses the concern raised by Bean et al. (2025): at least for the confounders tested here — output formatting constraints, preamble complexity, role prompts, and few-shot examples — measured scores appear robust on questions the model is confident about.

### The overlooked confounder: response generation stability

The more striking finding from TruthfulQA is that perturbations strongly affect whether a model responds at all. Few-shot prompting caused 60% of questions to yield no stable response with Qwen — a mechanism entirely distinct from score-shifting. Rather than degrading answer quality, certain perturbations dramatically increase the probability that no answer is produced.

This has direct implications for evaluation practice. If different models have different stability profiles under the same perturbation, raw accuracy scores will conflate "genuinely less capable" with "more likely to fail to respond." Benchmark comparisons that do not control for response completion rates may be unreliable.

Notably, this instability was model-specific: Nemotron-120B on ARC-Challenge showed a ~0.3% parse failure rate with no triple-run methodology required, compared to 17–60% instability rates for Qwen on TruthfulQA under the same perturbation conditions. The severity of this confounder depends heavily on whether the eval model uses extended chain-of-thought reasoning and how its token budget is configured.

### Hypothesis 1 (partially supported)

**H1** predicted that scores would vary significantly across surface-level perturbations. On the stable TruthfulQA dataset and the full ARC dataset, this was not observed. However, the TruthfulQA stable dataset is not representative of the full question set — the analysis does not rule out larger effects on harder questions.

### Practical recommendations for benchmark designers

1. **Report response completion rates** alongside accuracy scores. A 5-point accuracy drop and a 10-point completion rate drop tell very different causal stories.
2. **Validate that evaluation protocol choices** (few-shot prompting, output format constraints, system prompts) do not cause differential completion rate drops across models or question subsets.
3. **Use matched question IDs** when comparing conditions. Averaging over different question subsets introduces selection bias that can mask or manufacture apparent score differences.
4. **Test stability across models.** The severity of the reasoning-loop confounder depends on the model. Evaluations using extended chain-of-thought models should explicitly measure and report empty response rates.

---

## What's Next

A replication of the TruthfulQA experiment using Nemotron-120B (with Nemotron also serving as judge) is currently in progress. This will allow direct comparison between the Qwen and Nemotron results on the same benchmark and questions — in particular, testing whether the reasoning-loop instability finding is specific to Qwen or generalises to other reasoning models.

The statistical analysis phase is also in progress. Planned work includes two-way ANOVA decomposing variance attributable to perturbation type versus question difficulty, and Kendall's tau to test ranking stability across conditions. A full technical report, code repository, and perturbation dataset release will follow.

---

## References

- Bean et al. (2025). *Measuring what Matters: Construct Validity in LLM Benchmarks.*
- Lin et al. (2022). *TruthfulQA: Measuring How Models Mimic Human Falsehoods.*
- Clark et al. (2018). *Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge.*
- Sclar et al. (2023). *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design.*
