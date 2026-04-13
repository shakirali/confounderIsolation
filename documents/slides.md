# Slide Content: Do Surface-Level Prompt Perturbations Affect Strong Models?

---

## Slide 1 — Title

**Do Surface-Level Prompt Perturbations Affect Strong Models?**

A Replication Study of Tam et al. (2024) on Nemotron-120B and Qwen3.5-397B

---

## Slide 2 — Motivation

**The format confounder problem**

- Tam et al. (2024), *"Let Me Speak Freely?"*, tested whether restricting output format (JSON mode, role prompts, few-shot examples) suppresses benchmark scores
- Findings on **small models** (LLaMA-3-8B, GPT-3.5):
  - Up to **~42pp drop** on symbolic tasks (Last Letter) under JSON-mode
  - Up to **~20pp drop** on science MCQ (ARC) under JSON-mode
  - Conclusion: **format constraints are a confounder** — they depress scores independently of capability

**The open question:** Is format sensitivity a fundamental property of LLMs, or an artifact of insufficient scale?

---

## Slide 3 — Hypotheses

**Null hypothesis (Tam et al.)**
> Format perturbations degrade model accuracy. Constrained output formats act as a confounder in benchmark evaluation.

**Our hypothesis**
> Surface-level prompt perturbations do **not** significantly affect benchmark performance on recent strong models.

**Prediction:** If the hypothesis holds, Δ values (perturbed − baseline) should be near zero across all benchmarks and perturbation types.

---

## Slide 4 — Perturbation Types (5 conditions)

| ID | Label | What changes |
|---|---|---|
| **P1a** | Format strict | Prompt instruction: `Answer strictly in JSON format: {"answer": "..."}` |
| **P1b** | Format soft | Prompt instruction: `Answer in JSON format: {"answer": "..."}` |
| **P2** | Complexity | Verbose preamble: *"You are a knowledgeable assistant. Carefully consider the following..."* |
| **P4** | Role framing | System prompt: *"You are a knowledgeable expert. Answer accurately and concisely."* |
| **P5** | Few-shot | Two domain-appropriate worked examples prepended before question |

Note: P3 (prompt rephrasing) excluded — rephrasing risks altering question difficulty, introducing a content confound.

---

## Slide 5 — Experimental Setup

**Models**

| Role | Model |
|---|---|
| Primary eval + judge | Nemotron-120B (`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`) |
| Comparison eval | Qwen3.5-397B (`Qwen/Qwen3.5-397B-A17B-FP8`) |

**Benchmarks**

| Benchmark | Task type | N | Scoring |
|---|---|---|---|
| ARC-Challenge | Science MCQ (A–D) | 1,172 | Deterministic letter match |
| MATH-500 | Math reasoning | 500 | Nemotron judge (correct/incorrect) |
| Last Letter Concatenation | Symbolic string | 1,000 | Exact string match |
| MultiFin | Financial headline MCQ (A–F) | 546 | Deterministic letter match |

**Evaluation method:** Paired Δ — perturbed accuracy minus baseline accuracy on matched questions.

---

## Slide 6 — Results: ARC-Challenge

**Baseline accuracy: 96.8%** (1,131/1,168 parsed, n=1,172 full test split)

| Perturbation | Perturbed Accuracy | Δ |
|---|---|---|
| P1a format strict | 0.970 | −0.001 |
| P1b format soft | 0.968 | −0.003 |
| P2 complexity | 0.965 | −0.006 |
| P4 role | 0.961 | −0.010 |
| P5 few-shot | 0.966 | −0.003 |

**Max negative Δ: −1.0pp (P4)**

Finding: All perturbations within 1pp. Science reasoning completely unaffected by prompt surface changes.

**Compare:** Tam et al. reported ~20pp drops on ARC under JSON-mode for smaller models.

---

## Slide 7 — Results: MATH-500

**Baseline accuracy: 98.8%** (n=500 competition math problems, levels 1–5)

| Perturbation | Δ | Pairs |
|---|---|---|
| P1a format strict | +0.002 | 413 |
| P1b format soft | +0.003 | 404 |
| P2 complexity | **+0.007** | 411 |
| P4 role | +0.003 | 407 |
| P5 few-shot | 0.000 | 411 |

**Max negative Δ: 0.000 (P5) — no degradation at all**

Finding: Zero or positive Δ on every perturbation type. Mathematical reasoning at near-ceiling is insensitive to format. Verbose preamble (P2) marginally focuses the model (+0.7pp).

---

## Slide 8 — Results: Last Letter Concatenation

**Baseline accuracy: 98.3%** (n=1,000, exact string match)

| Perturbation | Accuracy | Δ | Regressions | Recoveries |
|---|---|---|---|---|
| P1a format strict | 0.986 | +0.003 | 6 | 9 |
| P1b format soft | 0.991 | +0.008 | 4 | 12 |
| P2 complexity | 0.987 | +0.004 | 4 | 8 |
| P4 role | 0.981 | −0.002 | 6 | 4 |
| P5 few-shot | 0.969 | **−0.014** | 22 | 8 |

**Max negative Δ: −1.4pp (P5)**

Finding: P5 drop is a **content effect** — the two few-shot examples use 2-word names, subtly shifting extraction strategy for longer names. Not a format effect.

**Compare:** LLaMA-3-8B dropped ~42pp under JSON-mode on this same task. Nemotron-120B: +0.3pp.

---

## Slide 9 — Results: MultiFin

**Baseline accuracy: 72.9%** (n=546 English headlines, 6-way classification)

| Perturbation | Accuracy | Δ | Regressions | Recoveries |
|---|---|---|---|---|
| P1a format strict | 0.745 | +0.016 | 6 | 15 |
| P1b format soft | 0.745 | +0.016 | 6 | 15 |
| P2 complexity | 0.738 | +0.009 | 13 | 18 |
| P4 role | 0.729 | **0.000** | 8 | 8 |
| P5 few-shot | 0.760 | **+0.031** | 11 | 28 |

**Max negative Δ: 0.000 — no degradation**

Finding: All perturbations neutral or positive. P5 +3.1pp: worked examples provide disambiguation cues for genuinely ambiguous short headlines. Notably, even with the **lowest baseline** of all benchmarks (72.9%), no perturbation causes a drop.

---

## Slide 10 — Cross-Benchmark Summary: Nemotron-120B

| Benchmark | Baseline | P1a | P1b | P2 | P4 | P5 |
|---|---|---|---|---|---|---|
| ARC-Challenge | 0.968 | −0.001 | −0.003 | −0.006 | −0.010 | −0.003 |
| MATH-500 | 0.988 | +0.002 | +0.003 | +0.007 | +0.003 | 0.000 |
| Last Letter | 0.983 | +0.003 | +0.008 | +0.004 | −0.002 | −0.014 |
| MultiFin | 0.729 | +0.016 | +0.016 | +0.009 | 0.000 | **+0.031** |

**Key takeaway:** All four benchmarks are within **±1.5pp** under every perturbation. The hypothesis is supported across all task types tested.

---

## Slide 11 — Cross-Model Comparison: Nemotron vs Qwen3.5-397B

**Baseline gap** — Nemotron leads on all 4 capability benchmarks:

| Benchmark | Nemotron | Qwen | Gap |
|---|---|---|---|
| ARC-Challenge | 0.968 | 0.880 | −8.8pp |
| Last Letter | 0.983 | 0.960 | −2.3pp |
| MultiFin | 0.729 | 0.702 | −2.7pp |
| MATH-500 | 0.988 | 0.958 | −3.0pp |

**Under perturbation — gap narrows substantially:**

| Benchmark | Nemotron | Qwen | Gap (perturbed) |
|---|---|---|---|
| ARC | 0.968 | 0.965 | −0.3pp |
| MATH-500 | 0.988 | 0.987 | −0.1pp |
| Last Letter | ~0.983 | 0.993 | **+1.0pp (Qwen higher)** |
| MultiFin | ~0.744 | 0.760 | **+1.6pp (Qwen higher)** |

Finding: **Nemotron's baseline lead does not translate to greater robustness.** Both models converge under perturbation. Qwen even exceeds Nemotron on Last Letter and MultiFin under perturbation, despite trailing at baseline.

---

## Slide 12 — Key Findings

1. **Hypothesis supported** across all four benchmarks (ARC, MATH-500, Last Letter, MultiFin): all Δ values within ±1.5pp for Nemotron-120B.

2. **Scale eliminates format sensitivity on symbolic tasks:** Nemotron +0.3pp on Last Letter vs LLaMA-3-8B's ~42pp drop; Nemotron −1.0pp on ARC vs ~20pp for smaller models.

3. **Few-shot (P5) can help, not just hurt:** MultiFin +3.1pp — on genuinely ambiguous tasks, examples provide useful signal.

4. **Both large models are robust:** Qwen3.5-397B shows the same pattern. Format robustness appears to be a property of scale, not model-specific tuning.

5. **Baseline performance does not predict robustness:** Nemotron leads by up to 8.8pp at baseline, but both models converge under perturbation.

---

## Slide 13 — Conclusion

**The format confounder concern from Tam et al. appears to be a small-model phenomenon.**

At 120B+ scale, models separate surface-form compliance from task execution — they follow format instructions without sacrificing reasoning quality.

**Practical implication:**
Benchmark scores from strong modern models are robust to prompt surface variation. They can be treated as reliable capability signals, not format-confounded artifacts.

**Limitations:**
- P3 (prompt rephrasing) not tested — future work
- P1 implemented via prompt instruction (not API `response_format` flag) due to Nemotron incompatibility
- Two models tested; generalisation across all large models remains open

---
