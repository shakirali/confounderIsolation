# Specification: Confounder Isolation in AI Safety Evaluations

## Purpose

AI safety research relies on benchmarks to determine whether models are safe enough to deploy. If those benchmarks are measuring the wrong thing, safety assessments may be built on false assurances.

Bean et al. (2025) reviewed 445 LLM benchmarks and found that most suffer from **confounding subtasks** — benchmark scores are silently inflated or deflated by auxiliary skills such as instruction following and output formatting, rather than the safety property the benchmark claims to measure. This problem is identified theoretically but no empirical evidence exists quantifying how much it actually matters in practice.

This project fills that gap by empirically testing whether scores on a widely-used AI safety benchmark (**TruthfulQA**) are genuinely measuring the target property — truthfulness — or are significantly driven by surface-level confounders.

---

## Research Question

> Do safety benchmark scores reflect genuine model safety properties, or are they significantly influenced by surface-level factors like output formatting, instruction complexity, and prompt phrasing?

---

## Hypotheses

- **H1:** Model scores on TruthfulQA vary significantly across prompt perturbations that should not affect truthfulness.
- **H2:** Weaker models are disproportionately affected by formatting and instruction complexity perturbations compared to stronger models.
- **H3:** Model rankings change across perturbation conditions, indicating benchmark scores are unstable.

---

## What Is Being Measured

Five perturbation types are applied to each TruthfulQA prompt. Each perturbation changes surface-level properties only — none should logically affect a model's underlying truthfulness:

| ID | Perturbation Type | Description |
|---|---|---|
| P1 | Output Format Constraint | Requires response in a specific format (e.g. JSON) |
| P2 | Instruction Complexity | Adds verbose preamble around the same question |
| P3 | Prompt Phrasing Variants | Rephrases the question without changing its meaning |
| P4 | Role Framing | Adds a system prompt assigning a persona or role |
| P5 | Few-Shot Examples | Prepends example Q&A pairs before the question |

---

## Models

| Model | Provider |
|---|---|
| llama3.1:8b | Ollama (local) — smoke tests only |
| Llama-3.1-8B-Instruct-Turbo | Together.ai — full run |
| Llama-3.1-70B-Instruct-Turbo | Together.ai — sampled (1,000 variants) |

---

## Primary Benchmark

**TruthfulQA** (Lin et al., 2022) — 817 questions designed to measure whether models produce truthful responses. Chosen because it is widely used in AI safety contexts and has a well-defined scoring mechanism.

---

## Why This Matters for AI Safety

If a model appears safe only because it is good at following formatting instructions, that is a false assurance. The model may behave unsafely in real-world deployments where prompts are messier and more varied.

This project produces direct empirical evidence that current safety evaluation pipelines may be less reliable than the field assumes, and gives benchmark designers concrete guidance on how to control for surface-level confounders — directly addressing the gap identified by Bean et al. (2025).

---

## Deliverables

- Perturbation dataset (817 questions × 6 conditions = 4,902 prompt variants)
- Raw model responses and scores
- Statistical analysis results
- Technical report
- Public GitHub repository
- HuggingFace dataset release

---

## Key References

- Bean et al. (2025) — *Measuring what Matters: Construct Validity in LLM Benchmarks* — primary motivation
- Lin et al. (2022) — TruthfulQA original paper
- Sclar et al. (2023) — prior work on LLM prompt formatting sensitivity
- Mazeika et al. (2024) — HarmBench (secondary benchmark if time allows)
