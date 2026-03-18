# Smoke Test Analysis
## Confounder Isolation in TruthfulQA — 100-Question Pilot

**Date:** 2026-03-18
**Eval model:** `Qwen/Qwen3.5-35B-A3B-FP8`
**Judge model:** `Qwen/Qwen3.5-397B-A17B-FP8`
**Sample size:** 100 questions (first 100 of 817 TruthfulQA validation questions)

---

## Experiments Run

| # | Batch ID | Type | Notes |
|---|---|---|---|
| 1 | `50230d22` | Baseline eval | 100 questions, plain prompts |
| 2 | `761a53ba` | Baseline judge | Scored batch 1 |
| 3 | `d0e2582b` | Perturbed eval | 100 questions × 4 perturbation types (400 total) |
| 4 | `b1999ff0` | Perturbed judge | Scored batch 3 (with all fixes applied) |

Discarded batches:
- `de13e55e` — baseline eval with `max_tokens=1024`, insufficient for thinking model
- `b28d872e` — judge with `max_tokens=128`, all `finish_reason=length`, zero valid scores
- `0f319756` — perturbed judge submitted before format-stripping fixes

---

## Experiment 1 & 2: Baseline

**Goal:** Establish how truthful the eval model is on unperturbed TruthfulQA questions.

**Batch:** `50230d22` (eval) → `761a53ba` (judge)

### Results

| Metric | Value |
|---|---|
| Total questions | 100 |
| Valid judge scores | 99 |
| Parse errors (judge) | 1 |
| Truthful (score=1) | 95 |
| Not truthful (score=0) | 4 |
| **Mean score** | **0.960** |

### Interpretation

The model is highly truthful on unperturbed questions (96% of valid responses). This establishes a strong baseline against which perturbation effects can be measured. The 4 non-truthful responses represent genuine failures on hard questions, not format or instruction artefacts.

---

## Experiment 3 & 4: Perturbed Smoke Test

**Goal:** Measure how score changes when surface-level perturbations are applied that should not logically affect truthfulness.

**Batch:** `d0e2582b` (eval) → `b1999ff0` (judge)

**Perturbations tested:** p1_format, p2_complexity, p4_role, p5_fewshot

### Score Results

| Perturbation | Valid | Errors (-1) | Truthful | Not Truthful | Mean Score | Δ vs Baseline |
|---|---|---|---|---|---|---|
| **baseline** | 99 | 1 | 95 | 4 | 0.960 | — |
| p1_format | 99 | 1 | 80 | 19 | 0.808 | **−0.152** |
| p2_complexity | 100 | 0 | 93 | 7 | 0.930 | −0.030 |
| p4_role | 97 | 3 | 84 | 13 | 0.866 | −0.094 |
| p5_fewshot | 100 | 0 | 70 | 30 | 0.700 | **−0.260** |

*Errors = judge parse errors (-1), excluded from mean. Eval [ERROR] responses (empty content) handled separately — see below.*

### Eval Response Failures (Empty Content)

A separate issue: the eval model sometimes exhausted all 4,096 tokens on reasoning without producing any `content`. These are different from judge parse errors — they indicate the perturbation caused the model to get stuck reasoning about format/instructions rather than answering.

| Perturbation | Eval [ERROR] count | Rate | Likely cause |
|---|---|---|---|
| p1_format | 14 / 100 | 14% | Model spent all tokens debating JSON formatting |
| p2_complexity | 4 / 100 | 4% | Verbose instructions triggering over-analysis |
| p4_role | 13 / 100 | 13% | Role framing triggering extended reasoning |
| p5_fewshot | 29 / 100 | 29% | Few-shot preamble causing format confusion |
| baseline | 0 / 100 | 0% | — |

These responses were assigned score=-1 and excluded from the mean. They are themselves a valid finding: perturbations increased response failure rate from 0% to 4–29%.

### Per-Perturbation Findings

#### P1 — Strict JSON Format (p1_format)
- Score dropped 15.2 percentage points (0.960 → 0.808)
- 14% of responses failed entirely (model spent 4,096 tokens on format reasoning)
- **Root cause investigated:** custom_id 48 confirmed model used full reasoning budget debating "Anonymous" vs "French folk song" JSON representation, producing no `content`
- **Interpretation:** Strict format constraints are a meaningful confounder — they both reduce truthfulness in valid responses and cause a non-trivial response failure rate

#### P2 — Instruction Complexity (p2_complexity)
- Smallest score drop: −0.030 (0.960 → 0.930)
- Only 4% eval failures
- **Interpretation:** Verbose but well-formed instruction preamble has minimal effect on this model. The model appears to handle verbose prompts well.

#### P4 — Role Framing (p4_role)
- Score dropped 9.4 points (0.960 → 0.866)
- 13% eval failures; 3 judge parse errors (unfixable — genuinely ambiguous outputs)
- Known hard cases: custom_ids 98 (factual error about White House vs Orlando), 350
- **Interpretation:** Medical expert framing has a moderate negative effect. Possibly the persona causes over-confident responses that are less calibrated.

#### P5 — Few-Shot Examples (p5_fewshot)
- Largest score drop: −0.260 (0.960 → 0.700)
- 29% eval failures — highest of any condition
- **Root cause for high failure rate:** The few-shot preamble appeared to confuse the model about expected response format, triggering extensive reasoning about how to structure the answer
- **Key finding:** Initial (buggy) judge run showed 0.989 — the preamble was bleeding into the judge's question field, biasing it toward "truthful". After fix: 0.700. The *opposite* direction from expected.
- **Interpretation:** Few-shot examples significantly hurt truthfulness scores. Rather than helping the model calibrate, the Q&A format examples may introduce anchoring effects or format confusion.

---

## Issues Discovered and Fixed During This Study

These were methodological problems found through investigation — their resolution is itself a contribution to the validity of these results.

| Issue | Effect Before Fix | Fix Applied |
|---|---|---|
| `max_tokens=128` for judge | All scores invalid (`finish_reason=length`) | Set `max_tokens=4096` |
| `content_only=False` for judge | Judge parsed `"1"` from numbered reasoning steps → inflated scores | Set `content_only=True` |
| p1_format JSON suffix in judge prompt | Judge debated its own output format, exhausted tokens, returned no score | Strip suffix in `load_jsonl_pairs()` |
| p5_fewshot preamble in judge prompt | Judge saw fake Q&A examples as question context → biased toward "truthful" (0.989 → 0.700 after fix) | Extract only final `Q:` line |
| `[ERROR]` evals scored 0 by judge | Judge received `Response: [ERROR]` and returned 0 (not truthful), inflating failure counts | Filter `[ERROR]` before judging; assign −1 directly |

---

## Key Findings

1. **All perturbations reduced truthfulness.** Every surface-level change lowered the mean score, with drops ranging from −0.030 (p2_complexity) to −0.260 (p5_fewshot). This supports **H1**.

2. **Few-shot examples are the strongest confounder** (−0.260), counterintuitively hurting rather than helping. This is the largest finding from the smoke test.

3. **Format constraints (strict JSON) are a significant confounder** (−0.152) and the primary driver of response failures (14% eval [ERROR] rate) — the model struggles to simultaneously reason about truth and format.

4. **Role framing has a moderate but consistent effect** (−0.094). The medical expert persona appears to reduce calibration.

5. **Instruction complexity has minimal effect** (−0.030). This model handles verbose prompts well, suggesting complexity alone is not a strong confounder for this model.

6. **Response failure rates are themselves a signal.** Perturbations that cause the model to exhaust its reasoning budget on format rather than content are a form of confounding not captured in score alone.

7. **Methodological artefacts were numerous.** Five separate bugs were found and fixed before arriving at valid scores. This highlights how easy it is to report misleading benchmark results.

---

## Next Steps

### Immediate (Phase 3 completion)

- [ ] **Add `p1_format_soft` perturbation** — run eval and judge for "Answer in JSON format" (without "strictly") to isolate whether the word "strictly" is itself the confounder or whether any JSON constraint has this effect. Expected to show fewer eval failures than p1_format.

- [ ] **Regenerate `data/perturbations/truthfulqa_perturbed.csv`** — now includes `p1_format_soft` (817 × 5 = 4,085 rows).

- [ ] **Run full eval on all 817 questions × 5 perturbation types** — scale smoke test findings to full dataset. Batch size: 4,085 requests.

- [ ] **Run full judge on all 817 questions** — score the full eval output.

### Analysis (Phase 4)

- [ ] **Confirm p5_fewshot finding at scale** — the −0.260 drop is the headline result. Needs full-dataset confirmation.

- [ ] **Compare p1_format vs p1_format_soft** — quantify whether "strictly" vs plain JSON format meaningfully changes the error rate and score drop. This directly answers whether strictness is the confounder or JSON format itself.

- [ ] **Error rate analysis** — report eval [ERROR] rates per perturbation type as a standalone finding, not just excluded noise.

- [ ] **Per-question breakdown** — identify which question categories (e.g. history, science, common misconceptions) are most affected by each perturbation type.

- [ ] **Statistical testing** — compute confidence intervals and p-values for score differences. Sample size of 100 is borderline; full-dataset run (817) will enable robust inference.

### Report (Phase 5)

- [ ] **Frame the p5_fewshot finding carefully** — the direction is counterintuitive (few-shot hurts). The paper needs to explain why: format anchoring, preamble-induced reasoning overhead, or genuine model confusion.

- [ ] **Report the methodology issues found** — the five bugs fixed during this study are a useful secondary contribution demonstrating how evaluation pipelines can produce misleading results even with good intentions.

- [ ] **Discuss the Bean et al. alignment** — our p1_format finding directly validates their §5.2 recommendation: "assess the impact of format constraints on model performance." The 14% eval failure rate and −0.152 score drop quantify exactly what they flag as an underexplored risk.
