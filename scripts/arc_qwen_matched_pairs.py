#!/usr/bin/env python3
"""
Matched-pairs accuracy analysis for ARC-Challenge Qwen3.5-397B.

For each perturbation type, restricts both baseline and perturbed accuracy
to the set of question_ids that parsed successfully in BOTH conditions,
eliminating the denominator-mismatch artefact.

Usage (repo root):
  uv run python scripts/arc_qwen_matched_pairs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent

# Batch paths
BASELINE_DIR = REPO / "experiments/doubleword_batches/arc_qwen/68555238-e404-40f3-a3f3-cd3b4ca571a3_arc_baseline_eval"
PERTURBED_DIR = REPO / "experiments/doubleword_batches/arc_qwen/34f05195-a532-4492-a97c-acc26f7d8128_arc_perturbed_eval"
PERTURBED_RETRY_DIR = REPO / "experiments/doubleword_batches/arc_qwen/9b7b2ad3-8cdc-4e26-bd35-6ee89e0a6889_arc_perturbed_retry"

# Gold CSVs
BASELINE_CSV = REPO / "data/baseline/arc_challenge_test_raw.csv"
PERTURBED_CSV = REPO / "data/perturbations/arc_challenge_test_perturbed.csv"

import re

LETTER_WORD = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
JSON_ANSWER_RE = re.compile(r'"answer"\s*:\s*"(.*?)"', re.DOTALL | re.IGNORECASE)


def strip_thinking(content: str) -> str:
    if "</think>" in content:
        return content.split("</think>", 1)[1].strip()
    if "Thinking Process:" in content:
        parts = content.rsplit("\n\n", 1)
        if len(parts) == 2:
            return parts[1].strip()
    return content


def message_text(rec: dict) -> str:
    err = rec.get("error")
    if err is not None and err != {} and err:
        return ""
    resp = rec.get("response")
    if not isinstance(resp, dict):
        return ""
    if resp.get("status_code") != 200:
        return ""
    try:
        choice = resp["body"]["choices"][0]
        msg = choice["message"]
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError):
        return ""
    content = strip_thinking((msg.get("content") or "").strip())
    if content:
        return content
    if finish_reason == "length":
        return ""
    return strip_thinking((msg.get("reasoning_content") or "").strip())


def extract_letter_json(text: str) -> str | None:
    for candidate in [text, text[text.find("{"):text.rfind("}") + 1]] if "{" in text else [text]:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                raw = obj.get("answer")
                if isinstance(raw, str):
                    m = LETTER_WORD.search(raw.upper())
                    if m:
                        return m.group(1).upper()
        except json.JSONDecodeError:
            pass
    m = JSON_ANSWER_RE.search(text)
    if m:
        m2 = LETTER_WORD.search(m.group(1).upper())
        if m2:
            return m2.group(1).upper()
    return None


def extract_letter_heuristic(text: str) -> str | None:
    if not text or not text.strip():
        return None
    t = text.strip()
    matches = LETTER_WORD.findall(t.upper())
    if matches:
        return matches[-1].upper()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        last = lines[-1].strip().upper()
        if len(last) == 1 and last in "ABCD":
            return last
    return None


def predict_letter(raw: str, ptype: str | None) -> str | None:
    if not raw:
        return None
    if ptype in ("p1_format", "p1_format_soft") or raw.strip().startswith("{"):
        j = extract_letter_json(raw)
        if j:
            return j
    h = extract_letter_heuristic(raw)
    return h


def normalize_gold(raw) -> str | None:
    s = str(raw).strip().upper()
    if len(s) == 1 and s in "ABCD":
        return s
    if s in "1234":
        return "ABCD"[int(s) - 1]
    return None


def load_output(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        records[int(rec["custom_id"])] = rec
    return records


def get_finish_reason(rec: dict) -> str | None:
    """Return finish_reason string, or None if response is errored/missing."""
    if rec is None:
        return None
    resp = rec.get("response")
    if not isinstance(resp, dict) or resp.get("status_code") != 200:
        return None
    try:
        return resp["body"]["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        return None


def has_think_close(rec: dict) -> bool:
    """Return True if the raw content contains a closing </think> tag."""
    if rec is None:
        return False
    resp = rec.get("response")
    if not isinstance(resp, dict) or resp.get("status_code") != 200:
        return False
    try:
        content = resp["body"]["choices"][0]["message"].get("content") or ""
        return "</think>" in content
    except (KeyError, IndexError, TypeError):
        return False


def score_baseline() -> pd.DataFrame:
    records = load_output(BASELINE_DIR / "output.jsonl")
    # Merge retry (overrides truncated rows)
    retry_path = BASELINE_DIR / "retry_output.jsonl"
    if retry_path.exists():
        for cid, rec in load_output(retry_path).items():
            records[cid] = rec

    gold_df = pd.read_csv(BASELINE_CSV).head(len(records)).reset_index(drop=True)
    gold_df["custom_id"] = range(len(gold_df))

    rows = []
    for _, gr in gold_df.iterrows():
        cid = int(gr["custom_id"])
        rec = records.get(cid)
        gold = normalize_gold(gr["answerKey"])
        finish = get_finish_reason(rec)
        think_complete = has_think_close(rec)
        raw = message_text(rec) if rec is not None else ""
        pred = predict_letter(raw, None)
        parsed = pred is not None
        correct = (pred == gold) if parsed else False
        rows.append({
            "question_id": int(gr["question_id"]),
            "gold": gold,
            "predicted": pred or "",
            "parsed": parsed,
            "finish_reason": finish,
            "think_complete": think_complete,
            "correct": correct,
        })
    return pd.DataFrame(rows)


def score_perturbed() -> pd.DataFrame:
    records = load_output(PERTURBED_DIR / "output.jsonl")
    # Merge retry (overrides truncated rows)
    for cid, rec in load_output(PERTURBED_RETRY_DIR / "output.jsonl").items():
        records[cid] = rec

    n_questions = len(records) // 5
    full = pd.read_csv(PERTURBED_CSV)
    qids = full["question_id"].unique()[:n_questions]
    gold_df = full[full["question_id"].isin(qids)].reset_index(drop=True)
    gold_df["custom_id"] = range(len(gold_df))

    rows = []
    for _, gr in gold_df.iterrows():
        cid = int(gr["custom_id"])
        rec = records.get(cid)
        gold = normalize_gold(gr["answerKey"])
        ptype = str(gr["perturbation_type"])
        raw = message_text(rec) if rec is not None else ""
        pred = predict_letter(raw, ptype)
        parsed = pred is not None
        correct = (pred == gold) if parsed else False
        rows.append({
            "question_id": int(gr["question_id"]),
            "perturbation_type": ptype,
            "gold": gold,
            "predicted": pred or "",
            "parsed": parsed,
            "correct": correct,
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("Scoring baseline...")
    baseline = score_baseline()
    n_stop = (baseline["finish_reason"] == "stop").sum()
    n_none = baseline["finish_reason"].isna().sum()
    n_length = (baseline["finish_reason"] == "length").sum()
    print(f"  Total: {len(baseline)}, parsed: {baseline['parsed'].sum()}")
    print(f"  finish_reason: stop={n_stop}, None={n_none}, length={n_length}")
    print(f"  Accuracy (all parsed):        {baseline.loc[baseline['parsed'], 'correct'].mean():.4f}")
    stop_parsed = baseline[(baseline["finish_reason"] == "stop") & baseline["parsed"]]
    none_parsed = baseline[baseline["finish_reason"].isna() & baseline["parsed"]]
    print(f"  Accuracy (stop rows only):    {stop_parsed['correct'].mean():.4f} (n={len(stop_parsed)})")
    print(f"  Accuracy (finish=None rows):  {none_parsed['correct'].mean():.4f} (n={len(none_parsed)})")

    print("\nScoring perturbed (with retry merge)...")
    perturbed = score_perturbed()

    ptypes = sorted(perturbed["perturbation_type"].unique())

    def run_analysis(label: str, baseline_subset: pd.DataFrame) -> None:
        clean_parsed_ids = set(baseline_subset.loc[baseline_subset["parsed"], "question_id"])

        print(f"\n--- {label} (n={len(clean_parsed_ids)} baseline questions) ---\n")
        print(f"{'Perturbation':<18} {'N matched':>10} {'Baseline acc':>13} {'Perturbed acc':>14} {'Δ':>8}")
        print("-" * 68)

        for ptype in ptypes:
            pert_sub = perturbed[perturbed["perturbation_type"] == ptype]
            pert_parsed_ids = set(pert_sub.loc[pert_sub["parsed"], "question_id"])
            matched_ids = clean_parsed_ids & pert_parsed_ids

            base_matched = baseline_subset[baseline_subset["question_id"].isin(matched_ids)]
            pert_matched = pert_sub[pert_sub["question_id"].isin(matched_ids)]

            base_acc = base_matched["correct"].mean()
            pert_acc = pert_matched["correct"].mean()
            delta = pert_acc - base_acc
            n = len(matched_ids)

            sign = "+" if delta >= 0 else ""
            print(f"{ptype:<18} {n:>10} {base_acc:>13.4f} {pert_acc:>14.4f} {sign+f'{delta:.4f}':>8}")

        all_pert_parsed = perturbed.groupby("question_id").apply(lambda g: g["parsed"].all())
        fully_parsed_ids = set(all_pert_parsed[all_pert_parsed].index) & clean_parsed_ids
        base_overall = baseline_subset[baseline_subset["question_id"].isin(fully_parsed_ids)]["correct"].mean()
        pert_overall = perturbed[perturbed["question_id"].isin(fully_parsed_ids)]["correct"].mean()
        delta_overall = pert_overall - base_overall
        sign = "+" if delta_overall >= 0 else ""
        print("-" * 68)
        print(f"{'Overall (all 5)':<18} {len(fully_parsed_ids):>10} {base_overall:>13.4f} {pert_overall:>14.4f} {sign+f'{delta_overall:.4f}':>8}")

    # Analysis 1: stop rows only
    run_analysis(
        "Matched pairs — baseline: finish_reason=stop",
        baseline[baseline["finish_reason"] == "stop"],
    )

    # Analysis 2: stop + think-complete rows (most reliable baseline)
    run_analysis(
        "Matched pairs — baseline: stop + think-complete (</think> present)",
        baseline[(baseline["finish_reason"] == "stop") & baseline["think_complete"]],
    )

    print()
    print("Notes:")
    print("  - 99 baseline rows had finish_reason=None (aborted); 33.7% accuracy → excluded")
    print("  - Of stop rows: 101 lacked </think> tag → 38.6% accuracy (incomplete thinking)")
    print("  - Think-complete stop rows: 971 rows, 98.5% accuracy — most reliable baseline")
    print("  - Perturbed batch: 5846/5860 stop, 1 None, 13 length — nearly all clean")


if __name__ == "__main__":
    main()
