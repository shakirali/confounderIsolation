#!/usr/bin/env python3
"""
Score ARC-Challenge Doubleword batch output vs gold answerKey (deterministic, no judge).

Joins output.jsonl rows (by custom_id) to the same CSV rows used when the batch was built:
  - Baseline: first N rows of data/baseline/arc_challenge_test_raw.csv (custom_id = row index).
  - Perturbed: first K unique question_ids × 5 types from data/perturbations/arc_challenge_test_perturbed.csv,
    using the same filter as perturbed_eval_smoke_test.py.

Predicted letter:
  - p1_format: try JSON (message.content) for "answer", else fall back to letter extraction.
  - Other types / failures: extract A–D from content (prefer last \\b[A-D]\\b token; then last line single letter).

Scoring: correct is 1 if predicted == gold, 0 if parsed but wrong, -1 if parse/API failure.
Gold ``answerKey`` may be A–D or digits 1–4 (HF index); both are normalized to A–D.

Usage (repo root):
  uv run python scripts/score_arc_mcq.py \\
    --output-jsonl experiments/doubleword_batches/arc/0861314c-4091-4c14-8d7f-09e7acae6289_arc_baseline_eval/output.jsonl \\
    --baseline

  uv run python scripts/score_arc_mcq.py --output-jsonl .../output.jsonl --perturbed --n-questions 10

  uv run python scripts/score_arc_mcq.py ... --out-csv results/scored_arc.csv

When a choice has ``finish_reason: "length"``, only ``content`` is used; no ``reasoning_content`` fallback.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

LETTER_WORD = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
# JSON "answer" field (string or bare token)
JSON_ANSWER_RE = re.compile(
    r'"answer"\s*:\s*"(.*?)"',
    re.DOTALL | re.IGNORECASE,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def strip_thinking(content: str) -> str:
    """Strip Qwen-style <think>...</think> reasoning block, return text after it."""
    if "</think>" in content:
        return content.split("</think>", 1)[1].strip()
    if "Thinking Process:" in content:
        parts = content.rsplit("\n\n", 1)
        if len(parts) == 2:
            return parts[1].strip()
    return content


def message_text_from_record(rec: dict, *, content_only: bool) -> str:
    """Visible assistant text for scoring: ``message.content``, else ``reasoning_content`` when allowed.

    If ``finish_reason`` is ``length``, use **only** ``content`` (even when empty). Do not fall back to
    ``reasoning_content``; truncated generations are left as parse failures without scraping traces.
    """
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
    if content_only:
        return content
    if content:
        return content
    if finish_reason == "length":
        return ""
    return strip_thinking((msg.get("reasoning_content") or "").strip())


def extract_letter_from_json_text(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    # Direct JSON object
    for candidate in _json_candidates(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        raw = obj.get("answer")
        if raw is None:
            continue
        if isinstance(raw, str):
            m = LETTER_WORD.search(raw.upper())
            if m:
                return m.group(1).upper()
        elif isinstance(raw, (int, float)):
            continue
    m = JSON_ANSWER_RE.search(text)
    if m:
        inner = m.group(1)
        m2 = LETTER_WORD.search(inner.upper())
        if m2:
            return m2.group(1).upper()
    return None


def _json_candidates(text: str) -> list[str]:
    out = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        out.append(text[start : end + 1])
    return out


def extract_letter_heuristic(text: str) -> str | None:
    """Last A–D as standalone token; else last non-empty line if it is exactly one letter."""
    if not text or not str(text).strip():
        return None
    t = str(text).strip()
    matches = LETTER_WORD.findall(t.upper())
    if matches:
        return matches[-1].upper()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        last = lines[-1].strip().upper()
        if len(last) == 1 and last in "ABCD":
            return last
    return None


def normalize_gold_answer_key(raw) -> str | None:
    """HF ARC labels may be A-D or digits 1-4 (option index). Map 1→A … 4→D."""
    s = str(raw).strip().upper()
    if len(s) == 1 and s in "ABCD":
        return s
    if s in "1234":
        return "ABCD"[int(s) - 1]
    return None


def predict_letter(raw: str, perturbation_type: str | None) -> tuple[str | None, str]:
    """
    Returns (letter_upper_or_None, method).
    method in {json, heuristic, empty}.
    """
    if not raw:
        return None, "empty"
    pt = perturbation_type or ""
    if pt == "p1_format" or raw.strip().startswith("{"):
        j = extract_letter_from_json_text(raw)
        if j:
            return j, "json"
    if pt == "p1_format_soft":
        j = extract_letter_from_json_text(raw)
        if j:
            return j, "json"
    h = extract_letter_heuristic(raw)
    if h:
        return h, "heuristic"
    return None, "unparsed"


def load_output_jsonl(path: Path) -> dict[int, dict]:
    by_id: dict[int, dict] = {}
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        cid = int(rec["custom_id"])
        by_id[cid] = rec
    return by_id


def baseline_gold_table(csv_path: Path, n: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path).head(n).reset_index(drop=True)
    df["custom_id"] = range(len(df))
    return df


def perturbed_gold_table(csv_path: Path, n_questions: int) -> pd.DataFrame:
    full = pd.read_csv(csv_path)
    qids = full["question_id"].unique()[:n_questions]
    df = full[full["question_id"].isin(qids)].reset_index(drop=True)
    df["custom_id"] = range(len(df))
    return df


def main() -> int:
    root = _repo_root()
    p = argparse.ArgumentParser(description="Score ARC MCQ batch output vs answerKey")
    p.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Path to batch output.jsonl",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--baseline",
        action="store_true",
        help="Use arc_challenge_test_raw.csv (first N rows); N = number of JSONL rows",
    )
    g.add_argument(
        "--perturbed",
        action="store_true",
        help="Use arc_challenge_test_perturbed.csv; set --n-questions to match the eval run",
    )
    p.add_argument(
        "--baseline-csv",
        type=Path,
        default=None,
        help="Default: <repo>/data/baseline/arc_challenge_test_raw.csv",
    )
    p.add_argument(
        "--perturbed-csv",
        type=Path,
        default=None,
        help="Default: <repo>/data/perturbations/arc_challenge_test_perturbed.csv",
    )
    p.add_argument(
        "--n-questions",
        type=int,
        default=None,
        help="Perturbed only: first K unique question_ids (must match arc_perturbed_eval --n)",
    )
    p.add_argument(
        "--content-only",
        action="store_true",
        help="Parse only message.content (no reasoning_content fallback)",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Write scored rows to this CSV",
    )
    args = p.parse_args()

    out_path = args.output_jsonl
    if not out_path.is_file():
        print(f"Missing output file: {out_path}", file=sys.stderr)
        return 1

    records = load_output_jsonl(out_path)
    n_lines = len(records)
    if n_lines == 0:
        print("No rows in output.jsonl", file=sys.stderr)
        return 1
    max_id = max(records.keys())
    expected = max_id + 1
    if len(records) != expected:
        print(
            f"Warning: custom_id set has {len(records)} rows but max_id={max_id} (expected {expected} contiguous ids).",
            file=sys.stderr,
        )

    if args.baseline:
        csv_path = args.baseline_csv or (root / "data" / "baseline" / "arc_challenge_test_raw.csv")
        if not csv_path.is_file():
            print(f"Missing baseline CSV: {csv_path}", file=sys.stderr)
            return 1
        gold_df = baseline_gold_table(csv_path, n_lines)
        pert_col = None
    else:
        if args.n_questions is None:
            print("--n-questions is required with --perturbed", file=sys.stderr)
            return 1
        csv_path = args.perturbed_csv or (root / "data" / "perturbations" / "arc_challenge_test_perturbed.csv")
        if not csv_path.is_file():
            print(f"Missing perturbed CSV: {csv_path}", file=sys.stderr)
            return 1
        gold_df = perturbed_gold_table(csv_path, args.n_questions)
        pert_col = "perturbation_type"
        if len(gold_df) != n_lines:
            print(
                f"Row count mismatch: output.jsonl has {n_lines} rows, "
                f"filtered perturbed CSV has {len(gold_df)} (n_questions={args.n_questions}).",
                file=sys.stderr,
            )
            return 1

    rows_out: list[dict] = []
    correct = 0
    wrong = 0
    parse_fail = 0

    for _, gr in gold_df.iterrows():
        cid = int(gr["custom_id"])
        rec = records.get(cid)
        gold = normalize_gold_answer_key(gr["answerKey"])
        if gold is None:
            print(f"Bad gold answerKey for custom_id={cid}: {gr['answerKey']!r}", file=sys.stderr)
            return 1

        ptype = str(gr[pert_col]) if pert_col else None
        raw = ""
        parse_method = "missing_row"
        if rec is not None:
            raw = message_text_from_record(rec, content_only=args.content_only)
            pred, parse_method = predict_letter(raw, ptype)
        else:
            pred = None

        if pred is None:
            score = -1
            parse_fail += 1
        elif pred == gold:
            score = 1
            correct += 1
        else:
            score = 0
            wrong += 1

        row = {
            "custom_id": cid,
            "question_id": gr["question_id"],
            "arc_id": gr.get("arc_id", ""),
            "answerKey": gold,
            "predicted": pred if pred is not None else "",
            "correct": score,
            "parse_method": parse_method,
            "raw_response": raw[:2000] if raw else "",
        }
        if ptype is not None:
            row["perturbation_type"] = ptype
        rows_out.append(row)

    scored = pd.DataFrame(rows_out)
    valid = scored[scored["correct"] != -1]
    n_valid = len(valid)
    acc = valid["correct"].mean() if n_valid else float("nan")

    print(f"Scored: {out_path}")
    print(f"Gold CSV: {csv_path}")
    print(f"Rows: {len(scored)} | correct: {correct} | wrong: {wrong} | parse_fail (-1): {parse_fail}")
    if n_valid:
        print(f"Accuracy (parsed only): {acc:.4f} ({correct}/{n_valid})")
    print("\nParse methods:")
    print(scored["parse_method"].value_counts().to_string())

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(args.out_csv, index=False)
        print(f"\nWrote {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
