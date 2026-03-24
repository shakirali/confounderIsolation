"""
Build ARC-Challenge perturbed prompts (5 types × each test question).

Reads baseline MCQ text from data/baseline/arc_challenge_test_raw.csv column `prompt`.
Applies the same perturbation pattern as TruthfulQA (generate_perturbations.py), with
science MCQ few-shots for p5.

Usage (from repo root):
    uv run python src/generate_arc_perturbations.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

_src = Path(__file__).resolve().parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from generate_perturbations import p1_format, p1_format_soft, p2_complexity, p4_role

load_dotenv()


def p5_fewshot_arc(mcq_block: str) -> str:
    """Two short science MCQ examples (stem + options + gold letter), then the target item."""
    examples = (
        "Example 1:\n"
        "What gas do plants primarily release during photosynthesis?\n\n"
        "Options:\n"
        "A) Carbon dioxide\n"
        "B) Oxygen\n"
        "C) Nitrogen\n"
        "D) Methane\n\n"
        "Answer with only the letter of the correct option (A, B, C, or D).\n"
        "Answer: B\n\n"
        "Example 2:\n"
        "Which planet is closest to the Sun?\n\n"
        "Options:\n"
        "A) Venus\n"
        "B) Earth\n"
        "C) Mercury\n"
        "D) Mars\n\n"
        "Answer with only the letter of the correct option (A, B, C, or D).\n"
        "Answer: C\n\n"
        "Now answer this question:\n\n"
    )
    return f"{examples}{mcq_block}"


def generate_arc_perturbations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    json_fmt = {"type": "json_object"}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ARC perturbations"):
        q_id = row["question_id"]
        stem = row["question"]
        mcq = row["prompt"]
        arc_id = row["arc_id"]
        answer_key = row["answerKey"]
        choices_json = row["choices_json"]

        sys_p4, user_p4 = p4_role(mcq)
        rows.append({
            "question_id": q_id,
            "arc_id": arc_id,
            "answerKey": answer_key,
            "choices_json": choices_json,
            "question": stem,
            "perturbation_type": "p1_format",
            "prompt_sent": p1_format(mcq),
            "system_prompt": None,
            "response_format": json_fmt,
        })
        rows.append({
            "question_id": q_id,
            "arc_id": arc_id,
            "answerKey": answer_key,
            "choices_json": choices_json,
            "question": stem,
            "perturbation_type": "p1_format_soft",
            "prompt_sent": p1_format_soft(mcq),
            "system_prompt": None,
            "response_format": None,
        })
        rows.append({
            "question_id": q_id,
            "arc_id": arc_id,
            "answerKey": answer_key,
            "choices_json": choices_json,
            "question": stem,
            "perturbation_type": "p2_complexity",
            "prompt_sent": p2_complexity(mcq),
            "system_prompt": None,
            "response_format": None,
        })
        rows.append({
            "question_id": q_id,
            "arc_id": arc_id,
            "answerKey": answer_key,
            "choices_json": choices_json,
            "question": stem,
            "perturbation_type": "p4_role",
            "prompt_sent": user_p4,
            "system_prompt": sys_p4,
            "response_format": None,
        })
        rows.append({
            "question_id": q_id,
            "arc_id": arc_id,
            "answerKey": answer_key,
            "choices_json": choices_json,
            "question": stem,
            "perturbation_type": "p5_fewshot",
            "prompt_sent": p5_fewshot_arc(mcq),
            "system_prompt": None,
            "response_format": None,
        })

    return pd.DataFrame(rows)


def main():
    os.makedirs("data/perturbations", exist_ok=True)
    df = pd.read_csv("data/baseline/arc_challenge_test_raw.csv")
    perturbed = generate_arc_perturbations(df)
    out_path = "data/perturbations/arc_challenge_test_perturbed.csv"
    perturbed.to_csv(out_path, index=False)
    print(f"\nSaved {len(perturbed)} rows to {out_path}")
    print(perturbed["perturbation_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
