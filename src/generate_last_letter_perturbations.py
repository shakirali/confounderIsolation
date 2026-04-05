"""
Build Last Letter Concatenation perturbed prompts (5 types × each example).

Reads baseline from data/baseline/last_letter_baseline.csv (column `prompt`).
Applies the same perturbation pattern as MATH-500 / ARC-Challenge.

p1_format uses a prompt-level JSON instruction (no API response_format flag)
because Nemotron returns empty content when response_format: json_object is set
alongside symbolic reasoning tasks.

Usage (from repo root):
    .venv/bin/python src/generate_last_letter_perturbations.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_src = Path(__file__).resolve().parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from generate_perturbations import p1_format_soft, p2_complexity, p4_role


def p1_format_last_letter(prompt: str) -> str:
    """p1_format for Last Letter: prompt + JSON instruction (no API response_format flag).

    Nemotron silently returns empty content when response_format: json_object is set
    (same behaviour as MATH-500). Using a prompt-level instruction instead.
    """
    return f'{prompt}\nAnswer strictly in JSON format: {{"answer": "your concatenated letters here"}}'


def p5_fewshot_last_letter(prompt: str) -> str:
    """Two worked examples of last-letter concatenation, then the target question."""
    examples = (
        'Example 1:\n'
        'Take the last letters of the words in "Elon Musk" and concatenate them.\n'
        'Answer: nk\n\n'
        'Example 2:\n'
        'Take the last letters of the words in "Barack Obama" and concatenate them.\n'
        'Answer: ka\n\n'
        'Now answer this:\n'
    )
    return f"{examples}{prompt}"


def generate_last_letter_perturbations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Last Letter perturbations"):
        uid = row["unique_id"]
        answer = row["answer"]
        full_name = row["full_name"]
        prompt = row["prompt"]

        sys_p4, user_p4 = p4_role(prompt)
        for ptype, prompt_sent, system_prompt, response_format in [
            ("p1_format",      p1_format_last_letter(prompt), None,   None),
            ("p1_format_soft", p1_format_soft(prompt),        None,   None),
            ("p2_complexity",  p2_complexity(prompt),         None,   None),
            ("p4_role",        user_p4,                       sys_p4, None),
            ("p5_fewshot",     p5_fewshot_last_letter(prompt),None,   None),
        ]:
            rows.append({
                "unique_id": uid,
                "full_name": full_name,
                "answer": answer,
                "perturbation_type": ptype,
                "prompt_sent": prompt_sent,
                "system_prompt": system_prompt,
                "response_format": response_format,
            })

    return pd.DataFrame(rows)


def main():
    os.makedirs("data/perturbations", exist_ok=True)
    df = pd.read_csv("data/baseline/last_letter_baseline.csv")
    perturbed = generate_last_letter_perturbations(df)
    out_path = "data/perturbations/last_letter_perturbed.csv"
    perturbed.to_csv(out_path, index=False)
    print(f"\nSaved {len(perturbed)} rows to {out_path}")
    print(perturbed["perturbation_type"].value_counts().to_string())

    assert len(perturbed) == 5000, f"Expected 5000 rows, got {len(perturbed)}"
    assert perturbed["perturbation_type"].nunique() == 5
    assert (perturbed.groupby("unique_id")["perturbation_type"].count() == 5).all()
    print("\nAll assertions passed.")

    # Show one example of each type
    print("\n--- Sample prompts ---")
    for ptype in perturbed["perturbation_type"].unique():
        sample = perturbed[perturbed["perturbation_type"] == ptype].iloc[0]
        print(f"\n[{ptype}] full_name={sample['full_name']} | answer={sample['answer']}")
        print(f"  prompt_sent: {sample['prompt_sent'][:120]!r}")


if __name__ == "__main__":
    main()
