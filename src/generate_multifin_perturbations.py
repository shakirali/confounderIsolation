"""
Build MultiFin perturbed prompts (5 types × each example).

Reads baseline from data/baseline/multifin_baseline.csv.
Options (A–F) and label set are fixed across all examples.

p1_format uses a prompt-level JSON instruction (no API response_format flag) —
Nemotron returns empty content when response_format: json_object is set.

Usage (from repo root):
    .venv/bin/python src/generate_multifin_perturbations.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_src = Path(__file__).resolve().parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from generate_perturbations import p2_complexity, p4_role
from load_multifin import LABELS, LETTERS, LABEL_TO_LETTER, build_prompt

_LETTERS_STR = ", ".join(LETTERS)
_OPTIONS_BLOCK = "\n".join(f"{l}. {lab}" for l, lab in zip(LETTERS, LABELS))

# Two fixed few-shot examples from the training set (different labels)
_FEWSHOT_EXAMPLES = (
    'Example 1:\n'
    'Headline: "Legal Flash: Social security of freelancers update"\n'
    'Answer: B\n\n'
    'Example 2:\n'
    'Headline: "Revenue Recognition"\n'
    'Answer: E\n\n'
    'Now classify this:\n'
)


def p1_format_multifin(prompt: str) -> str:
    """Prompt-level JSON instruction (no API flag — Nemotron incompatible)."""
    return f'{prompt}\nAnswer strictly in JSON format: {{"answer": "letter here"}}'


def p1_format_soft_multifin(prompt: str) -> str:
    return f'{prompt}\nAnswer in JSON format: {{"answer": "letter here"}}'


def p5_fewshot_multifin(prompt: str) -> str:
    """Prepend two worked classification examples, then the target prompt."""
    return f"{_FEWSHOT_EXAMPLES}{prompt}"


def generate_multifin_perturbations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="MultiFin perturbations"):
        uid = row["unique_id"]
        label = row["label"]
        answer_key = row["answer_key"]
        prompt = row["prompt"]

        sys_p4, user_p4 = p4_role(prompt)
        for ptype, prompt_sent, system_prompt in [
            ("p1_format",      p1_format_multifin(prompt),       None),
            ("p1_format_soft", p1_format_soft_multifin(prompt),  None),
            ("p2_complexity",  p2_complexity(prompt),             None),
            ("p4_role",        user_p4,                          sys_p4),
            ("p5_fewshot",     p5_fewshot_multifin(prompt),      None),
        ]:
            rows.append({
                "unique_id": uid,
                "text": row["text"],
                "label": label,
                "answer_key": answer_key,
                "perturbation_type": ptype,
                "prompt_sent": prompt_sent,
                "system_prompt": system_prompt,
                "response_format": None,
            })

    return pd.DataFrame(rows)


def main():
    os.makedirs("data/perturbations", exist_ok=True)
    df = pd.read_csv("data/baseline/multifin_baseline.csv")
    perturbed = generate_multifin_perturbations(df)
    out_path = "data/perturbations/multifin_perturbed.csv"
    perturbed.to_csv(out_path, index=False)
    print(f"\nSaved {len(perturbed)} rows to {out_path}")
    print(perturbed["perturbation_type"].value_counts().to_string())

    assert len(perturbed) == len(df) * 5, f"Expected {len(df)*5} rows, got {len(perturbed)}"
    assert perturbed["perturbation_type"].nunique() == 5
    print("\nAll assertions passed.")

    print("\n--- Sample prompts (first row, each type) ---")
    for ptype in ["p1_format", "p1_format_soft", "p2_complexity", "p4_role", "p5_fewshot"]:
        sample = perturbed[perturbed["perturbation_type"] == ptype].iloc[0]
        print(f"\n[{ptype}]")
        print(sample["prompt_sent"][:200])


if __name__ == "__main__":
    main()
