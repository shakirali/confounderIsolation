"""
Build MATH-500 perturbed prompts (5 types × each problem).

Reads baseline prompts from data/baseline/math500_baseline.csv column `prompt`.
Applies the same perturbation pattern as TruthfulQA / ARC-Challenge, with
math-specific few-shot examples for p5.

Usage (from repo root):
    .venv/bin/python src/generate_math500_perturbations.py
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


def p1_format_math(problem: str) -> str:
    """p1_format for MATH-500: problem + explicit JSON instruction (no API response_format flag).

    Nemotron returns empty content when response_format: json_object is set without a clear
    JSON schema in the prompt. Using a prompt-level instruction (like p1_format_soft) instead
    of the API flag produces a populated response.
    """
    return f'{problem}\nAnswer strictly in JSON format: {{"answer": "your final answer here"}}'


def p5_fewshot_math(prompt: str) -> str:
    """Two math examples (algebra + number theory) with worked answers, then the target problem."""
    examples = (
        "Example 1:\n"
        "Solve the following math problem. Show your reasoning, then state your final answer clearly.\n\n"
        "If $2x + 3 = 11$, what is the value of $x$?\n\n"
        "Solution: Subtract 3 from both sides: $2x = 8$. Divide by 2: $x = 4$.\n"
        "Answer: 4\n\n"
        "Example 2:\n"
        "Solve the following math problem. Show your reasoning, then state your final answer clearly.\n\n"
        "What is the remainder when $2^{10}$ is divided by $7$?\n\n"
        "Solution: $2^1=2,\\ 2^2=4,\\ 2^3=8\\equiv 1\\pmod{7}$, so the powers of 2 cycle with period 3. "
        "$10 = 3\\cdot3+1$, so $2^{10}\\equiv 2^1=2\\pmod{7}$.\n"
        "Answer: 2\n\n"
        "Now solve this problem:\n\n"
    )
    return f"{examples}{prompt}"


def generate_math500_perturbations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    json_fmt = {"type": "json_object"}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="MATH-500 perturbations"):
        uid = row["unique_id"]
        answer = row["answer"]
        subject = row["subject"]
        level = row["level"]
        prompt = row["prompt"]

        problem = row["problem"]
        sys_p4, user_p4 = p4_role(prompt)
        for ptype, prompt_sent, system_prompt, response_format in [
            ("p1_format",      p1_format_math(problem),  None,    None),
            ("p1_format_soft", p1_format_soft(prompt),   None,    None),
            ("p2_complexity",  p2_complexity(prompt),    None,    None),
            ("p4_role",        user_p4,                  sys_p4,  None),
            ("p5_fewshot",     p5_fewshot_math(prompt),  None,    None),
        ]:
            rows.append({
                "unique_id": uid,
                "answer": answer,
                "subject": subject,
                "level": level,
                "perturbation_type": ptype,
                "prompt_sent": prompt_sent,
                "system_prompt": system_prompt,
                "response_format": response_format,
            })

    return pd.DataFrame(rows)


def main():
    os.makedirs("data/perturbations", exist_ok=True)
    df = pd.read_csv("data/baseline/math500_baseline.csv")
    perturbed = generate_math500_perturbations(df)
    out_path = "data/perturbations/math500_perturbed.csv"
    perturbed.to_csv(out_path, index=False)
    print(f"\nSaved {len(perturbed)} rows to {out_path}")
    print(perturbed["perturbation_type"].value_counts().to_string())

    # Verify
    assert len(perturbed) == 2500, f"Expected 2500 rows, got {len(perturbed)}"
    assert perturbed["perturbation_type"].nunique() == 5
    assert (perturbed.groupby("unique_id")["perturbation_type"].count() == 5).all()
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
