"""
Load DDXPlus (test split, n=1000) from HuggingFace and save baseline CSV.

Dataset: aai530-group6/ddxplus
Task: MCQ diagnosis — given decoded symptoms, pick the correct condition from
the patient's differential diagnosis list.

Evidence codes (E_###) are decoded to natural language using release_evidences.json.
MCQ options are drawn from DIFFERENTIAL_DIAGNOSIS (shuffled); gold = PATHOLOGY letter.

Usage (from repo root):
    uv run python src/load_ddxplus.py
"""

import ast
import json
import random
import string
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset

_REPO = Path(__file__).resolve().parent.parent
_N_SAMPLE = 1000
_SEED = 42


def load_release_evidences() -> dict:
    """Load evidence code → metadata from HuggingFace repo file."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="aai530-group6/ddxplus",
        filename="release_evidences.json",
        repo_type="dataset",
    )
    with open(path) as f:
        return json.load(f)


def decode_evidence(evidence_code: str, ev_map: dict) -> str:
    """
    Decode a single evidence code to a natural language statement.

    Formats:
      - "E_91"           → boolean: question only (implied Yes)
      - "E_55_@_V_123"  → categorical/multi: "question: value"
    """
    if "_@_" in evidence_code:
        parts = evidence_code.split("_@_", 1)
        code = parts[0]
        value_code = parts[1]
    else:
        code = evidence_code
        value_code = None

    ev = ev_map.get(code, {})
    question = ev.get("question_en", code).rstrip("?").strip()

    if value_code is None:
        # Boolean — presence means Yes
        return f"{question}: Yes"
    else:
        # Categorical or multi-select — look up value meaning
        value_meaning = ev.get("value_meaning", {})
        val_label = value_meaning.get(value_code, {}).get("en", value_code)
        return f"{question}: {val_label}"


def build_prompt(age: int, sex: str, decoded_symptoms: list[str], options: list[str]) -> str:
    """Build the MCQ prompt for a patient."""
    sex_str = "Male" if sex == "M" else "Female"
    letters = list(string.ascii_uppercase[: len(options)])

    lines = [
        "A patient presents with the following:",
        f"Age: {age}, Sex: {sex_str}",
        "",
        "Reported symptoms and history:",
    ]
    for sym in decoded_symptoms:
        lines.append(f"- {sym}")
    lines.append("")
    lines.append("Select the most likely diagnosis from the options below:")
    for letter, opt in zip(letters, options):
        lines.append(f"{letter}. {opt}")
    lines.append("")
    lines.append(f"Respond with only the letter ({', '.join(letters)}).")
    return "\n".join(lines)


def load_ddxplus(n: int = _N_SAMPLE, seed: int = _SEED) -> pd.DataFrame:
    ev_map = load_release_evidences()

    ds = load_dataset("aai530-group6/ddxplus", split="test")
    # Take first n rows (deterministic; test split is already fixed)
    ds = ds.select(range(n))

    rng = random.Random(seed)
    rows = []
    for i, ex in enumerate(ds):
        age = ex["AGE"]
        sex = ex["SEX"]
        pathology = ex["PATHOLOGY"]

        # Decode evidences (stored as Python literal, not JSON)
        evidences_raw = ast.literal_eval(ex["EVIDENCES"]) if isinstance(ex["EVIDENCES"], str) else ex["EVIDENCES"]
        decoded = [decode_evidence(e, ev_map) for e in evidences_raw]

        # Build MCQ options from differential diagnosis (includes gold + candidates)
        diff_raw = ast.literal_eval(ex["DIFFERENTIAL_DIAGNOSIS"]) if isinstance(ex["DIFFERENTIAL_DIAGNOSIS"], str) else ex["DIFFERENTIAL_DIAGNOSIS"]
        # diff_raw is a list of [condition_name, probability] pairs
        candidates = [item[0] for item in diff_raw]
        # Ensure gold is in candidates (should always be)
        if pathology not in candidates:
            candidates.insert(0, pathology)
        # Shuffle options
        options = candidates[:]
        rng.shuffle(options)

        letters = list(string.ascii_uppercase[: len(options)])
        answer_key = letters[options.index(pathology)]

        prompt = build_prompt(age, sex, decoded, options)

        rows.append({
            "unique_id": f"ddx_{i:04d}",
            "age": age,
            "sex": sex,
            "pathology": pathology,
            "answer_key": answer_key,
            "candidates_json": json.dumps(options),
            "prompt": prompt,
        })

    return pd.DataFrame(rows)


def main():
    os.makedirs("data/baseline", exist_ok=True)
    df = load_ddxplus()
    out_path = "data/baseline/ddxplus_baseline.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    assert len(df) == _N_SAMPLE, f"Expected {_N_SAMPLE} rows, got {len(df)}"
    assert df["unique_id"].nunique() == len(df), "unique_id values are not unique"
    assert df["answer_key"].notna().all(), "Some rows have missing answer_key"
    # Verify gold is always in options
    for _, row in df.iterrows():
        opts = json.loads(row["candidates_json"])
        letters = list(string.ascii_uppercase[: len(opts)])
        gold_idx = letters.index(row["answer_key"])
        assert opts[gold_idx] == row["pathology"], f"Gold mismatch at {row['unique_id']}"
    print("Verification passed.")
    print(df[["unique_id", "age", "sex", "pathology", "answer_key"]].head(5).to_string())
    print("\nSample prompt:\n")
    print(df["prompt"].iloc[0])


if __name__ == "__main__":
    main()
