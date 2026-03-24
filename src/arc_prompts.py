"""MCQ prompt formatting for ARC-Challenge (allenai/ai2_arc)."""

import json
from typing import Any


def format_baseline_mcq(question: str, choices: dict[str, Any]) -> str:
    """
    Build the default user message: stem, labeled options (A–D), one-line answer instruction.

    `choices` matches the HF field: {"label": ["A", ...], "text": ["...", ...]}.
    """
    labels = choices["label"]
    texts = choices["text"]
    lines = [question.strip(), "", "Options:"]
    for lab, txt in zip(labels, texts, strict=True):
        lines.append(f"{lab}) {txt}")
    lines.append("")
    lines.append("Answer with only the letter of the correct option (A, B, C, or D).")
    return "\n".join(lines)


def choices_to_json(choices: dict[str, Any]) -> str:
    return json.dumps(choices, ensure_ascii=False)
