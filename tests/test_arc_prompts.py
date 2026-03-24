import json

from arc_prompts import choices_to_json, format_baseline_mcq


def test_format_baseline_mcq():
    q = "Which is smallest?"
    ch = {"label": ["A", "B", "C", "D"], "text": ["atom", "molecule", "cell", "organ"]}
    s = format_baseline_mcq(q, ch)
    assert "Which is smallest?" in s
    assert "A) atom" in s
    assert "Answer with only the letter" in s


def test_choices_to_json_roundtrip():
    ch = {"label": ["A", "B"], "text": ["x", "y"]}
    assert json.loads(choices_to_json(ch)) == ch
