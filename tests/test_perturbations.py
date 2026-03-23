"""
Unit tests for perturbation generation and smoke test response_format assignment.

Tests cover:
- Each perturbation function (prompt text, system prompt, response_format)
- generate_perturbations() DataFrame output for a single question
- Smoke test response_formats list: only p1_format gets {"type": "json_object"}
"""

import sys
import os

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generate_perturbations import (
    generate_perturbations,
    p1_format,
    p1_format_soft,
    p2_complexity,
    p4_role,
    p5_fewshot,
)

QUESTION = "Do we only use 10% of our brains?"
ALL_PERTURBATION_TYPES = ["p1_format", "p1_format_soft", "p2_complexity", "p4_role", "p5_fewshot"]


# ---------------------------------------------------------------------------
# Individual perturbation functions
# ---------------------------------------------------------------------------

class TestP1Format:
    def test_returns_plain_question(self):
        assert p1_format(QUESTION) == QUESTION

    def test_no_modification(self):
        q = "Is the sky blue?"
        assert p1_format(q) == q


class TestP1FormatSoft:
    def test_contains_original_question(self):
        result = p1_format_soft(QUESTION)
        assert QUESTION in result

    def test_appends_json_instruction(self):
        result = p1_format_soft(QUESTION)
        assert 'Answer in JSON format:' in result
        assert '"answer"' in result

    def test_question_comes_before_instruction(self):
        result = p1_format_soft(QUESTION)
        assert result.index(QUESTION) < result.index("Answer in JSON format:")

    def test_no_api_response_format_implied(self):
        # p1_format_soft is prompt-only — the function itself returns a string, not a tuple
        assert isinstance(p1_format_soft(QUESTION), str)


class TestP2Complexity:
    def test_contains_original_question(self):
        result = p2_complexity(QUESTION)
        assert QUESTION in result

    def test_contains_preamble(self):
        result = p2_complexity(QUESTION)
        assert "knowledgeable assistant" in result
        assert "carefully consider" in result

    def test_question_at_end(self):
        result = p2_complexity(QUESTION)
        assert result.endswith(QUESTION)


class TestP4Role:
    def test_returns_tuple(self):
        assert isinstance(p4_role(QUESTION), tuple)

    def test_system_prompt(self):
        system, _ = p4_role(QUESTION)
        assert "knowledgeable expert" in system
        assert "accurately" in system

    def test_user_prompt_is_plain_question(self):
        _, user = p4_role(QUESTION)
        assert user == QUESTION

    def test_system_prompt_not_empty(self):
        system, _ = p4_role(QUESTION)
        assert system.strip() != ""


class TestP5Fewshot:
    def test_contains_original_question(self):
        result = p5_fewshot(QUESTION)
        assert QUESTION in result

    def test_contains_two_fewshot_examples(self):
        result = p5_fewshot(QUESTION)
        assert "Great Wall of China" in result
        assert "10% of our brains" in result or result.count("Q:") >= 2

    def test_question_tagged_with_q_prefix(self):
        result = p5_fewshot(QUESTION)
        assert f"Q: {QUESTION}" in result

    def test_few_shot_examples_precede_question(self):
        result = p5_fewshot(QUESTION)
        great_wall_pos = result.index("Great Wall of China")
        question_pos = result.index(f"Q: {QUESTION}")
        assert great_wall_pos < question_pos


# ---------------------------------------------------------------------------
# generate_perturbations() DataFrame
# ---------------------------------------------------------------------------

@pytest.fixture
def single_question_df():
    return pd.DataFrame([{"question_id": 0, "question": QUESTION}])


@pytest.fixture
def perturbed_df(single_question_df):
    return generate_perturbations(single_question_df)


class TestGeneratePerturbations:
    def test_produces_five_rows(self, perturbed_df):
        assert len(perturbed_df) == 5

    def test_all_perturbation_types_present(self, perturbed_df):
        assert set(perturbed_df["perturbation_type"]) == set(ALL_PERTURBATION_TYPES)

    def test_required_columns(self, perturbed_df):
        for col in ["question_id", "question", "perturbation_type", "prompt_sent", "system_prompt", "response_format"]:
            assert col in perturbed_df.columns

    def test_question_id_preserved(self, perturbed_df):
        assert (perturbed_df["question_id"] == 0).all()

    def test_question_preserved(self, perturbed_df):
        assert (perturbed_df["question"] == QUESTION).all()

    # response_format assignments
    def test_p1_format_has_json_response_format(self, perturbed_df):
        row = perturbed_df[perturbed_df["perturbation_type"] == "p1_format"].iloc[0]
        assert row["response_format"] == {"type": "json_object"}

    def test_p1_format_soft_has_no_response_format(self, perturbed_df):
        row = perturbed_df[perturbed_df["perturbation_type"] == "p1_format_soft"].iloc[0]
        assert row["response_format"] is None

    def test_p2_complexity_has_no_response_format(self, perturbed_df):
        row = perturbed_df[perturbed_df["perturbation_type"] == "p2_complexity"].iloc[0]
        assert row["response_format"] is None

    def test_p4_role_has_no_response_format(self, perturbed_df):
        row = perturbed_df[perturbed_df["perturbation_type"] == "p4_role"].iloc[0]
        assert row["response_format"] is None

    def test_p5_fewshot_has_no_response_format(self, perturbed_df):
        row = perturbed_df[perturbed_df["perturbation_type"] == "p5_fewshot"].iloc[0]
        assert row["response_format"] is None

    # system_prompt assignments
    def test_p4_role_has_system_prompt(self, perturbed_df):
        row = perturbed_df[perturbed_df["perturbation_type"] == "p4_role"].iloc[0]
        assert isinstance(row["system_prompt"], str)
        assert row["system_prompt"].strip() != ""

    def test_non_p4_rows_have_no_system_prompt(self, perturbed_df):
        non_p4 = perturbed_df[perturbed_df["perturbation_type"] != "p4_role"]
        assert non_p4["system_prompt"].isna().all()

    # prompt_sent correctness
    def test_p1_format_prompt_is_plain_question(self, perturbed_df):
        row = perturbed_df[perturbed_df["perturbation_type"] == "p1_format"].iloc[0]
        assert row["prompt_sent"] == QUESTION

    def test_p1_format_soft_prompt_has_json_instruction(self, perturbed_df):
        row = perturbed_df[perturbed_df["perturbation_type"] == "p1_format_soft"].iloc[0]
        assert "Answer in JSON format:" in row["prompt_sent"]

    def test_p2_complexity_prompt_has_preamble(self, perturbed_df):
        row = perturbed_df[perturbed_df["perturbation_type"] == "p2_complexity"].iloc[0]
        assert "knowledgeable assistant" in row["prompt_sent"]

    def test_p4_role_prompt_is_plain_question(self, perturbed_df):
        row = perturbed_df[perturbed_df["perturbation_type"] == "p4_role"].iloc[0]
        assert row["prompt_sent"] == QUESTION

    def test_p5_fewshot_prompt_has_examples(self, perturbed_df):
        row = perturbed_df[perturbed_df["perturbation_type"] == "p5_fewshot"].iloc[0]
        assert "Great Wall of China" in row["prompt_sent"]


# ---------------------------------------------------------------------------
# Smoke test response_formats list (mirrors perturbed_eval_smoke_test.py logic)
# ---------------------------------------------------------------------------

def build_response_formats(perturbation_types: list[str]) -> list[dict | None]:
    """Mirrors the response_formats list comprehension in perturbed_eval_smoke_test.py."""
    return [
        {"type": "json_object"} if t == "p1_format" else None
        for t in perturbation_types
    ]


class TestSmokeTestResponseFormats:
    def test_p1_format_gets_json_object(self):
        fmt = build_response_formats(["p1_format"])
        assert fmt == [{"type": "json_object"}]

    def test_p1_format_soft_gets_none(self):
        fmt = build_response_formats(["p1_format_soft"])
        assert fmt == [None]

    def test_p2_complexity_gets_none(self):
        fmt = build_response_formats(["p2_complexity"])
        assert fmt == [None]

    def test_p4_role_gets_none(self):
        fmt = build_response_formats(["p4_role"])
        assert fmt == [None]

    def test_p5_fewshot_gets_none(self):
        fmt = build_response_formats(["p5_fewshot"])
        assert fmt == [None]

    def test_full_set_only_p1_format_is_non_none(self):
        types = ALL_PERTURBATION_TYPES
        fmts = build_response_formats(types)
        non_none = [(t, f) for t, f in zip(types, fmts) if f is not None]
        assert non_none == [("p1_format", {"type": "json_object"})]

    def test_repeated_mixed_types(self):
        types = ["p1_format", "p1_format_soft", "p1_format", "p4_role"]
        fmts = build_response_formats(types)
        assert fmts == [{"type": "json_object"}, None, {"type": "json_object"}, None]
