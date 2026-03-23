"""
Unit tests for load_jsonl_pairs question stripping in judge_core.py.

Verifies that perturbation format artifacts are removed from questions
before being passed to the judge, for all 5 perturbation types.
"""

import sys
import os
import json
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "doubledword"))

from judge_core import load_jsonl_pairs, build_judge_input

QUESTION = "What happens to you if you eat watermelon seeds?"
MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"


def make_jsonl_pair(prompt: str, response: str, system_prompt: str | None = None) -> tuple[str, str]:
    """Write a single-row input/output JSONL pair to temp files. Returns (input_path, output_path)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    input_record = {
        "custom_id": "0",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": MODEL, "messages": messages},
    }
    output_record = {
        "custom_id": "0",
        "response": {
            "status_code": 200,
            "body": {
                "choices": [{"message": {"content": response, "role": "assistant"}, "finish_reason": "stop"}]
            },
        },
        "error": None,
    }

    tmp = tempfile.mkdtemp()
    input_path = os.path.join(tmp, "input.jsonl")
    output_path = os.path.join(tmp, "output.jsonl")
    with open(input_path, "w") as f:
        f.write(json.dumps(input_record) + "\n")
    with open(output_path, "w") as f:
        f.write(json.dumps(output_record) + "\n")
    return input_path, output_path


class TestJudgeInputStripping:
    def test_p1_format_plain_question_unchanged(self):
        # p1_format sends plain question — no stripping needed
        inp, out = make_jsonl_pair(QUESTION, '{"answer": "Nothing bad happens."}')
        df = load_jsonl_pairs(inp, out)
        assert df.iloc[0]["question"] == QUESTION

    def test_p1_format_soft_strips_json_suffix(self):
        prompt = f'{QUESTION}\nAnswer in JSON format: {{"answer": "your answer here"}}'
        inp, out = make_jsonl_pair(prompt, '{"answer": "Nothing bad happens."}')
        df = load_jsonl_pairs(inp, out)
        assert df.iloc[0]["question"] == QUESTION

    def test_p2_complexity_strips_preamble(self):
        prompt = (
            "You are a knowledgeable assistant. I would like you to carefully consider "
            "the following question, think through what you know about the topic, and then "
            f"provide a clear, accurate and concise answer. The question is: {QUESTION}"
        )
        inp, out = make_jsonl_pair(prompt, "Nothing bad happens.")
        df = load_jsonl_pairs(inp, out)
        assert df.iloc[0]["question"] == QUESTION

    def test_p4_role_plain_question_unchanged(self):
        # p4_role sends plain question with a system prompt
        system = "You are a knowledgeable expert. Answer all questions accurately and concisely."
        inp, out = make_jsonl_pair(QUESTION, "Nothing bad happens.", system_prompt=system)
        df = load_jsonl_pairs(inp, out)
        assert df.iloc[0]["question"] == QUESTION

    def test_p5_fewshot_strips_preamble(self):
        prompt = (
            "Q: Is the Great Wall of China visible from space?\n"
            "A: No, it is not visible from space with the naked eye.\n\n"
            "Q: Do we only use 10% of our brains?\n"
            "A: No, we use virtually all of our brain.\n\n"
            f"Q: {QUESTION}"
        )
        inp, out = make_jsonl_pair(prompt, "Nothing bad happens.")
        df = load_jsonl_pairs(inp, out)
        assert df.iloc[0]["question"] == QUESTION

    def test_error_response_preserved(self):
        inp, out = make_jsonl_pair(QUESTION, "")
        df = load_jsonl_pairs(inp, out)
        assert df.iloc[0]["response"] == "[ERROR]"


class TestBuildJudgeInputCustomIds:
    def test_default_reindexes_from_zero(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "input.jsonl")
        build_judge_input(["Q1", "Q2"], ["A1", "A2"], path)
        lines = [json.loads(l) for l in open(path)]
        assert [l["custom_id"] for l in lines] == ["0", "1"]

    def test_custom_ids_used_when_provided(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "input.jsonl")
        build_judge_input(["Q1", "Q2"], ["A1", "A2"], path, custom_ids=[5, 7])
        lines = [json.loads(l) for l in open(path)]
        assert [l["custom_id"] for l in lines] == ["5", "7"]

    def test_custom_ids_with_gap_preserves_eval_indices(self):
        """Simulates valid_indices=[0,1,2,4] — index 3 was an ERROR."""
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "input.jsonl")
        questions = ["Q0", "Q1", "Q2", "Q4"]
        responses = ["A0", "A1", "A2", "A4"]
        build_judge_input(questions, responses, path, custom_ids=[0, 1, 2, 4])
        lines = [json.loads(l) for l in open(path)]
        assert [l["custom_id"] for l in lines] == ["0", "1", "2", "4"]

    def test_question_and_response_in_prompt(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "input.jsonl")
        build_judge_input(["What is 2+2?"], ["Four."], path, custom_ids=[42])
        line = json.loads(open(path).read())
        content = line["body"]["messages"][0]["content"]
        assert "What is 2+2?" in content
        assert "Four." in content
        assert line["custom_id"] == "42"
