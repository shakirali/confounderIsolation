"""Nemotron must not get Qwen-style /no_think in the user message (doubleword_client)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "doubledword"))

from doubleword_client import ARC_EVAL_MODEL, model_uses_no_think_user_prefix


def test_nemotron_skips_no_think_prefix():
    assert not model_uses_no_think_user_prefix(ARC_EVAL_MODEL)
    assert not model_uses_no_think_user_prefix("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4")


def test_qwen_uses_no_think_prefix():
    assert model_uses_no_think_user_prefix("Qwen/Qwen3.5-35B-A3B-FP8")
