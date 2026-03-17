"""
Shared Doubleword batch engine.

Provides the core API client, batch submission, polling, and result download
used by all Doubleword evaluation scripts.

Requires DOUBLEWORD_API_KEY in .env.
"""

import io
import json
import os
import time

import requests
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DOUBLEWORD_BASE_URL = "https://api.doubleword.ai/v1"
DEFAULT_COMPLETION_WINDOW = "1h"

# Available models (https://app.doubleword.ai/models)
# Model                                Intelligence  Cost (input/M)  Context  Released
# Qwen/Qwen3.5-9B                           32        $0.03          256k     Mar 2026
# Qwen/Qwen3.5-35B-A3B-FP8                  37        $0.05          256k     Feb 2026  ← default eval
# Qwen/Qwen3-14B-FP8                        13        $0.02           32k     Apr 2025
# Qwen/Qwen3.5-397B-A17B-FP8               45        $0.15          256k     Feb 2026  ← strong judge
# openai/gpt-oss-20b                        25        $0.02          128k     Aug 2025
# Qwen/Qwen3-VL-30B-A3B-Instruct-FP8       16        $0.05          256k     Oct 2025  (vision-language)
# Qwen/Qwen3-VL-235B-A22B-Instruct-FP8     21        $0.10          256k     Sep 2025  (vision-language)
DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"


def get_client() -> OpenAI:
    api_key = os.getenv("DOUBLEWORD_API_KEY")
    if not api_key:
        raise ValueError("DOUBLEWORD_API_KEY not set in environment")
    return OpenAI(base_url=DOUBLEWORD_BASE_URL, api_key=api_key)


def batch_dir(batch_id: str, label: str) -> str:
    """Return the local directory path for a batch, matching existing dirs by batch_id prefix."""
    base = os.path.join("experiments", "doubleword_batches")
    if os.path.isdir(base):
        for entry in os.listdir(base):
            if entry.startswith(batch_id):
                return os.path.join(base, entry)
    return os.path.join(base, f"{batch_id}_{label}")


def download_results(batch_id: str, num_requests: int, label: str = "batch", content_only: bool = False) -> list[str]:
    """Download results from a completed batch by ID, reordered by custom_id.

    Args:
        content_only: If True, only return message.content — do not fall back to reasoning_content.
                      Use this for judge scoring to avoid parsing thinking traces as scores.
    """
    client = get_client()
    status = client.batches.retrieve(batch_id)
    if status.status != "completed":
        raise RuntimeError(f"Batch {batch_id} is not completed (status: {status.status})")

    print(f"Downloading results from batch {batch_id}...")
    response = requests.get(
        f"{DOUBLEWORD_BASE_URL}/files/{status.output_file_id}/content",
        headers={"Authorization": f"Bearer {os.getenv('DOUBLEWORD_API_KEY')}"},
    )
    response.raise_for_status()

    bdir = batch_dir(batch_id, label)
    os.makedirs(bdir, exist_ok=True)
    output_path = os.path.join(bdir, "output.jsonl")
    with open(output_path, "w") as f:
        f.write(response.text)
    print(f"Saved output → {output_path}")

    results = ["[ERROR]"] * num_requests
    for line in response.text.strip().split("\n"):
        if not line:
            continue
        record = json.loads(line)
        idx = int(record["custom_id"])
        try:
            msg = record["response"]["body"]["choices"][0]["message"]
            if content_only:
                results[idx] = msg.get("content") or ""
            else:
                results[idx] = msg.get("content") or msg.get("reasoning_content") or "[ERROR]"
        except (KeyError, IndexError):
            results[idx] = "[ERROR]"

    return results


def submit_batch(
    prompts: list[str],
    model: str = DEFAULT_MODEL,
    system_prompts: list[str] | None = None,
    completion_window: str = DEFAULT_COMPLETION_WINDOW,
    max_tokens: int = 4096,
    enable_thinking: bool = True,
    content_only: bool = False,
    label: str = "batch",
) -> tuple[list[str], str]:
    """
    Submit prompts as a Doubleword batch job and return responses in order.

    Args:
        prompts: List of user prompts.
        model: Doubleword model ID.
        system_prompts: Optional per-prompt system prompts (same length as prompts).
        completion_window: "24h" (cheapest) or "1h" (faster).
        max_tokens: Max tokens per response.
        enable_thinking: Set False to disable Qwen3 thinking mode.
        label: Descriptive label appended to the batch directory (e.g. "eval", "judge").

    Returns:
        Tuple of (response strings, batch_id). Responses are same order as input;
        failed requests return "[ERROR]".
    """
    client = get_client()

    lines = []
    for i, prompt in enumerate(prompts):
        messages = []
        if system_prompts and system_prompts[i]:
            messages.append({"role": "system", "content": system_prompts[i]})
        user_content = f"/no_think\n{prompt}" if not enable_thinking else prompt
        messages.append({"role": "user", "content": user_content})

        lines.append(json.dumps({
            "custom_id": str(i),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": max_tokens,
            },
        }))

    jsonl_bytes = "\n".join(lines).encode("utf-8")

    print(f"Uploading batch file ({len(prompts)} requests)...")
    batch_file = client.files.create(
        file=("batch.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
        purpose="batch",
    )

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
    )
    print(f"Batch created: {batch.id} (window={completion_window})")

    bdir = batch_dir(batch.id, label)
    os.makedirs(bdir, exist_ok=True)
    input_path = os.path.join(bdir, "input.jsonl")
    with open(input_path, "wb") as f:
        f.write(jsonl_bytes)
    print(f"Saved input → {input_path}")

    poll_interval = 30
    with tqdm(desc="Waiting for batch", unit="poll") as pbar:
        while True:
            status = client.batches.retrieve(batch.id)
            counts = status.request_counts
            pbar.set_postfix({
                "status": status.status,
                "done": counts.completed,
                "total": counts.total,
            })
            pbar.update(1)

            if status.status == "completed":
                break
            elif status.status in ("failed", "expired", "cancelled"):
                raise RuntimeError(f"Batch {batch.id} ended with status: {status.status}")

            time.sleep(poll_interval)

    results = download_results(batch.id, len(prompts), label=label, content_only=content_only)
    return results, batch.id
