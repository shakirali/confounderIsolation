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
# nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4  36  $0.15/M       256k     Mar 2026  ← ARC_EVAL_MODEL
DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"

# ARC-Challenge experiment: canonical eval model (see agents/PLAN.md § ARC-Challenge)
ARC_EVAL_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"

# TruthfulQA × Nemotron experiment: same model as ARC eval
NEMOTRON_TQA_EVAL_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"

# MATH-500 × Nemotron experiment: same model
MATH500_EVAL_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"

# Local folders for batch JSONL (input.jsonl + output.jsonl per batch_id_label/)
DEFAULT_BATCH_ROOT = os.path.join("experiments", "doubleword_batches")
ARC_BATCH_ROOT = os.path.join("experiments", "doubleword_batches", "arc")
NEMOTRON_TQA_BATCH_ROOT = os.path.join("experiments", "doubleword_batches", "nemotron_tqa")
MATH500_BATCH_ROOT = os.path.join("experiments", "doubleword_batches", "math500")


def model_uses_no_think_user_prefix(model: str) -> bool:
    """Whether to prepend the Qwen-style ``/no_think`` line to the user message.

    Nemotron models should receive the task prompt only (no ``/no_think``).
    """
    return "nemotron" not in model.lower()


def get_client() -> OpenAI:
    api_key = os.getenv("DOUBLEWORD_API_KEY")
    if not api_key:
        raise ValueError("DOUBLEWORD_API_KEY not set in environment")
    return OpenAI(base_url=DOUBLEWORD_BASE_URL, api_key=api_key)


def batch_dir(batch_id: str, label: str, batch_root: str | None = None) -> str:
    """Return the local directory path for a batch, matching existing dirs by batch_id prefix."""
    base = batch_root or DEFAULT_BATCH_ROOT
    os.makedirs(base, exist_ok=True)
    if os.path.isdir(base):
        for entry in os.listdir(base):
            if entry.startswith(batch_id):
                return os.path.join(base, entry)
    return os.path.join(base, f"{batch_id}_{label}")


def download_results(
    batch_id: str,
    num_requests: int,
    label: str = "batch",
    content_only: bool = False,
    batch_root: str | None = None,
) -> list[str]:
    """Download results from a completed batch by ID, reordered by custom_id.

    Args:
        content_only: If True, only return message.content — do not fall back to reasoning_content.
                      Use this for judge scoring to avoid parsing thinking traces as scores.
    """
    client = get_client()
    status = client.batches.retrieve(batch_id)
    if status.status not in ("completed", "cancelled"):
        raise RuntimeError(f"Batch {batch_id} is not completed (status: {status.status})")
    if status.status == "cancelled":
        print(f"Warning: batch {batch_id} was cancelled — downloading partial results ({status.request_counts.completed}/{status.request_counts.total} completed).")

    print(f"Downloading results from batch {batch_id}...")
    response = requests.get(
        f"{DOUBLEWORD_BASE_URL}/files/{status.output_file_id}/content",
        headers={"Authorization": f"Bearer {os.getenv('DOUBLEWORD_API_KEY')}"},
    )
    response.raise_for_status()

    bdir = batch_dir(batch_id, label, batch_root=batch_root)
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


def _poll_and_download(
    batch_id: str,
    num_requests: int,
    label: str,
    content_only: bool,
    batch_root: str | None = None,
) -> tuple[list[str], str]:
    """Poll until batch completes then download results. Returns (responses, batch_id)."""
    client = get_client()
    poll_interval = 30
    with tqdm(desc="Waiting for batch", unit="poll") as pbar:
        while True:
            status = client.batches.retrieve(batch_id)
            counts = status.request_counts
            pbar.set_postfix({
                "status": status.status,
                "done": counts.completed,
                "total": counts.total,
            })
            pbar.update(1)

            if status.status == "completed":
                break
            elif status.status == "cancelled" and counts.completed > 0:
                print(f"Batch {batch_id} was cancelled with {counts.completed}/{counts.total} completed — downloading available results.")
                break
            elif status.status in ("failed", "expired", "cancelled"):
                raise RuntimeError(f"Batch {batch_id} ended with status: {status.status}")

            time.sleep(poll_interval)

    results = download_results(
        batch_id, num_requests, label=label, content_only=content_only, batch_root=batch_root
    )
    return results, batch_id


def submit_batch(
    prompts: list[str],
    model: str = DEFAULT_MODEL,
    system_prompts: list[str] | None = None,
    response_formats: list[dict | None] | None = None,
    completion_window: str = DEFAULT_COMPLETION_WINDOW,
    max_tokens: int = 4096,
    no_think_prefix: bool = False,
    content_only: bool = False,
    label: str = "batch",
    batch_root: str | None = None,
) -> tuple[list[str], str]:
    """
    Submit prompts as a Doubleword batch job and return responses in order.

    Args:
        batch_root: Directory containing `<batch_id>_<label>/` folders. Defaults to
            `experiments/doubleword_batches`. Use `ARC_BATCH_ROOT` (`experiments/doubleword_batches/arc`) for ARC-Challenge runs.
        no_think_prefix: If True, prepend ``/no_think`` for models that use that Qwen-style convention
            (ignored for Nemotron — see ``model_uses_no_think_user_prefix``). Default False: raw user text only.

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
        use_no_think = no_think_prefix and model_uses_no_think_user_prefix(model)
        user_content = f"/no_think\n{prompt}" if use_no_think else prompt
        messages.append({"role": "user", "content": user_content})

        body: dict = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if response_formats and response_formats[i]:
            body["response_format"] = response_formats[i]
        lines.append(json.dumps({
            "custom_id": str(i),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
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

    bdir = batch_dir(batch.id, label, batch_root=batch_root)
    os.makedirs(bdir, exist_ok=True)
    input_path = os.path.join(bdir, "input.jsonl")
    with open(input_path, "wb") as f:
        f.write(jsonl_bytes)
    print(f"Saved input → {input_path}")

    return _poll_and_download(
        batch.id, len(prompts), label=label, content_only=content_only, batch_root=batch_root
    )


def submit_batch_from_file(
    input_jsonl_path: str,
    num_requests: int,
    completion_window: str = DEFAULT_COMPLETION_WINDOW,
    content_only: bool = False,
    label: str = "batch",
    batch_root: str | None = None,
) -> tuple[list[str], str]:
    """
    Submit a pre-built input.jsonl as a Doubleword batch job.

    Renames the pending batch folder to <batch_id>_<label>/ after submission.

    Args:
        input_jsonl_path: Path to the pre-built input.jsonl (e.g. pending_<label>/input.jsonl).
        num_requests: Number of requests in the file.
        completion_window: "24h" or "1h".
        content_only: If True, only return message.content from responses.
        label: Batch directory label.
        batch_root: Where to place `<batch_id>_<label>/` after submit (default: `DEFAULT_BATCH_ROOT`).

    Returns:
        Tuple of (response strings, batch_id).
    """
    client = get_client()
    root = batch_root or DEFAULT_BATCH_ROOT
    os.makedirs(root, exist_ok=True)

    with open(input_jsonl_path, "rb") as f:
        jsonl_bytes = f.read()

    actual_requests = sum(1 for line in jsonl_bytes.decode().splitlines() if line.strip())
    print(f"Uploading batch file ({actual_requests} requests)...")
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

    # Rename pending_<label>/ → <batch_id>_<label>/
    pending_dir = os.path.dirname(os.path.abspath(input_jsonl_path))
    final_dir = os.path.join(root, f"{batch.id}_{label}")
    os.rename(pending_dir, final_dir)
    print(f"Renamed batch folder → {final_dir}")

    return _poll_and_download(
        batch.id, num_requests, label=label, content_only=content_only, batch_root=batch_root
    )
