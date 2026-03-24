# Claude Instructions

## Experiment Storage

Store experiment results under the `experiments/doubleword_batches/` folder (TruthfulQA and other default runs). Within that folder, organise files by batch ID, with each batch containing `input.jsonl` and `output.jsonl`.

**ARC-Challenge:** store batch JSONL under `experiments/doubleword_batches/arc/` (same per-batch layout as other Doubleword runs). When calling the Doubleword client from ARC scripts, pass `batch_root=ARC_BATCH_ROOT` (see `src/doubledword/doubleword_client.py`).

**Verify batch files:** Whenever you create `input.jsonl` or obtain `output.jsonl` for a batch, confirm they are correct before relying on them or moving on. At minimum: line count matches the intended number of requests; each line is valid JSON; `custom_id` values align with the row order and metadata you expect; for outputs, each completed line has a sensible `response` (no silent misalignment between input prompts and output content). Fix or rerun if verification fails.

## Progress Tracking

Keep `agents/PLAN.md` up to date as work progresses. After completing any task, phase, or batch job, update the relevant section with:
- Status markers (✅ done / ⏳ in progress or rerun needed / ❌ todo)
- Batch IDs and result file paths
- Any decisions or reasons for reruns
