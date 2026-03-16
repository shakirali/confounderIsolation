# Claude Instructions

## Experiment Storage

Store experiment results under the `experiments/doubleword_batches/` folder. Within that folder, organise files by batch ID, with each batch containing `input.jsonl` and `output.jsonl`.

## Progress Tracking

Keep `agents/PLAN.md` up to date as work progresses. After completing any task, phase, or batch job, update the relevant section with:
- Status markers (✅ done / ⏳ in progress or rerun needed / ❌ todo)
- Batch IDs and result file paths
- Any decisions or reasons for reruns
