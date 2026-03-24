"""Summarise judge scores by perturbation type for a given eval+judge batch pair."""

import json
import sys
import pandas as pd

eval_input = sys.argv[1]
judge_output = sys.argv[2]

TYPES = ['p1_format', 'p1_format_soft', 'p2_complexity', 'p4_role', 'p5_fewshot']

input_rows = {}
for line in open(eval_input):
    r = json.loads(line)
    cid = int(r['custom_id'])
    input_rows[cid] = TYPES[cid % 5]

scores = {}
for line in open(judge_output):
    r = json.loads(line)
    cid = int(r['custom_id'])
    try:
        score = json.loads(r['response']['body']['choices'][0]['message']['content'])['score']
    except Exception:
        score = -1
    scores[cid] = score

rows = [{'custom_id': cid, 'perturbation_type': ptype, 'score': scores.get(cid, -1)} for cid, ptype in input_rows.items()]
df = pd.DataFrame(rows)

print('=== Score by perturbation type ===')
summary = df[df['score'] != -1].groupby('perturbation_type')['score'].agg(['mean', 'sum', 'count'])
summary.columns = ['mean', 'truthful', 'valid']
print(summary.to_string())
print(f'\nOverall mean (excl errors): {df[df["score"] != -1]["score"].mean():.3f}')
print(f'Errors (-1): {(df["score"] == -1).sum()}')
