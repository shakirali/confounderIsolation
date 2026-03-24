"""Randomly sample N inputs from an eval batch and display key fields for manual verification."""

import json
import random
import sys

input_path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

TYPES = ['p1_format', 'p1_format_soft', 'p2_complexity', 'p4_role', 'p5_fewshot']

rows = []
for line in open(input_path):
    rows.append(json.loads(line))

random.seed(seed)
sample = sorted(random.sample(rows, n), key=lambda r: int(r['custom_id']))

for r in sample:
    cid = int(r['custom_id'])
    ptype = TYPES[cid % 5]
    msgs = r['body']['messages']
    system = next((m['content'] for m in msgs if m['role'] == 'system'), None)
    user = next(m['content'] for m in msgs if m['role'] == 'user')
    rf = r['body'].get('response_format')

    print(f"cid={cid} | type={ptype} | response_format={rf}")
    if system:
        print(f"  [system] {system}")
    print(f"  [user]   {user[:250]}")
    print()
