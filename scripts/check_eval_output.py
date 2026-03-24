"""Check eval output for errors, truncations, and sample responses by perturbation type."""

import json
import sys
from collections import defaultdict

output_path = sys.argv[1]

TYPES = ['p1_format', 'p1_format_soft', 'p2_complexity', 'p4_role', 'p5_fewshot']

stats = defaultdict(lambda: {'total': 0, 'empty': 0, 'length': 0})

for line in open(output_path):
    r = json.loads(line)
    cid = int(r['custom_id'])
    ptype = TYPES[cid % 5]
    msg = r['response']['body']['choices'][0]['message']
    finish = r['response']['body']['choices'][0]['finish_reason']
    content = msg.get('content') or ''

    stats[ptype]['total'] += 1
    if not content:
        stats[ptype]['empty'] += 1
    if finish == 'length':
        stats[ptype]['length'] += 1

print(f"{'Type':<20} {'Total':>6} {'Empty':>6} {'Length':>7}")
print("-" * 42)
for ptype in TYPES:
    s = stats[ptype]
    print(f"{ptype:<20} {s['total']:>6} {s['empty']:>6} {s['length']:>7}")
total_empty = sum(s['empty'] for s in stats.values())
total_length = sum(s['length'] for s in stats.values())
print("-" * 42)
print(f"{'TOTAL':<20} {500:>6} {total_empty:>6} {total_length:>7}")
