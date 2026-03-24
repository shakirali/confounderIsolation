"""Compare eval outputs between two batches for overlapping custom_ids."""

import json
import sys

path_a = sys.argv[1]
path_b = sys.argv[2]

def load_outputs(path):
    results = {}
    for line in open(path):
        r = json.loads(line)
        cid = int(r['custom_id'])
        msg = r['response']['body']['choices'][0]['message']
        results[cid] = msg.get('content') or ''
    return results

a = load_outputs(path_a)
b = load_outputs(path_b)

common = sorted(set(a.keys()) & set(b.keys()))
print(f"Overlapping custom_ids: {len(common)}")

diffs = [(cid, a[cid], b[cid]) for cid in common if a[cid] != b[cid]]
print(f"Different outputs: {len(diffs)}/{len(common)}")

if diffs:
    print("\nFirst 3 differences:")
    for cid, va, vb in diffs[:3]:
        print(f"\n  cid={cid}")
        print(f"  A: {va[:150]!r}")
        print(f"  B: {vb[:150]!r}")
