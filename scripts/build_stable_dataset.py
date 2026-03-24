"""
Build stable dataset from 3 eval runs.

A (question, perturbation_type) pair is included only if all 3 runs produced
non-empty content. Outputs a CSV with question_id, perturbation_type, and the
response from run 1 (used for judging).

Usage:
    python scripts/build_stable_dataset.py \
        --run1 experiments/doubleword_batches/eda159fa.../output.jsonl \
        --run2 experiments/doubleword_batches/c3531d4c.../output.jsonl \
        --run3 experiments/doubleword_batches/35f2c19b.../output.jsonl \
        --input experiments/doubleword_batches/eda159fa.../input.jsonl \
        --output data/stable_eval.csv
"""

import argparse
import json
import pandas as pd

TYPES = ['p1_format', 'p1_format_soft', 'p2_complexity', 'p4_role', 'p5_fewshot']


def load_content(path):
    results = {}
    for line in open(path):
        r = json.loads(line)
        cid = int(r['custom_id'])
        content = r['response']['body']['choices'][0]['message'].get('content') or ''
        results[cid] = content
    return results


def load_questions(input_path):
    questions = {}
    for line in open(input_path):
        r = json.loads(line)
        cid = int(r['custom_id'])
        msgs = r['body']['messages']
        user = next(m['content'] for m in msgs if m['role'] == 'user')
        questions[cid] = user
    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run1', required=True)
    parser.add_argument('--run2', required=True)
    parser.add_argument('--run3', required=True)
    parser.add_argument('--input', required=True, help='input.jsonl from run 1')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    r1 = load_content(args.run1)
    r2 = load_content(args.run2)
    r3 = load_content(args.run3)
    questions = load_questions(args.input)

    all_cids = sorted(r1.keys())
    stable, unstable = [], []

    for cid in all_cids:
        ptype = TYPES[cid % 5]
        c1 = r1.get(cid, '')
        c2 = r2.get(cid, '')
        c3 = r3.get(cid, '')
        if c1 and c2 and c3:
            stable.append({
                'custom_id': cid,
                'question_id': cid // 5,
                'perturbation_type': ptype,
                'prompt_sent': questions.get(cid, ''),
                'response': c1,
            })
        else:
            unstable.append((cid, ptype))

    df = pd.DataFrame(stable)
    df.to_csv(args.output, index=False)

    print(f"Total pairs:    {len(all_cids)}")
    print(f"Stable (all 3): {len(stable)} ({100*len(stable)/len(all_cids):.1f}%)")
    print(f"Unstable:       {len(unstable)} ({100*len(unstable)/len(all_cids):.1f}%)")
    print()

    if unstable:
        from collections import Counter
        counts = Counter(ptype for _, ptype in unstable)
        print("Unstable breakdown by perturbation type:")
        for ptype in TYPES:
            print(f"  {ptype:<20} {counts.get(ptype, 0)}")

    stable_questions = df['question_id'].nunique() if len(df) else 0
    total_questions = len(all_cids) // 5
    print(f"\nQuestions with all 5 perturbations stable: checking...")
    if len(df):
        fully_stable = df.groupby('question_id')['perturbation_type'].count()
        all_five = (fully_stable == 5).sum()
        print(f"  {all_five}/{total_questions} questions fully stable across all 5 perturbation types")

    print(f"\nSaved stable dataset → {args.output}")


if __name__ == '__main__':
    main()
