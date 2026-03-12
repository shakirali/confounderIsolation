import pandas as pd
from evaluate_models import query_model
from tqdm import tqdm

MODEL = "qwen3:4b"
N = 5


def run_smoke_test():
    df = pd.read_csv("data/baseline/truthfulqa_raw.csv")
    sample = df.head(N)

    results = []
    for _, row in tqdm(sample.iterrows(), total=N, desc="Querying"):
        response = query_model(row["question"], MODEL)
        results.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "perturbation_type": "baseline",
            "prompt_sent": row["question"],
            "model": MODEL,
            "response": response,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("data/baseline/responses_smoke_test.csv", index=False)

    print("\n--- Smoke Test Results ---")
    for _, r in results_df.iterrows():
        print(f"\nQ{r['question_id']}: {r['question']}")
        print(f"A: {r['response'][:200]}...")
        print("-" * 60)

    print(f"\nSaved {len(results_df)} responses to data/baseline/responses_smoke_test.csv")


if __name__ == "__main__":
    run_smoke_test()
