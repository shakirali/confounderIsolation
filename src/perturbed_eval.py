import pandas as pd
from tqdm import tqdm
from evaluate_models import query_model
from generate_perturbations import p1_format, p2_complexity, p4_role, p5_fewshot
from scoring import score_response

MODEL = "llama3.1:8b"
N = 50

PERTURBATIONS = {
    "p1_format":    lambda q: (p1_format(q), None),
    "p2_complexity": lambda q: (p2_complexity(q), None),
    "p4_role":      lambda q: (p4_role(q)[1], p4_role(q)[0]),
    "p5_fewshot":   lambda q: (p5_fewshot(q), None),
}


def run_perturbed_eval():
    df = pd.read_csv("data/baseline/truthfulqa_raw.csv").head(N)

    results = []
    for ptype, fn in PERTURBATIONS.items():
        print(f"\n--- {ptype} ---")
        for _, row in tqdm(df.iterrows(), total=N, desc=f"Querying"):
            prompt, system_prompt = fn(row["question"])
            response = query_model(prompt, MODEL, system_prompt=system_prompt)
            score = score_response(row["question"], response)
            results.append({
                "question_id": row["question_id"],
                "question": row["question"],
                "perturbation_type": ptype,
                "prompt_sent": prompt,
                "system_prompt": system_prompt,
                "model": MODEL,
                "response": response,
                "score": score,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("data/baseline/responses_perturbed_eval.csv", index=False)

    print("\n--- Results vs Baseline (98%) ---")
    summary = results_df.groupby("perturbation_type")["score"].mean().sort_values()
    for ptype, mean in summary.items():
        delta = mean - 0.98
        print(f"{ptype:20s}  score={mean:.2f}  delta={delta:+.2f}")


if __name__ == "__main__":
    run_perturbed_eval()
