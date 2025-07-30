import json
from typing import List
import numpy as np


def precision_at_k(predicted: List[str], relevant: List[str], k: int) -> float:
    predicted_k = predicted[:k]
    relevant_set = set(relevant)
    hits = sum(1 for item in predicted_k if item in relevant_set)
    return hits / k


def recall_at_k(predicted: List[str], relevant: List[str], k: int) -> float:
    relevant_set = set(relevant)
    predicted_k = predicted[:k]
    hits = sum(1 for item in predicted_k if item in relevant_set)
    return hits / len(relevant_set) if relevant else 0.0


def average_precision(predicted: List[str], relevant: List[str], k: int) -> float:
    relevant_set = set(relevant)
    hits, sum_precisions = 0, 0.0
    for i in range(min(k, len(predicted))):
        if predicted[i] in relevant_set:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(relevant_set) if relevant else 0.0


def evaluate(file_path: str, k: int = 10):
    with open(file_path, "r") as f:
        lines = [json.loads(line) for line in f]

    p_scores, r_scores, ap_scores = [], [], []
    for entry in lines:
        pred = [item["article_id"] for item in entry["results"]]
        gt = entry["ground_truth"]

        p = precision_at_k(pred, gt, k)
        r = recall_at_k(pred, gt, k)
        ap = average_precision(pred, gt, k)

        p_scores.append(p)
        r_scores.append(r)
        ap_scores.append(ap)

    print(f"\n🔍 Evaluation for Top-{k} results:")
    print(f"• Precision@{k}: {np.mean(p_scores):.4f}")
    print(f"• Recall@{k}:    {np.mean(r_scores):.4f}")
    print(f"• mAP@{k}:       {np.mean(ap_scores):.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to results JSONL file")
    parser.add_argument("--k", type=int, default=10, help="Top-K to evaluate")
    args = parser.parse_args()

    evaluate(args.file, args.k)
