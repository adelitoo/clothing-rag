import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

GROUND_TRUTH_FILE = "ground_truth.csv"
API_URL_TRANSFORMED = "http://127.0.0.1:8000/search/"
API_URL_BASELINE = "http://127.0.0.1:8000/search/baseline/"
EVALUATION_K = 10
RELEVANCE_THRESHOLD = 2  


def calculate_dcg_at_k(relevance_scores: list[float], k: int) -> float:
    relevance_scores = np.asarray(relevance_scores, dtype=float)[:k]
    discounts = np.log2(np.arange(len(relevance_scores)) + 2)
    return np.sum(relevance_scores / discounts)

def calculate_ndcg_at_k(relevance_scores: list[float], k: int) -> float:
    dcg = calculate_dcg_at_k(relevance_scores, k)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = calculate_dcg_at_k(ideal_scores, k)
    return dcg / idcg if idcg > 0 else 0.0

def calculate_average_precision_at_k(relevance_scores: list[float], k: int, threshold: int) -> float:
    relevance_scores = np.asarray(relevance_scores)[:k]
    binary_relevance = (relevance_scores >= threshold).astype(int)
    if np.sum(binary_relevance) == 0:
        return 0.0
    precision_at_k = np.cumsum(binary_relevance) / (np.arange(len(binary_relevance)) + 1)
    return np.sum(precision_at_k * binary_relevance) / np.sum(binary_relevance)

def calculate_reciprocal_rank_at_k(relevance_scores: list[float], k: int, threshold: int) -> float:
    relevance_scores = np.asarray(relevance_scores)[:k]
    binary_relevance = (relevance_scores >= threshold).astype(int)
    first_relevant_indices = np.where(binary_relevance == 1)[0]
    if len(first_relevant_indices) > 0:
        first_rank = first_relevant_indices[0] + 1
        return 1 / first_rank
    else:
        return 0.0


def create_radar_chart(labels, system_a_scores, system_b_scores):
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    system_a_scores += system_a_scores[:1]
    system_b_scores += system_b_scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, system_a_scores, color='#F44336', linewidth=2, linestyle='solid', label='System A (LLM)')
    ax.fill(angles, system_a_scores, color='#F44336', alpha=0.25)
    ax.plot(angles, system_b_scores, color='#4CAF50', linewidth=2, linestyle='solid', label='System B (Baseline)')
    ax.fill(angles, system_b_scores, color='#4CAF50', alpha=0.25)
    ax.set_ylim(0, 1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    plt.title('Overall System Performance Comparison', size=20, color='black', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plot_path = os.path.join("evaluation_reports", "evaluation_summary_radar.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Radar chart saved successfully to: {plot_path}")


def get_ranked_article_ids(query: str, top_k: int, api_url: str) -> list[int]:
    try:
        payload = {"query": query, "top_k": top_k}
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        results = response.json().get("results", [])
        return [int(item['article_id']) for item in results]
    except requests.exceptions.RequestException:
        return []


def main():
    print("🚀 Starting full evaluation (nDCG, MAP, MRR)...")

    try:
        gt_df = pd.read_csv(GROUND_TRUTH_FILE, sep=',')
        gt_df.dropna(subset=['relevance'], inplace=True)
        gt_df['relevance'] = gt_df['relevance'].astype(int)
    except FileNotFoundError:
        print(f"❌ Error: Ground truth file not found at '{GROUND_TRUTH_FILE}'")
        return

    relevance_map = {(row.query, row.article_id): row.relevance for row in gt_df.itertuples()}
    unique_queries = gt_df['query'].unique()
    print(f"✅ Ground truth loaded for {len(unique_queries)} unique queries.")

    results_data = []
    for query in tqdm(unique_queries, desc="Evaluating Queries"):
        ranked_list_a = get_ranked_article_ids(query, EVALUATION_K, API_URL_TRANSFORMED)
        ranked_list_b = get_ranked_article_ids(query, EVALUATION_K, API_URL_BASELINE)
        
        relevance_a = [relevance_map.get((query, aid), 0) for aid in ranked_list_a]
        relevance_b = [relevance_map.get((query, aid), 0) for aid in ranked_list_b]
        
        results_data.append({
            "query": query,
            "ndcg_A": calculate_ndcg_at_k(relevance_a, EVALUATION_K),
            "map_A": calculate_average_precision_at_k(relevance_a, EVALUATION_K, RELEVANCE_THRESHOLD),
            "mrr_A": calculate_reciprocal_rank_at_k(relevance_a, EVALUATION_K, RELEVANCE_THRESHOLD),
            "ndcg_B": calculate_ndcg_at_k(relevance_b, EVALUATION_K),
            "map_B": calculate_average_precision_at_k(relevance_b, EVALUATION_K, RELEVANCE_THRESHOLD),
            "mrr_B": calculate_reciprocal_rank_at_k(relevance_b, EVALUATION_K, RELEVANCE_THRESHOLD),
        })
        
    results_df = pd.DataFrame(results_data)

    print("\n--- 📊 Evaluation Complete ---")
    print(f"Metrics calculated @K={EVALUATION_K} with relevance threshold>={RELEVANCE_THRESHOLD}\n")
    
    mean_ndcg_a = results_df['ndcg_A'].mean()
    mean_map_a = results_df['map_A'].mean()
    mean_mrr_a = results_df['mrr_A'].mean()
    
    mean_ndcg_b = results_df['ndcg_B'].mean()
    mean_map_b = results_df['map_B'].mean()
    mean_mrr_b = results_df['mrr_B'].mean()
    
    summary = {
        "Metric": ["nDCG", "MAP", "MRR"],
        "System A (LLM)": [mean_ndcg_a, mean_map_a, mean_mrr_a],
        "System B (Baseline)": [mean_ndcg_b, mean_map_b, mean_mrr_b]
    }
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False, float_format="%.4f"))
    print("\n---------------------------------")
    
    print("🎨 Generating all plots")
    
    if not os.path.exists("evaluation_reports"):
        os.makedirs("evaluation_reports")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.bar(['System A (LLM)', 'System B (Baseline)'], [mean_ndcg_a, mean_ndcg_b], color=['#F44336', '#4CAF50'])
    ax1.set_ylabel(f'Mean nDCG@{EVALUATION_K}')
    ax1.set_title(f'Mean nDCG@{EVALUATION_K} Comparison')
    plot1_path = os.path.join("evaluation_reports", "ndcg_mean_comparison.png")
    plt.savefig(plot1_path)
    print(f"✅ Mean comparison plot saved to: {plot1_path}")

    fig2, ax2 = plt.subplots(figsize=(10, 7))
    sns.boxplot(data=results_df[['ndcg_A', 'ndcg_B']], ax=ax2, palette=['#F44336', '#4CAF50'])
    ax2.set_xticklabels(['System A (LLM)', 'System B (Baseline)'])
    ax2.set_ylabel(f'nDCG@{EVALUATION_K} Score')
    ax2.set_title(f'Distribution of nDCG@{EVALUATION_K} Scores')
    plot2_path = os.path.join("evaluation_reports", "ndcg_distribution_boxplot.png")
    plt.savefig(plot2_path)
    print(f"✅ Score distribution plot saved to: {plot2_path}")
    
    radar_labels = ["nDCG", "MAP", "MRR"]
    radar_scores_a = [mean_ndcg_a, mean_map_a, mean_mrr_a]
    radar_scores_b = [mean_ndcg_b, mean_map_b, mean_mrr_b]
    create_radar_chart(radar_labels, radar_scores_a, radar_scores_b)

    plt.show() 

if __name__ == "__main__":
    main()
