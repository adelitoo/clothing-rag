import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from config import QUERIES_FILE_PATH, API_URL, TOP_K, AVG_COSINE_SIMIL_RESULT

API_URL_BASELINE = "http://127.0.0.1:8000/search/baseline/"

def load_queries(file_path: str) -> list[str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        print(f"✅ Successfully loaded {len(queries)} queries from {file_path}")
        return queries
    except FileNotFoundError:
        print(f"❌ Error: The file {file_path} was not found.")
        return []

def get_average_similarity(query: str, top_k: int, api_url: str) -> float | None:
    payload = {"query": query, "top_k": top_k}
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status() 
        
        results = response.json().get("results", [])
        if not results:
            return 0.0 
            
        scores = [item['score'] for item in results]
        return np.mean(scores)

    except requests.exceptions.RequestException as e:
        print(f"\n⚠️ API request failed for query '{query}' at {api_url}: {e}")
        return None

def plot_comparison_distribution(df: pd.DataFrame, avg_main: float, avg_baseline: float):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.kdeplot(data=df, x='score', hue='model', fill=True, common_norm=False,
                palette={'Vector Search': 'skyblue', 'Baseline': 'lightcoral'}, ax=ax)

    ax.axvline(avg_main, color='blue', linestyle='--', linewidth=2, 
               label=f'Vector Search Avg: {avg_main:.4f}')
    ax.axvline(avg_baseline, color='red', linestyle='--', linewidth=2, 
               label=f'Baseline Avg: {avg_baseline:.4f}')

    ax.set_title(f'Comparison of Average Cosine Similarity Distributions (Top {TOP_K})', fontsize=16, pad=20)
    ax.set_xlabel('Average Cosine Similarity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    ax.set_xlim(0, 1)

    plot_filename = AVG_COSINE_SIMIL_RESULT.replace('.png', '_comparison.png')
    plt.savefig(plot_filename)
    print(f"\n📊 Comparison plot saved as {plot_filename}")
    plt.show()

def main():
    if not os.path.exists(QUERIES_FILE_PATH):
        print(f"Error: Query file not found at '{QUERIES_FILE_PATH}'.")
        print("Please create this file with one search query per line.")
        return

    queries = load_queries(QUERIES_FILE_PATH)
    if not queries:
        return
    
    evaluation_results = []
    
    print(f"\n🚀 Starting evaluation for {len(queries)} queries on both models...")
    
    for query in tqdm(queries, desc="Evaluating Queries"):
        main_avg_score = get_average_similarity(query, top_k=TOP_K, api_url=API_URL)
        if main_avg_score is not None:
            evaluation_results.append({'score': main_avg_score, 'model': 'Vector Search'})

        baseline_avg_score = get_average_similarity(query, top_k=TOP_K, api_url=API_URL_BASELINE)
        if baseline_avg_score is not None:
            evaluation_results.append({'score': baseline_avg_score, 'model': 'Baseline'})

    if not evaluation_results:
        print("\n❌ No scores were calculated. Check API connections and query file.")
        return
    
    results_df = pd.DataFrame(evaluation_results)
    
    avg_main = results_df[results_df['model'] == 'Vector Search']['score'].mean()
    avg_baseline = results_df[results_df['model'] == 'Baseline']['score'].mean()
    
    print("\n--- Evaluation Complete ---")
    print(f"📈 LLM Search Model | Overall Average Cosine Similarity: {avg_main:.4f}")
    print(f"📉 Baseline Model      | Overall Average Cosine Similarity: {avg_baseline:.4f}")
    
    plot_comparison_distribution(results_df, avg_main, avg_baseline)

if __name__ == "__main__":
    main()