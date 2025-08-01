import json
import requests
import pandas as pd
import numpy as np
import ollama
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime


def get_search_results_transformed(query: str, top_k: int, api_url: str = "http://127.0.0.1:8000/search/"):
    try:
        payload = {"query": query, "top_k": top_k}
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("results", [])
    except requests.exceptions.RequestException as e:
        print(f"API call (System A) failed for query '{query}': {e}")
        return []

def get_search_results_baseline(query: str, top_k: int, api_url: str = "http://127.0.0.1:8000/search/baseline/"):
    try:
        payload = {"query": query, "top_k": top_k}
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("results", [])
    except requests.exceptions.RequestException as e:
        print(f"API call (System B) failed for query '{query}': {e}")
        return []


def calculate_mean_distance(results):
    if not results: return None
    return np.mean([item['score'] for item in results])

def calculate_diversity(results, articles_df, category_field='garment_group_name'):
    if not results: return 0
    article_ids = [int(item['article_id']) for item in results]
    retrieved_articles = articles_df[articles_df['article_id'].isin(article_ids)]
    if retrieved_articles.empty: return 0
    return retrieved_articles[category_field].nunique()

def llm_as_judge(query, results_a, results_b, articles_df):
    def format_results(results):
        if not results: return "No results found.\n"
        article_ids = [int(item['article_id']) for item in results]
        details = articles_df[articles_df['article_id'].isin(article_ids)]
        details_map = {row['article_id']: row['prod_name'] for _, row in details.iterrows()}
        formatted_str = ""
        for i, item in enumerate(results):
            prod_name = details_map.get(int(item['article_id']), "Unknown Product")
            similarity = 1 - item['score'] 
            formatted_str += f"{i+1}. {prod_name} (Similarity: {similarity:.3f})\n"
        return formatted_str

    prompt = (
        "You are an impartial and expert judge of a fashion search engine's quality. "
        "Your task is to compare two sets of search results for a given query and decide which one is better. "
        "Evaluate based on relevance to the query, quality, and diversity of the results.\n\n"
        f"**Query:** \"{query}\"\n\n"
        "--- **Result Set A (Advanced Search)** ---\n"
        f"{format_results(results_a)}\n"
        "--- **Result Set B (Basic Search)** ---\n"
        f"{format_results(results_b)}\n\n"
        "**Instructions:**\n"
        "1. Analyze both result sets carefully.\n"
        "2. Determine which set is a better response to the query.\n"
        "3. Respond in a strict JSON format. The 'preference' value must be one of: 'A', 'B', or 'Tie'.\n\n"
        "Example Response:\n"
        "{\"preference\": \"A\", \"reasoning\": \"Set A provided a better variety of relevant styles, while Set B was too repetitive.\"}"
    )
    
    try:
        response = ollama.chat(model="llama3.1:8b", messages=[{"role": "user", "content": prompt}], options={"temperature": 0.0}, format="json")
        return json.loads(response['message']['content'])
    except Exception as e:
        print(f"LLM Judge failed for query '{query}': {e}")
        return {"preference": "Error", "reasoning": str(e)}


def save_results(eval_df, fig):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"evaluation_reports/"
    
    os.makedirs(folder_name, exist_ok=True)
    
    csv_path = os.path.join(folder_name, "detailed_report.csv")
    plot_path = os.path.join(folder_name, "summary_plot.png")
    
    eval_df.to_csv(csv_path, index=False)
    fig.savefig(plot_path, bbox_inches='tight')
    
    print("\n--- Results Saved ---")
    print(f"📊 Detailed report saved to: {csv_path}")
    print(f"📈 Summary plot saved to: {plot_path}")



if __name__ == "__main__":
    QUERIES_FILE = "../../data/fashion_queries.csv"
    ARTICLES_CSV_PATH = "../../data/complete_articles.csv"
    EVALUATION_K = 10
    
    print("Loading data...")
    with open(QUERIES_FILE, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]
    articles_df = pd.read_csv(ARTICLES_CSV_PATH)
    
    results_summary = []
    queries_to_test = queries 
    print(f"🚀 Starting A/B evaluation on {len(queries_to_test)} queries...")

    for query in tqdm(queries_to_test, desc="A/B Evaluating Queries"):
        results_a = get_search_results_transformed(query, top_k=EVALUATION_K)
        results_b = get_search_results_baseline(query, top_k=EVALUATION_K)
        
        judgement = llm_as_judge(query, results_a, results_b, articles_df)
        
        results_summary.append({
            "query": query,
            "distance_a": calculate_mean_distance(results_a), "distance_b": calculate_mean_distance(results_b),
            "diversity_a": calculate_diversity(results_a, articles_df), "diversity_b": calculate_diversity(results_b, articles_df),
            "llm_preference": judgement['preference'], "llm_reasoning": judgement['reasoning']
        })

    eval_df = pd.DataFrame(results_summary)
    
    print("\n\n--- UNATTENDED A/B EVALUATION REPORT ---")
    print("\nSystem A = LLM Transformed Query | System B = Baseline Raw Query")
    
    print("\n--- LLM Judge Preferences ---")
    preferences = eval_df['llm_preference'].value_counts(normalize=True).reindex(['A', 'B', 'Tie']).fillna(0) * 100
    print(preferences.to_string(float_format="%.1f%%"))
    
    print("\n--- Mean Average Distance@K (Lower is Better) ---")
    print(f"System A: {eval_df['distance_a'].mean():.4f}")
    print(f"System B: {eval_df['distance_b'].mean():.4f}")
        
    print("\n--- Mean Unique Categories@K (Context Dependent) ---")
    print(f"System A: {eval_df['diversity_a'].mean():.2f}")
    print(f"System B: {eval_df['diversity_b'].mean():.2f}")
        
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.set_theme(style="whitegrid")
    fig.suptitle("A/B Test Evaluation: (A) LLM-Transform vs (B) Baseline")

    preferences.plot(kind='pie', ax=axes[0], autopct='%1.1f%%', labels=['System A Wins', 'System B Wins', 'Tie'],
                     colors=['#4CAF50', '#F44336', '#FFC107'], wedgeprops=dict(width=0.4))
    axes[0].set_title("LLM Preference Distribution")
    axes[0].set_ylabel('')

    sns.kdeplot(eval_df['distance_a'].dropna(), ax=axes[1], label='System A', fill=True, bw_adjust=1.5)
    sns.kdeplot(eval_df['distance_b'].dropna(), ax=axes[1], label='System B', fill=True, bw_adjust=1.5)
    axes[1].set_title("Distribution of Average Result Distances")
    axes[1].set_xlabel("Average Cosine Distance (Lower is Better)")
    axes[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_results(eval_df, fig)
    
    plt.show()