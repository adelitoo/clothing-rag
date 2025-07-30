import csv
import numpy as np
import httpx
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from embeddings.embedding_utils import embed_text_query
from utils.model_utils import load_clip_model_and_processor

# === CONFIG ===
BACKEND_URL = "http://localhost:8000"  # Adjust if deployed elsewhere
TOP_K = 20
QUERY_CSV_PATH = "./data/fashion_queries.csv"

def compute_cosine_similarity(a, b):
    a = np.array(a).reshape(1, -1)
    b = np.array(b).reshape(1, -1)
    return cosine_similarity(a, b)[0][0]

def search_backend(query, top_k):
    """Sends a POST request to the /search/ endpoint."""
    payload = {"query": query, "top_k": top_k}
    try:
        response = httpx.post(f"{BACKEND_URL}/search/", json=payload, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Request failed for query '{query}': {e}")
        return None

def main():
    model, processor = load_clip_model_and_processor()

    with open(QUERY_CSV_PATH, newline="") as csvfile:
        reader = csv.reader(csvfile)
        queries = [row[0] for row in reader if row]

    per_query_scores = []
    failed_queries = []

    for query in tqdm(queries, desc="Evaluating queries"):
        result = search_backend(query, TOP_K)
        if not result or "results" not in result:
            failed_queries.append(query)
            continue

        query_emb = embed_text_query(model, processor, query)
        similarities = []

        for item in result["results"]:
            embedding = item.get("embedding")
            if embedding:
                sim = compute_cosine_similarity(query_emb, embedding)
                similarities.append(sim)

        if similarities:
            avg_sim = np.mean(similarities)
            per_query_scores.append(avg_sim)
        else:
            failed_queries.append(query)

    if per_query_scores:
        overall_avg = np.mean(per_query_scores)
        print(f"\n✅ Average cosine similarity across all queries: {overall_avg:.4f}")
    else:
        print("❌ No valid results found for any queries.")

    if failed_queries:
        print(f"\n⚠️ Failed queries ({len(failed_queries)}):")
        for fq in failed_queries[:5]:
            print(f"  - {fq}")
        if len(failed_queries) > 5:
            print("  ...")

if __name__ == "__main__":
    main()
