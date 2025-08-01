import requests
import pandas as pd
from tqdm import tqdm
import os

QUERIES_FILE = "../../data/fashion_queries.csv"
ARTICLES_CSV_PATH = "../../data/complete_articles.csv"
IMAGE_BASE_DIR = "../../data/images"
OUTPUT_ANNOTATION_FILE = "to_annotate.csv"
API_URL_TRANSFORMED = "http://127.0.0.1:8000/search/"
API_URL_BASELINE = "http://127.0.0.1:8000/search/baseline/"
EVALUATION_K = 5

def get_image_path(article_id: int) -> str:
    padded_id = str(article_id).zfill(10)
    folder_name = padded_id[:3]
    filename = f"{padded_id}.jpg"
    abs_path = os.path.abspath(os.path.join(IMAGE_BASE_DIR, folder_name, filename))
    return abs_path

def create_hyperlink_formula(path: str) -> str:
    return f'=HYPERLINK("file:///{path}", "View Image")'

def get_search_results(query: str, top_k: int, api_url: str):
    try:
        payload = {"query": query, "top_k": top_k}
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
        results = response.json().get("results", [])
        return [int(item['article_id']) for item in results]
    except requests.exceptions.RequestException as e:
        print(f"API call failed for query '{query}' at {api_url}: {e}")
        return []

def main():
    print("🚀 Starting the process to generate the annotation file...")

    print(f"Loading queries from '{QUERIES_FILE}'...")
    with open(QUERIES_FILE, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]
    
    print(f"Loading article metadata from '{ARTICLES_CSV_PATH}'...")
    articles_df = pd.read_csv(ARTICLES_CSV_PATH, dtype={'article_id': int})

    records_to_annotate = []
    seen_pairs = set()

    print(f"\n🔎 Pooling results for {len(queries)} queries...")
    for query in tqdm(queries, desc="Processing queries"):
        results_a = get_search_results(query, EVALUATION_K, API_URL_TRANSFORMED)
        results_b = get_search_results(query, EVALUATION_K, API_URL_BASELINE)
        pooled_article_ids = set(results_a) | set(results_b)
        
        for article_id in pooled_article_ids:
            if (query, article_id) not in seen_pairs:
                records_to_annotate.append({"query": query, "article_id": article_id})
                seen_pairs.add((query, article_id))

    if not records_to_annotate:
        print("❌ No records were generated.")
        return

    print("\n✍️ Merging results and creating image links...")
    annotation_df = pd.DataFrame(records_to_annotate)
    enriched_df = pd.merge(
        annotation_df,
        articles_df[['article_id', 'prod_name', 'img_caption']],
        on='article_id',
        how='left'
    )
    
    enriched_df['image_path'] = enriched_df['article_id'].apply(get_image_path)
    enriched_df['image_link'] = enriched_df['image_path'].apply(create_hyperlink_formula)
    
    enriched_df['relevance'] = ''
    
    final_df = enriched_df[['query', 'article_id', 'prod_name', 'img_caption', 'image_link', 'relevance']]

    final_df.to_csv(OUTPUT_ANNOTATION_FILE, index=False)
    
    print(f"\n✅ Success! Annotation file created at: '{OUTPUT_ANNOTATION_FILE}'")

if __name__ == "__main__":
    main()
