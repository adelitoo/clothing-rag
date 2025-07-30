import csv
import json
import pandas as pd
import re
from tqdm import tqdm

STOPWORDS = {
    "for",
    "the",
    "and",
    "a",
    "an",
    "in",
    "on",
    "with",
    "of",
    "to",
    "at",
    "by",
    "from",
}


def normalize(text):
    return re.sub(r"[^a-z0-9\s]", "", text.lower())


def load_catalog(catalog_path):
    df = pd.read_csv(catalog_path)
    df.fillna("", inplace=True)

    text_fields = [
        "prod_name",
        "detail_desc",
        "product_type_name",
        "product_group_name",
        "colour_group_name",
        "perceived_colour_master_name",
        "index_name",
        "garment_group_name",
    ]

    products = []
    for _, row in df.iterrows():
        searchable_text = " ".join(str(row.get(col, "")) for col in text_fields)
        products.append(
            {"article_id": int(row["article_id"]), "text": normalize(searchable_text)}
        )
    return products


def generate_ground_truth(queries_path, catalog_path, output_path):
    catalog = load_catalog(catalog_path)

    # Updated to read queries from a CSV file
    queries = []
    with open(queries_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "query" in row:
                queries.append(row["query"])

    output_lines = []

    for query in tqdm(queries, desc="Generating ground truth"):
        norm_query = normalize(query)
        keywords = set(word for word in norm_query.split() if word not in STOPWORDS)

        matching_ids = []
        for product in catalog:
            product_tokens = set(
                tok for tok in product["text"].split() if tok not in STOPWORDS
            )

            # --- FIX IS HERE ---
            # Define common_keywords before using it in the if statement.
            # This must be inside the inner loop.
            common_keywords = keywords.intersection(product_tokens)

            match_threshold = 0.75

            if keywords and (len(common_keywords) / len(keywords)) >= match_threshold:
                matching_ids.append(product["article_id"])

        output_lines.append({"query": query, "ground_truth": matching_ids})

    with open(output_path, "w") as out_file:
        for line in output_lines:
            json.dump(line, out_file)
            out_file.write("\n")

    print(f"✅ Ground truth written to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries", required=True, help="Path to .txt or .csv file with queries"
    )
    parser.add_argument(
        "--catalog", required=True, help="Path to product metadata (CSV)"
    )
    parser.add_argument("--output", default="ground_truth.jsonl", help="Output file")

    args = parser.parse_args()

    generate_ground_truth(args.queries, args.catalog, args.output)
