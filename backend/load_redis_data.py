import pandas as pd
import redis
import json
from tqdm import tqdm
import config


def load_data_to_redis():
    """
    Reads the complete articles CSV and loads each article's data into
    a unique Redis key as a JSON object.
    """
    print("🚀 Starting data load into Redis...")

    # --- Step 1: Connect to Redis ---
    try:
        # Using redis-py's JSON client capabilities
        r = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT)
        r.ping()
        print(
            f"✅ Successfully connected to Redis at {config.REDIS_HOST}:{config.REDIS_PORT}"
        )
    except redis.exceptions.ConnectionError as e:
        print(f"❌ Could not connect to Redis: {e}")
        return

    # --- Step 2: Read and prepare the CSV data ---
    try:
        df = pd.read_csv(config.COMPLETE_ARTICLES_CSV_PATH, dtype={"article_id": str})
        # Replace NaN values with empty strings to prevent JSON errors
        df.fillna("", inplace=True)
        print(f"✅ Successfully loaded and prepared {len(df)} articles from CSV.")
    except FileNotFoundError:
        print(
            f"❌ Error: The file was not found at {config.COMPLETE_ARTICLES_CSV_PATH}"
        )
        return

    # --- Step 3: Load data into Redis using a pipeline for high performance ---
    print("⏳ Loading article data into Redis (this may take a moment)...")
    pipe = r.pipeline()
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Uploading to Redis"):
        # The key for each article, e.g., "article:0108775015"
        key = f"article:{row['article_id']}"
        # Convert the row to a dictionary, then to a JSON string
        value = row.to_dict()
        # Use the JSON.SET command via the pipeline
        pipe.json().set(key, "$", value)

    # Execute all commands in the pipeline at once
    print("⏳ Executing Redis pipeline...")
    pipe.execute()
    print(f"✅ Successfully loaded {len(df)} articles into Redis.")


if __name__ == "__main__":
    load_data_to_redis()
