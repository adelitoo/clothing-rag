# scripts/load_redis_data.py

import sys
from pathlib import Path
import pandas as pd
import redis
import json
from tqdm import tqdm
from src.fashion_search.core.config import settings

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def load_data_to_redis():
    print("🚀 Starting data load into Redis...")
    try:
        r = redis.Redis(host=settings.REDIS_HOST, port=int(settings.REDIS_PORT))
        df = pd.read_csv(settings.COMPLETE_ARTICLES_CSV_PATH, dtype={"article_id": str})
        df.fillna("", inplace=True)
    except Exception as e:
        print(f"❌ Failed during setup: {e}")
        return

    print("⏳ Loading article data into Redis with standardized keys and paths...")
    pipe = r.pipeline()
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Uploading to Redis"):
        value = row.to_dict()
        article_id_str = str(value["article_id"])

        padded_id = article_id_str.zfill(10)

        key = f"article:{article_id_str}"

        value["image_path"] = f"{padded_id[:3]}/{padded_id}.jpg"

        json_value = json.dumps(value)
        pipe.set(key, json_value)

    print("⏳ Executing Redis pipeline...")
    pipe.execute()
    print(f"✅ Successfully loaded {len(df)} articles into Redis.")


if __name__ == "__main__":
    load_data_to_redis()
