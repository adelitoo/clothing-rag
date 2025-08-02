import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from config import ARTICLES_CSV_PATH, IMAGE_BASE_DIR, COMPLETE_ARTICLES_CSV_PATH

def _validate_image_data(df: pd.DataFrame, image_base_dir: str) -> pd.DataFrame:
    print("Pre-filtering valid images...")
    valid_rows = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Checking images"):
        article_id = row["article_id"]
        padded_id = str(article_id).zfill(10)
        subfolder = padded_id[:3]
        image_path = os.path.join(image_base_dir, subfolder, f"{padded_id}.jpg")

        if not os.path.exists(image_path):
            continue

        try:
            with Image.open(image_path) as img:
                img.verify()  
            valid_rows.append(row)
        except Exception:
            print(f"⚠️ Corrupted image found and skipped: {image_path}")
            try:
                os.remove(image_path)
            except OSError as e:
                print(f"❌ Could not delete {image_path}: {e}")
            continue
            
    print(f"Found {len(valid_rows)} valid images out of {len(df)}")
    return pd.DataFrame(valid_rows)


def clean_csv():
    try:
        df = pd.read_csv(ARTICLES_CSV_PATH)
        print(f"📄 Loaded {len(df)} articles")
    except FileNotFoundError:
        print(f"❌ Error: CSV file not found at {ARTICLES_CSV_PATH}")
        return None, 0

    filtered_df = _validate_image_data(df, IMAGE_BASE_DIR)

    if not filtered_df.empty:
        filtered_df.to_csv(COMPLETE_ARTICLES_CSV_PATH, index=False)
        print(f"📁 Saved filtered CSV to {COMPLETE_ARTICLES_CSV_PATH}")

    return filtered_df, len(filtered_df)