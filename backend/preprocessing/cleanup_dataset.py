
import pandas as pd
from models.image_dataset import ImageDataset
from config import ARTICLES_CSV_PATH, IMAGE_BASE_DIR, COMPLETE_ARTICLES_CSV_PATH

def clean_csv():
    try:
        df = pd.read_csv(ARTICLES_CSV_PATH)
        print(f"📄 Loaded {len(df)} articles")
    except FileNotFoundError:
        print(f"❌ Error: CSV file not found at {ARTICLES_CSV_PATH}")
        return None, 0

    dataset = ImageDataset(df, IMAGE_BASE_DIR)
    valid_indices = dataset.get_valid_indices()
    filtered_df = df.iloc[valid_indices].copy()

    print(f"✅ Filtered to {len(filtered_df)} valid rows")

    filtered_df.to_csv(COMPLETE_ARTICLES_CSV_PATH, index=False)
    print(f"📁 Saved filtered CSV to {COMPLETE_ARTICLES_CSV_PATH}")

    return filtered_df, len(filtered_df)