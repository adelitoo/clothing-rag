import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'image_loader')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.image_dataset import ImageDataset
from config import IMAGE_BASE_DIR, CSV_PATH, FILTERED_CSV_PATH


def clean_csv():
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"📄 Loaded {len(df)} articles")
    except FileNotFoundError:
        print(f"❌ Error: CSV file not found at {CSV_PATH}")
        return None, 0

    dataset = ImageDataset(df, IMAGE_BASE_DIR)
    valid_indices = dataset.get_valid_indices()
    filtered_df = df.iloc[valid_indices].copy()

    print(f"✅ Filtered to {len(filtered_df)} valid rows")

    filtered_df.to_csv(FILTERED_CSV_PATH, index=False)
    print(f"📁 Saved filtered CSV to {FILTERED_CSV_PATH}")

    return filtered_df, len(filtered_df)
