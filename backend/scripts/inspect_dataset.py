import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.fashion_search.core.config import settings

def inspect_column_values(df: pd.DataFrame, column_name: str):
    print(f"\n✅ Found {df[column_name].nunique()} unique values for '{column_name}':")
    
    unique_values = sorted(df[column_name].dropna().unique())
    
    for value in unique_values:
        print(f"  - '{value}'")

def main():
    print("🚀 Inspecting all important dataset columns...")
    
    try:
        df = pd.read_csv(settings.COMPLETE_ARTICLES_CSV_PATH, dtype={"article_id": str})
        print(f"✅ Successfully loaded dataset with {len(df)} rows.")
    except FileNotFoundError:
        print(f"❌ ERROR: Could not find the dataset at '{settings.COMPLETE_ARTICLES_CSV_PATH}'")
        return
    except Exception as e:
        print(f"❌ ERROR: Failed to load the dataset: {e}")
        return

    columns_to_inspect = [
        "index_name",
        "product_type_name",
        "colour_group_name",
        "graphical_appearance_name",
    ]
    
    for column in columns_to_inspect:
        if column in df.columns:
            inspect_column_values(df, column)
            print("-" * 50) 
        else:
            print(f"\n⚠️ WARNING: Column '{column}' not found in the dataset.")

if __name__ == "__main__":
    main()