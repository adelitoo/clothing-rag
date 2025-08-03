import re
from typing import List, Dict, Any
from fastapi import Request
from pathlib import Path
from ...core.config import settings


def extract_article_ids_robust(text: str) -> List[tuple]:
    """
    Enhanced regex parsing that handles multiple response formats.
    Returns a list of (article_id, score) tuples.
    """
    found_items = []

    # Pattern 1: Standard "Article ID: 123 (Relevance: 0.85)"
    pattern1 = re.findall(r"Article\s+ID\s*:?\s*(\d+).*?Relevance\s*:?\s*([\d\.]+)", text, re.IGNORECASE | re.DOTALL)
    found_items.extend(pattern1)

    # Pattern 2: "Article ID: 123 (...)" - for cases where score is not parsed
    pattern2 = re.findall(r"Article\s+ID\s*:?\s*(\d+)\s*\([^)]+\)", text, re.IGNORECASE)
    for article_id in pattern2:
        found_items.append((article_id, "1.0")) # Assign default high score

    # Pattern 3: Just a list of numbers, fallback
    pattern3 = re.findall(r"\b(\d{6,})\b", text) # Assumes article IDs are 6+ digits
    for article_id in pattern3:
        found_items.append((article_id, "0.8")) # Assign default medium score

    # Remove duplicates, keeping the first occurrence
    seen = set()
    unique_items = []
    for item_id, score in found_items:
        if item_id not in seen:
            seen.add(item_id)
            unique_items.append((item_id, score))

    return unique_items


def enrich_search_results(results: List[Dict[str, Any]], request: Request) -> List[Dict[str, Any]]:
    base_url = str(request.base_url)
    for item in results:
        if 'image_path' in item:
            item['image_url'] = f"{base_url}images/{item['image_path']}"
    return results

def get_image_path_from_id(article_id: str) -> Path | None:
    padded_id = article_id.zfill(10)
    path = settings.IMAGE_BASE_DIR / padded_id[:3] / f"{padded_id}.jpg"
    return path if path.exists() else None