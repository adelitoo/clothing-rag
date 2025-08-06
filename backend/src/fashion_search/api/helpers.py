import re
from typing import List, Dict, Any
from fastapi import Request
from pathlib import Path
from ..core.config import settings

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