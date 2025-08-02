from typing import List, Dict
from fastapi import Request
from .file_utils import get_image_path
import config
from functools import cache


def enrich_search_results(hits: List[Dict], request: Request) -> List[Dict]:
    base_url = str(request.base_url).rstrip("/")
    for item in hits:
        path_obj = get_image_path(item["article_id"])
        if path_obj:
            relative_path = path_obj.relative_to(config.IMAGE_BASE_DIR)
            item["image_url"] = f"{base_url}/images/{relative_path.as_posix()}"
        else:
            item["image_url"] = None
    return hits
