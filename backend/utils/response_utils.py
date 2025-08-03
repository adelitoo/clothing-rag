# utils/response_utils.py

from typing import List, Dict
from fastapi import Request
from .file_utils import get_image_path
import config


def enrich_search_results(hits: List[Dict], request: Request) -> List[Dict]:
    """
    Enriches search results with image URLs and logs the process for debugging.
    """
    base_url = str(request.base_url).rstrip("/")
    print(f"\n🔎 [Enrichment] Starting enrichment for {len(hits)} items...")

    enriched_count = 0
    for i, item in enumerate(hits):
        article_id = item.get("article_id")
        print(f"  - Processing item {i + 1}/{len(hits)} with article_id: {article_id}")

        if not article_id:
            print("    - ⚠️ Skipping item due to missing article_id.")
            item["image_url"] = None
            continue

        # This function is the critical step. Let's see what it returns.
        path_obj = get_image_path(article_id)

        if path_obj and path_obj.exists():
            # The image path was found and exists on disk.
            relative_path = path_obj.relative_to(config.IMAGE_BASE_DIR)
            image_url = f"{base_url}/images/{relative_path.as_posix()}"
            item["image_url"] = image_url
            print(f"    - ✅ Successfully generated image_url: {image_url}")
            enriched_count += 1
        else:
            # get_image_path returned None or the path does not exist.
            print(
                f"    - ❌ Failed to find a valid image path for article_id: {article_id}. Path object was: {path_obj}"
            )
            item["image_url"] = None

    print(
        f"🔎 [Enrichment] Finished. Successfully added URLs to {enriched_count} of {len(hits)} items."
    )
    return hits
