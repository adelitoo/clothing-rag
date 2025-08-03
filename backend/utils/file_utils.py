from functools import lru_cache
from pathlib import Path
import config


@lru_cache(maxsize=1024)
def get_image_path(article_id: int) -> Path | None:
    padded_id = str(article_id).zfill(10)
    folder_name = padded_id[:3]
    filename = f"{padded_id}.jpg"

    path = config.IMAGE_BASE_DIR / folder_name / filename

    return path if path.exists else None
