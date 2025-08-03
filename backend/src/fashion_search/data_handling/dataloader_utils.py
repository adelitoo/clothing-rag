import torch
from torch.utils.data import DataLoader
from ..core.config import settings


def _collate_fn(batch):
    """Custom collate function to handle batching of images and article IDs."""
    pixel_values, article_ids = zip(*batch)
    pixel_values = torch.stack(pixel_values)
    return pixel_values, list(article_ids)


def create_image_dataloader(dataset):
    return DataLoader(
        dataset,
        batch_size=settings.IMAGE_BATCH_SIZE,
        num_workers=settings.NUM_WORKERS,
        pin_memory=settings.PIN_MEMORY,
        shuffle=False,
        collate_fn=_collate_fn,
    )
