import torch
from torch.utils.data import DataLoader
from config import IMAGE_BATCH_SIZE, NUM_WORKERS, PIN_MEMORY

def collate_fn(batch):
    pixel_values, indices, article_ids = zip(*batch)
    pixel_values = torch.stack(pixel_values)
    return pixel_values, list(indices), list(article_ids)

def create_image_dataloader(dataset):
    return DataLoader(
        dataset,
        batch_size=IMAGE_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        collate_fn=collate_fn,
    )
