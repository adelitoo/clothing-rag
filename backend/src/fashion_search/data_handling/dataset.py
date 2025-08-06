import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, df, image_base_dir, processor):
        self.df = df.reset_index(drop=True)
        self.image_base_dir = image_base_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        article_id = str(row["article_id"]).zfill(10)

        subfolder = article_id[:3]
        image_path = os.path.join(self.image_base_dir, subfolder, f"{article_id}.jpg")

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs["pixel_values"].squeeze(0), article_id
        except Exception as e:
            print(f"❌ Failed to load image at {image_path}: {e}")
            dummy_tensor = torch.zeros(
                (3, self.processor.size["height"], self.processor.size["width"])
            )
            return dummy_tensor, article_id
