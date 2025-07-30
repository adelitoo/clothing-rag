import os
import torch                   
from PIL import Image          
from torch.utils.data import Dataset
from tqdm import tqdm 
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, df, image_base_dir, processor=None, validate=True):
        self.df = df
        self.image_base_dir = image_base_dir
        self.processor = processor
        self.valid_items = []

        if validate:
            print("Pre-filtering valid images...")
            for idx, article_id in enumerate(
                tqdm(df["article_id"], desc="Checking images")
            ):
                padded_id = str(article_id).zfill(10)
                subfolder = padded_id[:3]
                image_path = os.path.join(image_base_dir, subfolder, f"{padded_id}.jpg")

                if os.path.exists(image_path):
                    try:
                        with Image.open(image_path) as img:
                            img.verify()
                        self.valid_items.append((idx, article_id, image_path))
                    except:
                        try:
                            os.remove(image_path)
                        except Exception as e:
                            print(f"⚠️ Could not delete {image_path}: {e}")
                        continue

            print(f"Found {len(self.valid_items)} valid images out of {len(df)}")
        else:
            for idx, article_id in enumerate(df["article_id"]):
                padded_id = str(article_id).zfill(10)
                subfolder = padded_id[:3]
                image_path = os.path.join(image_base_dir, subfolder, f"{padded_id}.jpg")
                if os.path.exists(image_path):
                    self.valid_items.append((idx, article_id, image_path))

    def get_valid_indices(self):
        return [item[0] for item in self.valid_items]
    
    def __len__(self):
        return len(self.valid_items)

    def __getitem__(self, idx):
        original_idx, article_id, image_path = self.valid_items[idx]

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
            return pixel_values, original_idx, article_id
        except Exception as e:
            print(f"❌ Failed to load image at {image_path}: {e}")
            dummy_tensor = torch.zeros((3, 224, 224))
            return dummy_tensor, original_idx, article_id
