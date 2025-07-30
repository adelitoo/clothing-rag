import torch
import os
import pandas as pd
from tqdm import tqdm

from transformers import logging as transformers_logging

from utils.dataloader_utils import create_image_dataloader
from utils.model_utils import load_captioning_model_and_processor
from models.image_dataset import ImageDataset
from config import (
    CSV_PATH,
    IMAGE_BASE_DIR,
    DEVICE,
)

def caption_images(dataloader, model, processor, device):
    captions_dict = {}
    model.eval()

    for batch in tqdm(dataloader, desc="🖼️ Captioning images"):
        pixel_values, indices, article_ids = batch
        pixel_values = pixel_values.to(device)

        try:
            with torch.no_grad(), torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                generated_ids = model.generate(
                    pixel_values=pixel_values,
                    max_length=50,
                    num_beams=5,
                    repetition_penalty=1.5,
                    early_stopping=True
                )
            captions = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for art_id, caption in zip(article_ids, captions):
                captions_dict[art_id] = caption

        except RuntimeError as e:
            print(f"⚠️ Batch failed: {str(e)}")
            for art_id in article_ids:
                captions_dict[art_id] = "caption_generation_failed"


    return captions_dict


def generate_and_save_captions():
    os.environ["TRANSFORMERS_NO_TQDM"] = "1"
    transformers_logging.set_verbosity_error()

    df = pd.read_csv(CSV_PATH)
    print(f"📄 Loaded {len(df)} articles")

    processor, model = load_captioning_model_and_processor()

    dataset = ImageDataset(df, IMAGE_BASE_DIR, processor, validate=False)
    dataloader = create_image_dataloader(dataset)

    captions_dict = caption_images(dataloader, model, processor, DEVICE)
    df["img_caption"] = df["article_id"].map(captions_dict)

    filtered_df = df[df["img_caption"] != "caption_generation_failed"]
    print(f"✅ Generated captions for {len(filtered_df)}/{len(df)} images")

    filtered_df.to_csv(CSV_PATH, index=False)
    print(f"💾 Updated CSV saved to {CSV_PATH}")

    return filtered_df