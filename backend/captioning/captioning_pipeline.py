import pandas as pd
import torch
from tqdm import tqdm
from transformers import logging as transformers_logging

from utils.dataloader_utils import create_image_dataloader
from utils.model_utils import load_captioning_model_and_processor
from utils.dataset_utils import ImageDataset

class CaptioningPipeline:
    def __init__(self, config):
        self.config = config
        transformers_logging.set_verbosity_error()
        
        print("⏳ Loading Captioning model and processor...")
        self.processor, self.model = load_captioning_model_and_processor()
        self.device = self.config.DEVICE
        self.model.to(self.device)

    def _prepare_dataloader(self, df: pd.DataFrame):
        print("🛠️ Preparing dataset and dataloader...")
        dataset = ImageDataset(df, self.config.IMAGE_BASE_DIR, self.processor)
        dataloader = create_image_dataloader(dataset)
        return dataloader

    def _generate_captions(self, dataloader) -> dict:
        captions_dict = {}
        self.model.eval()

        for batch in tqdm(dataloader, desc="🖼️ Captioning images"):
            pixel_values, _, article_ids = batch
            pixel_values = pixel_values.to(self.device)

            try:
                with torch.no_grad(), torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu'):
                    generated_ids = self.model.generate(
                        pixel_values=pixel_values, max_length=50, num_beams=5,
                        repetition_penalty=1.5, early_stopping=True
                    )
                captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                for art_id, caption in zip(article_ids, captions):
                    captions_dict[art_id] = caption
            except RuntimeError as e:
                print(f"⚠️ Batch failed: {e}")
                for art_id in article_ids:
                    captions_dict[art_id] = "caption_generation_failed"
        return captions_dict

    def _save_results(self, df: pd.DataFrame, captions_dict: dict) -> int:
        print("💾 Merging results and saving...")
        df["img_caption"] = df["article_id"].map(captions_dict)

        original_count = len(df)
        filtered_df = df[df["img_caption"] != "caption_generation_failed"].copy()
        final_count = len(filtered_df)
        
        save_path = self.config.ARTICLES_CSV_PATH 
        filtered_df.to_csv(save_path, index=False)
        print(f"   Generated captions for {final_count}/{original_count} images.")
        print(f"   Updated CSV saved to {save_path}")
        return final_count

    def run(self) -> dict:
        print("🚀 Starting captioning pipeline...")
        df = pd.read_csv(self.config.ARTICLES_CSV_PATH)
        dataloader = self._prepare_dataloader(df)
        captions_dict = self._generate_captions(dataloader)
        final_count = self._save_results(df, captions_dict)
        
        print("\n✅ Captioning pipeline completed successfully!")
        return {"captions_generated": final_count}