import pandas as pd
import torch
from tqdm import tqdm

from ..core.config import settings
from ..core.model_loader import load_captioning_model_and_processor
from ..data_handling.dataset import ImageDataset
from ..data_handling.dataloader import create_image_dataloader

GENERATION_CONFIG = {
    "max_length": 50,
    "num_beams": 4,
    "repetition_penalty": 1.5,
    "early_stopping": True,
}

class CaptioningPipeline:
    def __init__(self):
        self.device = settings.DEVICE
        self.processor, self.model = load_captioning_model_and_processor()
        self.model.to(self.device).eval()

    def _generate_captions(self, dataloader) -> dict:
        captions_dict = {}
        print("✍️ Generating image captions...")

        for pixel_values, article_ids in tqdm(dataloader, desc="Captioning Batches"):
            pixel_values = pixel_values.to(self.device)

            try:
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        pixel_values=pixel_values, **GENERATION_CONFIG
                    )
                
                captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                for art_id, caption in zip(article_ids, captions):
                    captions_dict[art_id] = caption.strip()
            except Exception as e:
                print(f"⚠️ A batch failed during captioning: {e}")
                for art_id in article_ids:
                    captions_dict[art_id] = "caption_generation_failed"
                    
        return captions_dict

    def run(self) -> dict:
        print("--- 🏃 Running Captioning Pipeline ---")
        try:
            df = pd.read_csv(settings.ARTICLES_CSV_PATH, dtype={'article_id': str})
        except FileNotFoundError:
            print(f"❌ Error: Raw articles file not found at {settings.ARTICLES_CSV_PATH}")
            return {"status": "failed", "error": "articles.csv not found"}

        dataset = ImageDataset(df, settings.IMAGE_BASE_DIR, self.processor)
        dataloader = create_image_dataloader(dataset)
        
        captions_dict = self._generate_captions(dataloader)
        
        print("💾 Merging captions and saving to processed CSV...")
        df["img_caption"] = df["article_id"].map(captions_dict)
        
        original_count = len(df)
        df.dropna(subset=['img_caption'], inplace=True)
        df = df[df["img_caption"] != "caption_generation_failed"].copy()
        final_count = len(df)

        df.to_csv(settings.COMPLETE_ARTICLES_CSV_PATH, index=False)
        print(f"   - Generated and saved {final_count}/{original_count} captions.")
        print(f"   - Updated CSV saved to {settings.COMPLETE_ARTICLES_CSV_PATH}")
        
        print("--- ✅ Captioning Pipeline Finished ---")
        return {"status": "success", "captions_generated": final_count}