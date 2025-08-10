import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import logging as transformers_logging

from ..core.config import settings
from ..core.model_loader import load_clip_model_and_processor

class EmbeddingPipeline:
    def __init__(self):
        transformers_logging.set_verbosity_error()
        self.device = settings.DEVICE
        self.model, self.processor = load_clip_model_and_processor()
        self.model.to(self.device)

    @staticmethod
    def _create_rich_text_description(row: pd.Series) -> str:
        parts = [
            row.get('prod_name'),
            row.get('product_type_name'),
            row.get('product_group_name'),
            row.get('colour_group_name'),
            row.get('detail_desc'),
            row.get('img_caption')
        ]
        return ' '.join([str(p) for p in parts if pd.notna(p) and str(p).strip()])

    def _load_data(self) -> pd.DataFrame:
        print(f"📄 Loading data from: {settings.COMPLETE_ARTICLES_CSV_PATH}")
        df = pd.read_csv(settings.COMPLETE_ARTICLES_CSV_PATH, dtype={'article_id': str})
        print(f"   - Loaded {len(df)} articles.")
        return df

    def _create_rich_text(self, df: pd.DataFrame) -> list[str]:
        print("📝 Creating rich text descriptions from metadata...")
        tqdm.pandas(desc="Featurizing rows")
        rich_texts = df.progress_apply(self._create_rich_text_description, axis=1).tolist()
        return rich_texts

    def _generate_embeddings(self, texts: list[str]) -> np.ndarray:
        print(f"🧠 Generating embeddings in batches of {settings.TEXT_BATCH_SIZE}...")
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), settings.TEXT_BATCH_SIZE), desc="Embedding Batches"):
                batch_texts = texts[i:i + settings.TEXT_BATCH_SIZE]
                inputs = self.processor(
                    text=batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=77
                ).to(self.device)
                
                batch_embeds = self.model.get_text_features(**inputs)
                batch_embeds /= batch_embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(batch_embeds.cpu().numpy())
        
        return np.vstack(all_embeddings)

    def _save_artifacts(self, df: pd.DataFrame, embeddings: np.ndarray):
        print(f"💾 Saving artifacts to {settings.EMBEDDING_SAVE_PATH}...")
        article_ids_array = df["article_id"].to_numpy()

        np.savez(
            settings.EMBEDDING_SAVE_PATH,
            embeddings=embeddings,
            article_ids=article_ids_array
        )

        print(f"   - Saved {len(embeddings)} embeddings and {len(article_ids_array)} article IDs.")

    def run(self) -> dict:
        df = self._load_data()
        texts = self._create_rich_text(df)
        embeddings = self._generate_embeddings(texts)
        self._save_artifacts(df, embeddings)
        
        print("\n✅ Embedding pipeline completed successfully!")
        
        return {
            "status": "success",
            "embeddings_generated": len(embeddings),
        }