import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import logging as transformers_logging
from ..core.model_loader import load_clip_model_and_processor

def create_rich_text_description(row: pd.Series) -> str:
    parts = [
        row.get('prod_name'),
        row.get('product_type_name'),
        row.get('product_group_name'),
        row.get('colour_group_name'),
        row.get('perceived_colour_value_name'),
        row.get('perceived_colour_master_name'),
        row.get('department_name'),
        row.get('section_name'),
        row.get('garment_group_name'),
        row.get('detail_desc'),
        row.get('img_caption')
    ]
    return ' '.join([str(p) for p in parts if pd.notna(p)])


class EmbeddingPipeline:
    def __init__(self, config):
        self.config = config
        transformers_logging.set_verbosity_error()
        print("⏳ Loading CLIP model and processor...")
        self.model, self.processor = load_clip_model_and_processor()
        self.device = self.config.DEVICE
        self.model.to(self.device)

    def _load_data(self) -> pd.DataFrame:
        print(f"📄 Loading data from: {self.config.COMPLETE_ARTICLES_CSV_PATH}")
        df = pd.read_csv(self.config.COMPLETE_ARTICLES_CSV_PATH)
        print(f"   Loaded {len(df)} articles.")
        return df

    def _create_rich_text(self, df: pd.DataFrame) -> list[str]:
        print("📝 Creating rich text descriptions from metadata...")
        rich_texts = [create_rich_text_description(row) for _, row in tqdm(df.iterrows(), total=len(df), desc="Featurizing")]
        return rich_texts

    def _generate_embeddings(self, texts: list[str]) -> np.ndarray:
        print(f"🔄 Processing {len(texts)} texts in batches of {self.config.TEXT_BATCH_SIZE}...")
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.config.TEXT_BATCH_SIZE), desc="Embedding Batches"):
            batch_texts = texts[i:i + self.config.TEXT_BATCH_SIZE]
            inputs = self.processor(
                text=batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=77
            ).to(self.device)
            
            with torch.no_grad():
                batch_embeds = self.model.get_text_features(**inputs)
                batch_embeds = batch_embeds / batch_embeds.norm(dim=-1, keepdim=True)
                all_embeddings.append(batch_embeds.cpu())
        
        return torch.cat(all_embeddings, dim=0).numpy()

    def _save_artifacts(self, df: pd.DataFrame, embeddings: np.ndarray):
        print("💾 Saving artifacts...")
        
        save_path = self.config.EMBEDDING_SAVE_PATH
        np.savez(
            save_path,
            embeddings=embeddings,
            indices=np.arange(len(embeddings), dtype=np.int64),
            embedding_type="text_only"
        )
        print(f"   - Saved {len(embeddings)} embeddings to {save_path}")

        df.to_csv(self.config.COMPLETE_ARTICLES_CSV_PATH, index=False)
        print(f"   - Saved complete dataset with {len(df)} articles to {self.config.COMPLETE_ARTICLES_CSV_PATH}")

    def run(self) -> dict:
        df = self._load_data()
        texts = self._create_rich_text(df)
        embeddings = self._generate_embeddings(texts)
        self._save_artifacts(df, embeddings)
        
        print("\n✅ Embedding pipeline completed successfully!")
        
        return {
            "count": len(embeddings),
            "dimension": embeddings.shape[1],
            "save_path": str(self.config.EMBEDDING_SAVE_PATH)
        }