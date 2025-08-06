import numpy as np
import pandas as pd
from typing import Dict, Any

from ..core.config import settings
from .base import PipelineStep
from ..preprocessing.cleanup import clean_csv
from ..captioning.captioning_pipeline import CaptioningPipeline
from ..embeddings.embedding_pipeline import EmbeddingPipeline
from ..milvus_client.vector_db_client import VectorDBClient

class CleanupStep(PipelineStep):
    def run(self) -> Dict[str, Any]:
        print("🚀 [1/4] Starting Dataset Cleanup...")
        _, count = clean_csv()
        print("✅ Cleanup complete.")
        return {"status": "OK", "articles_kept": count}

class CaptioningStep(PipelineStep):
    def run(self) -> Dict[str, Any]:
        print("🚀 [2/4] Starting Image Captioning...")
        caption_pipeline = CaptioningPipeline()
        caption_results = caption_pipeline.run()
        print("✅ Captioning complete.")
        return {"status": "OK", **caption_results}

class EmbeddingStep(PipelineStep):
    def run(self) -> Dict[str, Any]:
        print("🚀 [3/4] Starting Text Embedding Generation...")
        embedding_pipeline = EmbeddingPipeline()
        embedding_results = embedding_pipeline.run()
        print("✅ Embedding generation complete.")
        return {"status": "OK", **embedding_results}

class DbInsertionStep(PipelineStep):
    def __init__(self, db_client: VectorDBClient):
        self.db_client = db_client

    def run(self) -> Dict[str, Any]:
        print("🚀 [4/4] Starting DB Insertion...")
        try:
            df = pd.read_csv(settings.COMPLETE_ARTICLES_CSV_PATH, dtype={'article_id': str})
            embeddings_data = np.load(settings.EMBEDDING_SAVE_PATH, allow_pickle=True)
            
            embeddings = embeddings_data['embeddings']
            article_ids = [str(aid) for aid in embeddings_data['article_ids']]
            
            df_filtered = df[df['article_id'].isin(article_ids)].set_index('article_id').loc[article_ids].reset_index()

            self.db_client.set_collection("articles", recreate=True)
            self.db_client.insert(df_filtered, embeddings)
            self.db_client.create_index()
            
            print("✅ DB Insertion complete.")
            return {"status": "success", "inserted_count": len(article_ids)}
        except Exception as e:
            print(f"--- ❌ DB Insertion Step Failed: {e} ---")
            return {"status": "failed", "error": str(e)}