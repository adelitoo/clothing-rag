import numpy as np
import pandas as pd
from typing import Dict, Any

import config
from .base import PipelineStep
from preprocessing.cleanup_dataset import clean_csv
from captioning.captioning_pipeline import CaptioningPipeline
from embeddings.embedding_pipeline import EmbeddingPipeline
from milvus.vector_db_client import VectorDBClient

class CleanupStep(PipelineStep):
    def run(self) -> Dict[str, Any]:
        print("🚀 [1/4] Starting Dataset Cleanup...")
        _, count = clean_csv()
        print("✅ Cleanup complete.")
        return {"status": "OK", "articles_kept": count}

class CaptioningStep(PipelineStep):
    def run(self) -> Dict[str, Any]:
        print("🚀 [2/4] Starting Image Captioning...")
        caption_pipeline = CaptioningPipeline(config)
        caption_results = caption_pipeline.run()
        print("✅ Captioning complete.")
        return {"status": "OK", **caption_results}

class EmbeddingStep(PipelineStep):
    def run(self) -> Dict[str, Any]:
        print("🚀 [3/4] Starting Text Embedding Generation...")
        embedding_pipeline = EmbeddingPipeline(config)
        embedding_results = embedding_pipeline.run()
        print("✅ Embedding generation complete.")
        return {"status": "OK", **embedding_results}

class DbInsertionStep(PipelineStep):
    def __init__(self, db_client: VectorDBClient):
        self.db_client = db_client

    def run(self) -> Dict[str, Any]:
        print("🚀 [4/4] Starting DB Insertion...")
        
        data = np.load(config.EMBEDDING_SAVE_PATH, allow_pickle=False)
        embeddings, indices = data["embeddings"], data["indices"]
        
        df = pd.read_csv(config.COMPLETE_ARTICLES_CSV_PATH)
        if len(df) != len(indices):
            print(f"⚠️ Warning: Mismatch between CSV rows ({len(df)}) and embedding indices ({len(indices)}). Using indices to slice.")
            df = df.iloc[indices]

        article_ids = df["article_id"].tolist()
        
        self.db_client.set_collection("articles", dim=embeddings.shape[1], recreate=True)
        self.db_client.insert(article_ids, embeddings)
        self.db_client.create_index()
        
        print("✅ DB Insertion complete.")
        return {"status": "OK", "vectors_inserted": len(article_ids)}
