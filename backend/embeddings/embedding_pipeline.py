import pandas as pd
import numpy as np
from transformers import logging as transformers_logging
from utils.model_utils import load_clip_model_and_processor
from embeddings.embedding_utils import (
    save_text_embeddings,  
    generate_clip_text_embeddings, 
)
from config import (
    CSV_PATH, 
    EMBEDDING_SAVE_PATH,
    COMPLETE_CSV_PATH, 
    TEXT_BATCH_SIZE,
)

def run_embedding_pipeline():
    transformers_logging.set_verbosity_error()
    
    print("⏳ Loading CLIP model...")
    model, processor = load_clip_model_and_processor()
    
    print("📄 Loading data...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} articles")
    
    print("📝 Generating CLIP text embeddings...")
    
    text_embeddings = generate_clip_text_embeddings(
        df, model, processor, batch_size=TEXT_BATCH_SIZE
    )
    
    print(f"✅ Generated {text_embeddings.shape[0]} text embeddings of dimension {text_embeddings.shape[1]}")
    
    successful_indices = list(range(len(df)))
    
    print("💾 Saving text-only embeddings...")
    save_text_embeddings(text_embeddings, EMBEDDING_SAVE_PATH, successful_indices)
    
    df.to_csv(COMPLETE_CSV_PATH, index=False)
    print(f"📊 Saved complete dataset with {len(df)} articles")
    
    print("✅ Embedding pipeline completed successfully!")
    print(f"   - Embedding dimension: {text_embeddings.shape[1]}")
    print(f"   - Total articles embedded: {len(successful_indices)}")
    print(f"   - Saved to: {EMBEDDING_SAVE_PATH}")
