import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from config import DEVICE

def create_rich_text_description(row):
    parts = []
    
    if pd.notna(row.get('prod_name')):
        parts.append(str(row['prod_name']))
    
    if pd.notna(row.get('product_type_name')):
        parts.append(str(row['product_type_name']))
    
    if pd.notna(row.get('product_group_name')):
        parts.append(str(row['product_group_name']))
    
    color_parts = []
    for color_field in ['colour_group_name', 'perceived_colour_value_name', 'perceived_colour_master_name']:
        if pd.notna(row.get(color_field)):
            color_parts.append(str(row[color_field]))
    
    if color_parts:
        parts.append(' '.join(color_parts))
    
    for field in ['department_name', 'section_name', 'garment_group_name']:
        if pd.notna(row.get(field)):
            parts.append(str(row[field]))
    
    if pd.notna(row.get('detail_desc')):
        parts.append(str(row['detail_desc']))
    
    if pd.notna(row.get('img_caption')):
        parts.append(str(row['img_caption']))
    
    return ' '.join(parts)

def generate_clip_text_embeddings(df, model, processor, batch_size=64):    
    print("📝 Creating rich text descriptions from metadata...")
    texts = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        rich_text = create_rich_text_description(row)
        texts.append(rich_text)
    
    print(f"📋 Sample text description:")
    print(f"   {texts[0][:200]}...")
    print(f"   Length: {len(texts[0])} characters")
    
    device = next(model.parameters()).device
    embeddings = []
    
    print(f"🔄 Processing {len(texts)} texts in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Text embedding batches"):
        batch_texts = texts[i:i + batch_size]
        
        inputs = processor(
            text=batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=77  
        ).to(device)
        
        with torch.no_grad():
            batch_embeds = model.get_text_features(**inputs)
            batch_embeds = batch_embeds / batch_embeds.norm(dim=-1, keepdim=True)
            embeddings.append(batch_embeds.cpu())
    
    text_embeds = torch.cat(embeddings, dim=0).numpy()
    
    print(f"✅ Generated {text_embeds.shape[0]} embeddings of dimension {text_embeds.shape[1]}")
    return text_embeds

def save_text_embeddings(text_embeddings, path, successful_indices=None):
    if successful_indices is not None:
        embeddings_to_save = text_embeddings[successful_indices]
        indices_to_save = np.array(successful_indices, dtype=np.int64)
    else:
        embeddings_to_save = text_embeddings
        indices_to_save = np.arange(len(text_embeddings), dtype=np.int64)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path,
             embeddings=embeddings_to_save,
             indices=indices_to_save,
             embedding_type="text_only")  
    
    print(f"✅ Saved {embeddings_to_save.shape[0]} text embeddings of dim {embeddings_to_save.shape[1]} to {path}")

def embed_text_query(model, processor, text: str):
    device = next(model.parameters()).device
    
    inputs = processor(
        text=[text], 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=77
    ).to(device)
    
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    
    return text_embedding[0].cpu().numpy().tolist()
