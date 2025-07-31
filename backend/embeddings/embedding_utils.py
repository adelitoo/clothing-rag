import torch
import numpy as np

def embed_text_query(model, processor, text: str) -> list[float]:
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