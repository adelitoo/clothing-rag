import torch
from functools import lru_cache


@lru_cache(maxsize=256)
def embed_text_query(model, processor, text: str) -> list[float]:
    if not text or not text.strip():
        try:
            dim = model.text_projection.shape[1]
        except AttributeError:
            dim = 512
        return [0.0] * dim

    device = next(model.parameters()).device
    inputs = processor(
        text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77
    ).to(device)

    with torch.no_grad():
        if hasattr(model, "get_text_features"):
            text_embedding = model.get_text_features(**inputs)
        else:
            outputs = model.text_model(**inputs)
            text_embedding = outputs.pooler_output

        norm = text_embedding.norm(p=2, dim=-1, keepdim=True)
        epsilon = 1e-8
        normalized_embedding = text_embedding / (norm + epsilon)

    return normalized_embedding[0].cpu().numpy().tolist()
