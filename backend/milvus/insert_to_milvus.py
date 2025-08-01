import os
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from functools import lru_cache
from yaspin import yaspin
from yaspin.spinners import Spinners
from config import MILVUS_HOST, MILVUS_PORT, INS_BATCH_SIZE

def connect_milvus():
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

def create_collection(name="articles", dim=512):  
    """Create collection optimized for text embeddings with COSINE similarity"""
    fields = [
        FieldSchema(
            name="article_id", dtype=DataType.INT64, is_primary=True, auto_id=False
        ),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="Article text embeddings")
    collection = Collection(name, schema)
    
    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "COSINE",  
            "index_type": "IVF_FLAT",
            "params": {"nlist": 256},  
        },
    )
    
    print(f"✅ Created collection '{name}' with COSINE similarity and dim={dim}")
    return collection

def insert_embeddings(collection, article_ids, embeddings, batch_size=INS_BATCH_SIZE):
    total = len(article_ids)
    with yaspin(Spinners.arc, text="Starting insertion…") as spinner:
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_ids = article_ids[start:end]
            batch_vectors = embeddings[start:end].tolist()
            spinner.text = f"Inserting {start:>6}–{end:<6} of {total}"
            collection.insert([batch_ids, batch_vectors])
        collection.flush()
        spinner.ok("✅ Inserted all batches")

@lru_cache(maxsize=1024)
def get_image_path(article_id: int) -> str:
    padded_id = str(article_id).zfill(10)
    folder_name = padded_id[:3]
    filename = f"{padded_id}.jpg"
    return os.path.join("data", "images", folder_name, filename)

def search_similar_items(collection, query_embedding, top_k=10):
    search_params = {
        "metric_type": "COSINE",  
        "params": {"nprobe": 64}  
    }
    
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["article_id"],
    )
    
    hits = results[0]
    result_items = []
    
    for hit in hits:
        article_id = hit.entity.get("article_id") if hasattr(hit, "entity") else hit.id
        score = hit.distance  
        image_path = get_image_path(article_id)
        
        if not os.path.exists(image_path):
            image_path = None
            
        result_items.append(
            {"article_id": article_id, "image": image_path, "score": score}
        )
    
    return result_items
