from pymilvus import (
    connections,
    CollectionSchema,
    Collection,
    utility,
)
from yaspin import yaspin
import pandas as pd
import numpy as np
from ..core.config import settings

class VectorDBClient:

    def __init__(self, host: str, port: str | int):
        self.host = host
        self.port = port
        self.collection = None
        self.field_names = [field.name for field in settings.SCHEMA_FIELDS]
        self.scalar_field_names = [field.name for field in settings.SCHEMA_FIELDS if field.name != "embedding"]
        self._connect()

    def _connect(self):
        try:
            connections.connect("default", host=self.host, port=self.port)
        except Exception as e:
            print(f"❌ Milvus connection failed: {e}")
            raise

    def set_collection(self, name: str, recreate: bool = False):
        if recreate and utility.has_collection(name):
            print(f"🗑️ Dropping existing collection: {name}")
            Collection(name).drop()
        if not utility.has_collection(name):
            self._create_collection_schema(name)
        self.collection = Collection(name)

    def _create_collection_schema(self, name: str):
        schema = CollectionSchema(settings.SCHEMA_FIELDS, description="Fashion articles with hybrid search metadata")
        self.collection = Collection(name, schema)
        print(f"  - Created collection '{name}' from the central schema definition.")

    def insert(self, data_df: pd.DataFrame, embeddings: np.ndarray, batch_size: int = 1000):
        if not self.collection:
            raise Exception("Collection not set.")
        
        data_df['embedding'] = list(embeddings)
        total = len(data_df)

        with yaspin(text="Starting insertion...", color="yellow") as spinner:
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                batch_df = data_df.iloc[start:end]
                
                batch_data = [list(batch_df[field_name]) for field_name in self.field_names]
                
                spinner.text = f"➡️ Inserting batch {start:>6}–{end:<6} of {total}"
                self.collection.insert(batch_data)
                
            spinner.text = "⏳ Flushing data to Milvus..."
            self.collection.flush()
            spinner.ok("✅")
            spinner.text = f"Insertion and flush complete for {total} items."

    def create_index(self):
        if not self.collection:
            raise Exception("Collection not set.")

        vector_index_params = {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 256}}
        self.collection.create_index(field_name="embedding", index_params=vector_index_params)

        for field_name in self.scalar_field_names:
            if not self.collection.has_index(index_name=f"idx_{field_name}") and field_name != "article_id":
                self.collection.create_index(field_name=field_name, index_name=f"idx_{field_name}")

        self.collection.load()

    def search(self, vectors: list[list[float]], top_k: int, filter_expression: str = None) -> list[dict]:
        if not self.collection:
            raise Exception("Collection not set.")

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 64}}
        results = self.collection.search(
            data=vectors,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=self.scalar_field_names, 
            expr=filter_expression,
        )

        hits = []
        for hit in results[0]:
            entity_data = {field: hit.entity.get(field) for field in self.scalar_field_names}
            entity_data['score'] = hit.distance
            hits.append(entity_data)
        
        print(f"✅ Search returned {len(hits)} results")
        if hits:
            print(f"  First result: article_id={hits[0]['article_id']}, score={hits[0]['score']}")
        return hits