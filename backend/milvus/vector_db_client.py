from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from yaspin import yaspin
import numpy as np


class VectorDBClient:
    def __init__(self, host: str, port: str | int):
        self.host = host
        self.port = port
        self.collection = None
        self._connect()

    def _connect(self):
        try:
            print(f"🔌 Connecting to Milvus at {self.host}:{self.port}")
            connections.connect("default", host=self.host, port=self.port)
        except Exception as e:
            print(f"❌ Milvus connection failed: {e}")
            raise

    def set_collection(self, name: str, dim: int, recreate: bool = False):
        if recreate and utility.has_collection(name):
            print(f"🗑️ Dropping existing collection: {name}")
            Collection(name).drop()
        if not utility.has_collection(name):
            self._create_collection_schema(name, dim)
        self.collection = Collection(name)
        print(f"✅ Collection '{name}' is ready.")

    def _create_collection_schema(self, name: str, dim: int):
        fields = [
            FieldSchema(name="article_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, description=f"{name} embeddings")
        self.collection = Collection(name, schema)
        print(f"  - Created collection '{name}' with schema.")

    def insert(self, ids: list, embeddings: np.ndarray, batch_size: int = 1000):
        if not self.collection:
            raise Exception("Collection not set. Call set_collection() first.")
        total = len(ids)
        with yaspin(text="Starting insertion...", color="yellow") as spinner:
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                spinner.text = f"➡️ Inserting batch {start:>6}–{end:<6} of {total}"
                self.collection.insert([ids[start:end], embeddings[start:end]])
            spinner.text = "⏳ Flushing data to Milvus (this can take a few minutes)..."
            self.collection.flush()
            spinner.ok("✅")
            spinner.text = f"Insertion and flush complete for {total} items."

    def create_index(self):
        if not self.collection:
            raise Exception("Collection not set. Call set_collection() first.")

        print("⏳ Creating index on 'embedding' field...")
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 256},
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        print("✅ Index created successfully.")
        print("⏳ Loading collection into memory...")
        self.collection.load()
        print("✅ Collection loaded.")

    def search(self, vectors: list[list[float]], top_k: int) -> list[dict]:
        if not self.collection:
            raise Exception("Collection not set.")

        self.collection.load()

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 64}}

        results = self.collection.search(
            data=vectors,
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["article_id"],
        )

        hits = []
        for hit in results[0]:
            hits.append(
                {"article_id": hit.entity.get("article_id"), "score": hit.distance}
            )

        print(f"✅ Search returned {len(hits)} results")
        if hits:
            print(
                f"  First result: article_id={hits[0]['article_id']}, score={hits[0]['score']}"
            )

        return hits
