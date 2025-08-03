# services/redis_search_service.py

from typing import Dict, Any
from redis_client.redis_db_client import RedisDBClient
from milvus.vector_db_client import VectorDBClient
from llm.query_enhancer import LLMQueryEnhancer
from embeddings.embedding_utils import embed_text_query


class RedisSearchService:
    def __init__(
        self,
        redis_client: RedisDBClient,
        db_client: VectorDBClient,
        llm_enhancer: LLMQueryEnhancer,
        model,
        processor,
    ):
        self.redis_client = redis_client
        self.milvus = db_client
        self.llm = llm_enhancer
        self.model = model
        self.processor = processor

    def search(self, query: str, top_k: int) -> Dict[str, Any]:
        cache_key = f"cache:query:{query}"

        ## FIX: Corrected self.redis to self.redis_client to fix AttributeError.
        cached_data = self.redis_client.get_json(cache_key)
        if cached_data:
            print(f"✅ Cache HIT for query: '{query}'")
            return {**cached_data, "source": "cache"}

        print(f"❌ Cache MISS for query: '{query}'")
        transformed_query = self.llm.transform(query)
        if not transformed_query:
            transformed_query = query

        transformed_query = transformed_query.strip().strip('"').strip("'")
        summary = self.llm.summarize(transformed_query)

        query_embedding = [
            float(num)
            for num in embed_text_query(self.model, self.processor, transformed_query)
        ]

        raw_milvus_hits = self.milvus.search([query_embedding], top_k=top_k)

        data_to_cache = {
            "transformed_query": transformed_query,
            "summary": summary,
            "milvus_results": raw_milvus_hits,
        }
        ## FIX: Corrected self.redis to self.redis_client to fix AttributeError.
        self.redis_client.set_json(cache_key, data_to_cache, ttl=3600)

        return {**data_to_cache, "source": "live"}

    def search_baseline(self, query: str, top_k: int) -> Dict[str, Any]:
        print(f"Executing baseline search for query: '{query}'")

        query_embedding = [
            float(num) for num in embed_text_query(self.model, self.processor, query)
        ]

        raw_milvus_hits = self.milvus.search([query_embedding], top_k=top_k)

        return {
            "transformed_query": query,
            "summary": "Baseline search does not provide a summary.",
            "milvus_results": raw_milvus_hits,
        }
