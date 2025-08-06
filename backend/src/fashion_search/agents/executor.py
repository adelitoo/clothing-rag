from typing import List, Dict
from ..schemas.agent_schemas import SearchResult, OutfitPlan
from ..services.redis_search_service import RedisSearchService
from ..milvus_client.vector_db_client import VectorDBClient
from ..embeddings.embedding_utils import embed_text_query

class FashionSearchExecutor:
    def __init__(self, search_service: RedisSearchService):
        self.search_service = search_service
        self.db_client: VectorDBClient = self.search_service.milvus

    def _build_filter_expression(self, filters: Dict[str, str]) -> str | None:
        if not filters:
            return None
        
        parts = []
        for key, value in filters.items():
            escaped_value = str(value).replace("'", "\\'")
            parts.append(f"{key} == '{escaped_value}'")
        
        expression = " and ".join(parts)
        return expression if expression else None

    def search_category(self, plan: OutfitPlan, category: str, description: str, top_k: int = 12) -> List[SearchResult]:
        try:
            filter_expr = self._build_filter_expression(plan.filters)
            
            print(f"🔍 Searching {category}: '{description}' with filter: {filter_expr}")

            query_embedding = embed_text_query(
                model=self.search_service.model,
                processor=self.search_service.processor,
                text=description
            )
            
            milvus_hits = self.db_client.search(
                vectors=[query_embedding], 
                top_k=top_k, 
                filter_expression=filter_expr
            )

            search_results = [
                SearchResult(
                    article_id=str(hit.get("article_id")),
                    score=hit.get("score", 0.0),
                    category=category,
                )
                for hit in milvus_hits
            ]

            print(f"✅ Found {len(search_results)} {category} items")
            return search_results

        except Exception as e:
            print(f"❌ Search failed for {category}: {e}")
            return []