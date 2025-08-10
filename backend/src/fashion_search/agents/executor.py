import json
from typing import List, Dict
from ..schemas.agent_schemas import SearchResult, OutfitPlan
from ..services.redis_search_service import RedisSearchService
from ..milvus_client.vector_db_client import VectorDBClient
from ..embeddings.embedding_utils import embed_text_query


class FashionSearchExecutor:
    def __init__(self, search_service: RedisSearchService):
        self.search_service = search_service
        self.db_client: VectorDBClient = self.search_service.milvus

        print("🔄 FashionSearchExecutor: Loading category map from Redis...")
        try:
            raw_map = self.search_service.redis_client.get_hash("app:category_map")
            self.category_map = {
                key: json.loads(value) for key, value in raw_map.items()
            }

            if not self.category_map:
                print(
                    "⚠️ WARNING: Category map not found in Redis. Hybrid search filtering will be limited."
                )
            else:
                print(f"✅ Loaded {len(self.category_map)} category mappings.")
        except Exception as e:
            print(f"❌ CRITICAL: Failed to load or parse category map from Redis: {e}")
            self.category_map = {}

    def _build_filter_expression(self, filters: Dict[str, any]) -> str | None:
        if not filters:
            return None

        parts = []
        for key, value in filters.items():
            if isinstance(value, list):
                formatted_list = json.dumps(value)
                parts.append(f"{key} IN {formatted_list}")
            else:
                escaped_value = str(value).replace("'", "\\'")
                parts.append(f"{key} == '{escaped_value}'")

        expression = " and ".join(parts)
        return expression if expression else None

    def search_category(
        self, plan: OutfitPlan, category: str, description: str, top_k: int = 12
    ) -> List[SearchResult]:
        try:
            current_filters = plan.filters.copy()

            if product_types_list := self.category_map.get(category.lower()):
                current_filters["product_type_name"] = product_types_list

            filter_expr = self._build_filter_expression(current_filters)

            print(
                f"🔍 Searching {category}: '{description}' with filter: {filter_expr}"
            )

            query_embedding = embed_text_query(
                model=self.search_service.model,
                processor=self.search_service.processor,
                text=description,
            )

            milvus_hits = self.db_client.search(
                vectors=[query_embedding], top_k=top_k, filter_expression=filter_expr
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
