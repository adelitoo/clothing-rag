from typing import List
from ..schemas.agent_schemas import SearchResult
from ..services.redis_search_service import RedisSearchService

class FashionSearchExecutor:
    def __init__(self, search_service: RedisSearchService):
        self.search_service = search_service

    def search_category(self, category: str, description: str, top_k: int = 12) -> List[SearchResult]:
        try:
            print(f"🔍 Searching {category}: '{description}'")
            combined_query = f"{description} {category}"
            results = self.search_service.search_baseline(query=combined_query, top_k=top_k)
            milvus_hits = results.get("milvus_results", [])

            search_results = []
            for hit in milvus_hits:
                search_results.append(
                    SearchResult(
                        article_id=hit.get("article_id", "Unknown"),
                        score=hit.get("score", 0.0),
                        category=category,
                    )
                )

            print(f"✅ Found {len(search_results)} {category} items")
            return search_results

        except Exception as e:
            print(f"❌ Search failed for {category}: {e}")
            return []