# src/fashion_search/agents/formatter.py
from typing import List
from ..schemas.agent_schemas import SearchResult

class ResultFormatter:
    def format_results(self, all_results: List[SearchResult], original_query: str, is_single_item: bool) -> str:
        if not all_results:
            return "I couldn't find any items matching your request. Please try a different search term."

        if is_single_item:
            top_results = sorted(all_results, key=lambda x: x.score, reverse=True)[:8]
            response = f"Here are some great options for '{original_query}':\n\n"
            for result in top_results:
                response += f"- Article ID: {result.article_id} (Relevance: {result.score:.2f})\n"
        else:
            by_category = {}
            for result in all_results:
                by_category.setdefault(result.category, []).append(result)

            response = f"Here's a complete outfit recommendation for '{original_query}':\n\n"
            for category, results in by_category.items():
                top_results = sorted(results, key=lambda x: x.score, reverse=True)[:4]
                response += f"🏷️ {category.upper()}:\n"
                for result in top_results:
                    response += f"- Article ID: {result.article_id} (Relevance: {result.score:.2f})\n"
                response += "\n"

        return response.strip()