import instructor
from openai import OpenAI
from typing import List, Dict
from ..schemas.agent_schemas import FormattedResponse, RecommendedArticle, SearchResult
from ..core.config import settings

class ResultFormatter:
    def __init__(self, llm):
        openai_client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        self.client = instructor.from_openai(openai_client)
        self.model_name = getattr(llm, 'model', settings.LLM_JUDGE_MODEL)

    def format_results(self, categorized_results: Dict[str, List[SearchResult]], original_query: str, is_single_item: bool = False) -> FormattedResponse:
        print("✅ Structuring results with deterministic formatter...")
        if not categorized_results:
            return FormattedResponse(
                summary_text="I couldn't find any items matching your request. Please try a different search term.",
                categorized_articles={}
            )
        
        return self._fallback_format(categorized_results, original_query)

    def _fallback_format(self, categorized_results: Dict[str, List[SearchResult]], original_query: str) -> FormattedResponse:
        category_names = [name.capitalize() for name in categorized_results.keys()]
        summary_text = f"For your outfit '{original_query}', I found a selection of items including: {', '.join(category_names)}. Explore the curated collections below!"
        
        categorized_articles: Dict[str, List[RecommendedArticle]] = {}
        
        for category, results in categorized_results.items():
            if category not in categorized_articles:
                categorized_articles[category] = []
            
            for result in results:
                categorized_articles[category].append(RecommendedArticle(
                    article_id=result.article_id,
                    relevance_score=round(result.score, 4)
                ))
        
        return FormattedResponse(
            summary_text=summary_text,
            categorized_articles=categorized_articles
        )