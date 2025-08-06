import instructor
from openai import OpenAI
from typing import List
from ..schemas.agent_schemas import SearchResult, FormattedResponse
from ..core.config import settings

class ResultFormatter:
    def __init__(self, llm):
        openai_client = OpenAI(
            base_url="http://localhost:11434/v1",  
            api_key="ollama"  
        )
        
        self.client = instructor.from_openai(openai_client)
        
        self.model_name = getattr(llm, 'model', 'llama3.2') 
        
        prompt_path = settings.PROMPTS_DIR / "result_formatter_prompt.txt"
        self.prompt_template = prompt_path.read_text()

    def format_results(self, all_results: List[SearchResult], original_query: str, is_single_item: bool = False) -> FormattedResponse:
        if not all_results:
            return FormattedResponse(
                summary_text="I couldn't find any items matching your request. Please try a different search term.",
                recommended_articles=[]
            )

        found_items_text = "\n".join(
            [f"- Article ID: {res.article_id} (Relevance: {res.score:.2f})" for res in all_results]
        )
        
        prompt = self.prompt_template.format(found_items_text=found_items_text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                response_model=FormattedResponse,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1, 
            )
            return response
        except Exception as e:
            print(f"❌ Error formatting results with instructor: {e}")
            return self._fallback_format(all_results, original_query)
    
    def _fallback_format(self, all_results: List[SearchResult], original_query: str) -> FormattedResponse:
        from ..schemas.agent_schemas import RecommendedArticle
        
        summary_lines = [f"Here are the items I found for '{original_query}':\n"]
        recommended_articles = []
        
        for result in all_results:
            summary_lines.append(f"- Article ID: {result.article_id} (Relevance: {result.score:.2f})")
            recommended_articles.append(RecommendedArticle(
                article_id=result.article_id,
                relevance_score=result.score
            ))
        
        return FormattedResponse(
            summary_text="\n".join(summary_lines),
            recommended_articles=recommended_articles
        )