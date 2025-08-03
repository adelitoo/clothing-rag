# src/fashion_search/agents/orchestrator.py
import traceback
from llama_index.llms.ollama import Ollama

from .planner import OutfitPlanner
from .executor import FashionSearchExecutor
from .formatter import ResultFormatter
from ..services.redis_search_service import RedisSearchService

class MultiFashionAgent:
    def __init__(self, search_service: RedisSearchService, llm: Ollama):
        self.planner = OutfitPlanner(llm)
        self.executor = FashionSearchExecutor(search_service)
        self.formatter = ResultFormatter()

    def process_query(self, query: str) -> str:
        try:
            print(f"🎯 Processing query: '{query}'")

            plan = self.planner.analyze_query(query)
            print(f"📋 Plan: {len(plan.categories)} categories, single_item={plan.is_single_item}")

            all_results = []
            for i, category in enumerate(plan.categories):
                description = plan.descriptions[i] if i < len(plan.descriptions) else f"{query} {category}"
                results = self.executor.search_category(category, description)
                all_results.extend(results)

            print(f"✅ Total results found: {len(all_results)}")

            formatted_response = self.formatter.format_results(all_results, query, plan.is_single_item)
            return formatted_response

        except Exception as e:
            print(f"❌ Multi-agent processing failed: {e}")
            traceback.print_exc()
            return f"I encountered an error while processing your request: {str(e)}"