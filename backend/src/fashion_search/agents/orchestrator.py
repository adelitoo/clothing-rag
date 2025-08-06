import traceback
from llama_index.llms.ollama import Ollama

from .planner import OutfitPlanner
from .executor import FashionSearchExecutor
from .formatter import ResultFormatter
from .filterer import FilterAgent 
from ..services.redis_search_service import RedisSearchService
from ..core.config import settings
from .filterer import FilterAgent

class MultiFashionAgent:
    def __init__(self, search_service: RedisSearchService, llm: Ollama):
        self.planner = OutfitPlanner(llm)
        self.executor = FashionSearchExecutor(search_service)
        self.formatter = ResultFormatter(llm)
        self.filterer = FilterAgent(llm)

    def process_query(self, query: str):
        try:
            print(f"🎯 Processing query: '{query}'")

            plan = self.planner.analyze_query(query)
            
            extracted_filters_raw = self.filterer.extract_filters(query)
            
            plan.filters = extracted_filters_raw.get("filters", extracted_filters_raw)
            
            print(f"📋 Plan: {len(plan.categories)} categories, single_item={plan.is_single_item}, filters={plan.filters}")

            all_results = []
            for i, category in enumerate(plan.categories):
                description = plan.descriptions[i] if i < len(plan.descriptions) else f"{query} {category}"
                results = self.executor.search_category(plan, category, description)
                all_results.extend(results)

            print(f"✅ Total results found: {len(all_results)}")
            formatted_response_obj = self.formatter.format_results(all_results, query)
            return formatted_response_obj.model_dump()

        except Exception as e:
            print(f"❌ Multi-agent processing failed: {e}")
            traceback.print_exc()
            return {"error": f"I encountered an error: {str(e)}"}