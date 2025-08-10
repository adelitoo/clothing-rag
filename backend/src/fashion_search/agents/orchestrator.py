import traceback
from llama_index.llms.ollama import Ollama
from typing import Dict, List  

from .planner import OutfitPlanner
from .executor import FashionSearchExecutor
from .formatter import ResultFormatter
from .filterer import FilterAgent
from ..services.redis_search_service import RedisSearchService
from ..memory.conversation_manager import ConversationManager
from .refiner import QueryRefinementAgent
from ..schemas.api_schemas import SearchRequest 
from ..schemas.agent_schemas import SearchResult, OutfitPlan


class MultiFashionAgent:
    def __init__(self, search_service: RedisSearchService, llm: Ollama, conversation_manager: ConversationManager, query_refiner: QueryRefinementAgent):
        self.planner = OutfitPlanner(llm)
        self.executor = FashionSearchExecutor(search_service)
        self.formatter = ResultFormatter(llm)
        self.filterer = FilterAgent(llm)
        self.conversation_manager = conversation_manager 
        self.query_refiner = query_refiner 

    def process_query(self, request: SearchRequest):
        try:
            original_query = request.query
            session_id = request.session_id

            if session_id:
                history = self.conversation_manager.get_history(session_id)
                refined_query = self.query_refiner.refine(original_query, history)
            else:
                refined_query = original_query

            print(f"🎯 Processing refined query: '{refined_query}'")
            
            plan_cache_key = f"cache:plan:{refined_query.strip().lower()}"
            cached_plan = self.conversation_manager.redis.get_json(plan_cache_key)

            if cached_plan:
                plan = OutfitPlan(**cached_plan)
                print(f"✅ Planner Cache HIT for: '{refined_query}'")
            else:
                print(f"❌ Planner Cache MISS for: '{refined_query}'")
                plan = self.planner.analyze_query(refined_query)
                if plan.categories:
                    self.conversation_manager.redis.set_json(plan_cache_key, plan.__dict__, ttl=86400)
            
            filter_cache_key = f"cache:filter:{refined_query.strip().lower()}"
            cached_filters = self.conversation_manager.redis.get_json(filter_cache_key)
            
            if cached_filters is not None:
                base_filters = cached_filters
                print(f"✅ Filterer Cache HIT for: '{refined_query}'")
            else:
                print(f"❌ Filterer Cache MISS for: '{refined_query}'")
                extracted_filters_raw = self.filterer.extract_filters(refined_query)
                base_filters = extracted_filters_raw.get("filters", {})
                self.conversation_manager.redis.set_json(filter_cache_key, base_filters, ttl=86400)
            
            print(
                f"📋 Plan: {len(plan.categories)} categories, single_item={plan.is_single_item}, base_filters={base_filters}"
            )

            categorized_results: Dict[str, List[SearchResult]] = {}
            for i, category in enumerate(plan.categories):
                description = (
                    plan.descriptions[i]
                    if i < len(plan.descriptions)
                    else f"{refined_query} {category}"
                )
                if not plan.is_single_item:
                    active_filters = ({"index_name": base_filters.get("index_name")} if "index_name" in base_filters else {})
                else:
                    active_filters = base_filters

                plan.filters = active_filters
                results = self.executor.search_category(plan, category, description)
                if results:
                    categorized_results[category] = results

            formatted_response_obj = self.formatter.format_results(
                categorized_results, refined_query, plan.is_single_item
            )
            final_response = formatted_response_obj.model_dump()

            if session_id:
                summary = final_response.get("summary_text", "I found some items for you.")
                self.conversation_manager.add_turn(session_id, original_query, summary)
            
            return {
                "response_payload": final_response,
                "refined_query": refined_query 
            }        
        except Exception as e:
            print(f"❌ Multi-agent processing failed: {e}")
            traceback.print_exc()
            return {"error": f"I encountered an error: {str(e)}"}