import traceback
import json
from fastapi import APIRouter, HTTPException, Request

from ...agents.orchestrator import MultiFashionAgent
from ...redis_client.redis_db_client import RedisDBClient
from ...schemas.api_schemas import SearchRequest
from ...api.helpers import enrich_search_results

router = APIRouter(prefix="/agent", tags=["Agent Recommendations"])

@router.post("/recommend/")
def agent_recommendation(request: SearchRequest, http_request: Request):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        multi_agent: MultiFashionAgent = http_request.app.state.multi_fashion_agent
        redis_client: RedisDBClient = http_request.app.state.redis_client
    except AttributeError:
        raise HTTPException(status_code=503, detail="A required service is not available.")

    raw_cache_key = f"cache:agent:{request.query.strip().lower()}"
    if cached_response := redis_client.get_json(raw_cache_key):
        print(f"✅ Agent Cache HIT for raw query: '{request.query}'")
        return {**cached_response, "source": "agent_cache"}
    
    print(f"❌ Agent Cache MISS for raw query: '{request.query}'")

    try:
        agent_output = multi_agent.process_query(request)
        
        agent_response_dict = agent_output.get("response_payload", {})
        refined_query = agent_output.get("refined_query")

        if refined_query:
            refined_cache_key = f"cache:agent:{refined_query.strip().lower()}"
            if cached_response := redis_client.get_json(refined_cache_key):
                print(f"✅ Agent Cache HIT for refined query: '{refined_query}'")
                enriched_cached_articles = {}
                for category, articles in cached_response.get("categorized_articles", {}).items():
                    enriched_cached_articles[category] = enrich_search_results(articles, http_request)
                
                cached_response["categorized_articles"] = enriched_cached_articles
                return {**cached_response, "source": "agent_cache_refined"}

        summary_text = agent_response_dict.get("summary_text", "No summary available.")
        categorized_articles = agent_response_dict.get("categorized_articles", {})
        
        enriched_categorized_articles = {}
        for category, articles in categorized_articles.items():
            enriched_list_for_category = []
            if not articles: continue
            for article in articles:
                article_id = str(article.get("article_id")).zfill(10)
                score = article.get("relevance_score", 0.0)
                if item_data := redis_client.get_json(f"article:{article_id}"):
                    item_data["relevance_score"] = score
                    enriched_list_for_category.append(item_data)
                else:
                    print(f"⚠️ Data for article ID {article_id} not found in Redis.")
            if enriched_list_for_category:
                enriched_categorized_articles[category] = enrich_search_results(enriched_list_for_category, http_request)

        response_data = {
            "summary_text": summary_text,
            "categorized_articles": enriched_categorized_articles,
        }

        if enriched_categorized_articles and refined_query:
            final_cache_key = f"cache:agent:{refined_query.strip().lower()}"
            redis_client.set_json(final_cache_key, response_data, ttl=3600)
            print(f"✅ Cached result using refined key: '{refined_query}'")
        
        print("--- DEBUG: FINAL LIVE PAYLOAD ---")
        print(json.dumps(response_data, indent=2))
        print("---------------------------------")

        return {**response_data, "source": "multi_agent"}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Multi-agent system failed to process the query: {e}",
        )