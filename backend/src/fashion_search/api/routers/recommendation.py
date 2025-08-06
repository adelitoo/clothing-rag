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

    cache_key = f"cache:agent:{request.query.strip().lower()}"
    if cached_response := redis_client.get_json(cache_key):
        print(f"✅ Agent Cache HIT for query: '{request.query}'")
        return {**cached_response, "source": "agent_cache"}
    
    print(f"❌ Agent Cache MISS for query: '{request.query}'")

    try:
        agent_response_dict = multi_agent.process_query(request.query)

        summary_text = agent_response_dict.get("summary_text", "No summary available.")
        recommended_articles = agent_response_dict.get("recommended_articles", [])

        final_results = []
        if recommended_articles:
            results_with_details = []
            for article in recommended_articles:
                article_id = str(article.get("article_id")).zfill(10)
                score = article.get("relevance_score", 0.0)
                
                if item_data := redis_client.get_json(f"article:{article_id}"):
                    item_data["score"] = score
                    results_with_details.append(item_data)
                else:
                    print(f"⚠️ Data for article ID {article_id} not found in Redis.")

            final_results = enrich_search_results(results_with_details, http_request)

        response_data = {
            "summary": summary_text,
            "results": final_results,
            "total_items_found": len(final_results),
        }

        if final_results:
            redis_client.set_json(cache_key, response_data, ttl=3600)
        
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