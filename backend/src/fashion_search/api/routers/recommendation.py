import traceback
import json 
from fastapi import APIRouter, HTTPException, Request

from ...agents.orchestrator import MultiFashionAgent
from ...redis_client.redis_db_client import RedisDBClient
from ...schemas.api_schemas import SearchRequest
from ..helpers.helpers import enrich_search_results, extract_article_ids_robust

router = APIRouter(prefix="/agent", tags=["Agent Recommendations"])

@router.post("/recommend/")
def agent_recommendation(request: SearchRequest, http_request: Request):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        multi_agent: MultiFashionAgent = http_request.app.state.multi_fashion_agent
        redis_client: RedisDBClient = http_request.app.state.redis_client
    except AttributeError:
        raise HTTPException(
            status_code=503, detail="A required service is not available."
        )

    cache_key = f"cache:multi_agent:{request.query.strip().lower()}"
    if cached_response := redis_client.get_json(cache_key):
        print(f"✅ Multi-Agent Cache HIT for query: '{request.query}'")
        cached_response.setdefault("results", [])

        print("--- DEBUG: FINAL CACHED PAYLOAD ---")
        print(json.dumps(cached_response, indent=2))
        print("-----------------------------------")

        return {**cached_response, "source": "multi_agent_cache"}

    print(f"❌ Multi-Agent Cache MISS for query: '{request.query}'")

    try:
        agent_summary_text = multi_agent.process_query(request.query)
        found_items = extract_article_ids_robust(agent_summary_text)

        final_results = []
        if found_items:
            results_with_details = []
            for article_id_str, score_str in found_items:
                if item_data := redis_client.get_json(f"article:{article_id_str}"):
                    try:
                        item_data["score"] = float(score_str)
                    except (ValueError, TypeError):
                        item_data["score"] = 0.0
                    results_with_details.append(item_data)
                else:
                    print(f"⚠️ Data for article ID {article_id_str} not found in Redis.")
            final_results = enrich_search_results(results_with_details, http_request)

        response_data = {
            "summary": agent_summary_text,
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
            status_code=500, detail=f"Agent failed to process query: {e}"
        )
