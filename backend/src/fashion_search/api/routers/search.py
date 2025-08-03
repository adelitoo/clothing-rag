from fastapi import APIRouter, Request, HTTPException
from ...schemas.api_schemas import SearchRequest
from ...services.redis_search_service import RedisSearchService
from ..helpers.helpers import enrich_search_results

router = APIRouter(prefix="/search", tags=["Standard Search"])


@router.post("/")
def search_items(request: SearchRequest, http_request: Request):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")
        
    try:
        search_service: RedisSearchService = http_request.app.state.search_service
    except AttributeError:
        raise HTTPException(status_code=503, detail="Search service not available.")
    
    search_data = search_service.search(request.query, request.top_k)
    
    final_results = enrich_search_results(search_data.get("milvus_results", []), http_request)

    return {
        "original_query": request.query,
        "transformed_query": search_data.get("transformed_query"),
        "summary": search_data.get("summary"),
        "results": final_results,
        "source": search_data.get("source", "live") 
    }


@router.post("/baseline/")
def search_items_baseline(request: SearchRequest, http_request: Request):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    try:
        search_service: RedisSearchService = http_request.app.state.search_service
    except AttributeError:
        raise HTTPException(status_code=503, detail="Search service not available.")

    search_data = search_service.search_baseline(request.query, request.top_k)
    
    final_results = enrich_search_results(search_data.get("milvus_results", []), http_request)
    
    return {
        "original_query": request.query,
        "results": final_results,
    }