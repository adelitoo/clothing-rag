import config
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles

from models.pipeline_options import PipelineOptions
from models.search_request import SearchRequest
from pipeline.steps import CleanupStep, CaptioningStep, EmbeddingStep, DbInsertionStep
from embeddings.embedding_utils import embed_text_query
from llm.query_enhancer import LLMQueryEnhancer
from milvus.vector_db_client import VectorDBClient
from utils.model_utils import load_clip_model_and_processor
from redis_client.redis_db_client import RedisDBClient
from services.search_service import SearchService
from utils.response_utils import enrich_search_results

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server starting up...")
    try:
        db_client = VectorDBClient(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        db_client.set_collection("articles", dim=config.EMB_DIM)
        app.state.db_client = db_client
        
        prompt_dir = Path(__file__).parent / "prompts"
        app.state.llm_enhancer = LLMQueryEnhancer(model=config.LLM_JUDGE_MODEL, prompt_dir=prompt_dir)

        app.state.redis_client = RedisDBClient(host=config.REDIS_HOST, port=config.REDIS_PORT)

        model, processor = load_clip_model_and_processor()
        app.state.clip_model = model
        app.state.clip_processor = processor
        
        app.state.search_service = SearchService(
            redis_client=app.state.redis_client,
            db_client=app.state.db_client,
            llm_enhancer=app.state.llm_enhancer,
            model=app.state.clip_model,
            processor=app.state.clip_processor
        )
        
        print("✅ Models, DB clients, and LLM enhancer are loaded and ready.")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        traceback.print_exc()
    yield
    print("🔌 Server shutting down...")

app = FastAPI(lifespan=lifespan)
app.mount("/images", StaticFiles(directory=config.IMAGE_BASE_DIR), name="images")

@app.get("/")
def root(): return {"status": "Backend running"}


@app.post("/pipeline/")
def run_configurable_pipeline(options: PipelineOptions, request: Request):
    print(f"Received pipeline request with options: {options.model_dump_json(indent=2)}")
    
    try:
        db_client: VectorDBClient = request.app.state.db_client
    except AttributeError:
        raise HTTPException(status_code=503, detail="Database client not available.")

    step_map = {
        "run_cleanup": ("cleanup", CleanupStep()),
        "run_captioning": ("captioning", CaptioningStep()),
        "run_embeddings": ("embeddings", EmbeddingStep()),
        "run_db_insertion": ("db_insertion", DbInsertionStep(db_client=db_client)),
    }

    pipeline_to_run = [
        (result_key, step)
        for option, (result_key, step) in step_map.items()
        if getattr(options, option)
    ]

    if not pipeline_to_run:
        return {"message": "No pipeline steps were selected to run."}

    results = {}
    try:
        for result_key, step in pipeline_to_run:
            step_result = step.run()
            results[result_key] = step_result
            
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"A pipeline step failed: {str(e)}")

    return results



@app.post("/search/")
def search_items(request: SearchRequest, http_request: Request):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")
        
    search_service: SearchService = http_request.app.state.search_service
    
    search_data = search_service.search(request.query, request.top_k)
    
    final_results = enrich_search_results(search_data["milvus_results"], http_request)

    return {
        "original_query": request.query,
        "transformed_query": search_data["transformed_query"],
        "summary": search_data["summary"],
        "results": final_results,
        "source": search_data.get("source", "live") 
    }

@app.post("/search/baseline/")
def search_items_baseline(request: SearchRequest, http_request: Request):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    search_service: SearchService = http_request.app.state.search_service
    
    search_data = search_service.search_baseline(request.query, request.top_k)
    
    final_results = enrich_search_results(search_data["milvus_results"], http_request)
    
    return {
        "original_query": request.query,
        "results": final_results,
    }