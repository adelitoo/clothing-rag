import numpy as np
import pandas as pd
import config
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles

from models.pipeline_options import PipelineOptions
from models.search_request import SearchRequest
from preprocessing.cleanup_dataset import clean_csv
from captioning.captioning_pipeline import CaptioningPipeline
from embeddings.embedding_pipeline import EmbeddingPipeline
from embeddings.embedding_utils import embed_text_query
from llm.query_enhancer import LLMQueryEnhancer
from milvus.vector_db_client import VectorDBClient
from utils.model_utils import load_clip_model_and_processor
from utils.file_utils import get_image_path

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server starting up...")
    try:
        db_client = VectorDBClient(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        db_client.set_collection("articles", dim=config.EMB_DIM)
        app.state.db_client = db_client
        
        prompt_dir = Path(__file__).parent / "prompts"
        app.state.llm_enhancer = LLMQueryEnhancer(model=config.LLM_JUDGE_MODEL, prompt_dir=prompt_dir)

        model, processor = load_clip_model_and_processor()
        app.state.clip_model = model
        app.state.clip_processor = processor
        
        print("✅ Models, DB client, and LLM enhancer are loaded and ready.")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        traceback.print_exc()
    yield
    print("🔌 Server shutting down...")

app = FastAPI(lifespan=lifespan)
app.mount("/images", StaticFiles(directory=config.IMAGE_BASE_DIR), name="images")

def _enrich_search_results(hits: list[dict], request: Request) -> list[dict]:
    base_url = str(request.base_url).rstrip('/')
    for item in hits:
        path_obj = get_image_path(item["article_id"])
        if path_obj:
            relative_path = path_obj.relative_to(config.IMAGE_BASE_DIR)
            item["image_url"] = f"{base_url}/images/{relative_path.as_posix()}"
        else:
            item["image_url"] = None
    return hits

def _perform_search(query_text: str, top_k: int, request: Request) -> list[dict]:
    db_client: VectorDBClient = request.app.state.db_client
    model = request.app.state.clip_model
    processor = request.app.state.clip_processor
    query_embedding = embed_text_query(model, processor, query_text)
    hits = db_client.search(query_embedding, top_k=top_k)
    return _enrich_search_results(hits, request)

@app.get("/")
def root(): return {"status": "Backend running"}

@app.post("/pipeline/")
def run_configurable_pipeline(options: PipelineOptions, request: Request):
    results = {}
    print(f"Received pipeline request with options: {options.model_dump_json(indent=2)}")

    try:
        if options.run_cleanup:
            print("🚀 [1/4] Starting Dataset Cleanup...")
            _, count = clean_csv()
            results["cleanup"] = {"status": "OK", "articles_kept": count}
            print("✅ Cleanup complete.")

        if options.run_captioning:
            print("🚀 [2/4] Starting Image Captioning...")
            caption_pipeline = CaptioningPipeline(config)
            caption_results = caption_pipeline.run()
            results["captioning"] = {"status": "OK", **caption_results}
            print("✅ Captioning complete.")
            
        if options.run_embeddings:
            print("🚀 [3/4] Starting Text Embedding Generation...")
            embedding_pipeline = EmbeddingPipeline(config)
            embedding_results = embedding_pipeline.run()
            results["embeddings"] = {"status": "OK", **embedding_results}
            print("✅ Embedding generation complete.")

        if options.run_db_insertion:
            print("🚀 [4/4] Starting DB Insertion...")
            db_client: VectorDBClient = request.app.state.db_client
            
            data = np.load(config.EMBEDDING_SAVE_PATH, allow_pickle=False)
            embeddings, indices = data["embeddings"], data["indices"]
            
            df = pd.read_csv(config.COMPLETE_ARTICLES_CSV_PATH)
            if len(df) != len(indices):
                 print(f"⚠️ Warning: Mismatch between CSV rows ({len(df)}) and embedding indices ({len(indices)}). Using indices to slice.")
                 df = df.iloc[indices]

            article_ids = df["article_id"].tolist()
            
            db_client.set_collection("articles", dim=embeddings.shape[1], recreate=True)
            db_client.insert(article_ids, embeddings)
            db_client.create_index()
            results["db_insertion"] = {"status": "OK", "vectors_inserted": len(article_ids)}
            print("✅ DB Insertion complete.")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"A pipeline step failed: {e}")
        
    return {"status": "Pipeline run finished", "details": results}

@app.post("/search/")
def search_items(request: SearchRequest, http_request: Request):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")
    llm_enhancer: LLMQueryEnhancer = http_request.app.state.llm_enhancer
    transformed_query = llm_enhancer.transform(request.query)
    if not transformed_query:
        transformed_query = request.query
    summary = llm_enhancer.summarize(transformed_query)
    results = _perform_search(transformed_query, request.top_k, http_request)
    return {
        "original_query": request.query,
        "transformed_query": transformed_query,
        "summary": summary,
        "results": results
    }

@app.post("/search/baseline/")
def search_items_baseline(request: SearchRequest, http_request: Request):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")
    results = _perform_search(request.query, request.top_k, http_request)
    return {
        "original_query": request.query,
        "results": results,
    }