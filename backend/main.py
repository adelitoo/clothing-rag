import os
import sys
import traceback
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from pymilvus import Collection, utility
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from preprocessing.cleanup_dataset import clean_csv
from captioning.generate_img_captions import generate_and_save_captions
from embeddings.embedding_pipeline import run_embedding_pipeline
from embeddings.embedding_utils import embed_text_query
from milvus.insert_to_milvus import connect_milvus, create_collection, insert_embeddings
from models.pipeline_options import PipelineOptions
from models.search_request import SearchRequest
from utils.model_utils import load_clip_model_and_processor
from milvus.insert_to_milvus import search_similar_items
from llm.query_transformer import transform_query_with_ollama, summarize_transformed_query
from config import EMBEDDING_SAVE_PATH, COMPLETE_CSV_PATH, EMB_DIM

sys.stdout.reconfigure(line_buffering=True)
os.system('cls' if os.name == 'nt' else 'clear')   



@asynccontextmanager
async def startup(app: FastAPI):
    
    model, processor = load_clip_model_and_processor()
    print("✅ CLIP model and processor loaded")
    
    yield

    print("🚀 Starting up, connecting to Milvus and loading models...")

    try:
        connect_milvus()
        collection_name = "articles"

        embedding_type = "text_only"  
        embedding_dim = EMB_DIM  
        
        if os.path.exists(EMBEDDING_SAVE_PATH):
            data = np.load(EMBEDDING_SAVE_PATH, allow_pickle=False)
            embeddings = data["embeddings"]
            embedding_dim = embeddings.shape[1]
            
            if "embedding_type" in data:
                embedding_type = str(data["embedding_type"])
                print(f"📊 Found {embedding_type} embeddings with dimension {embedding_dim}")
            else:
                print(f"📊 Found embeddings with dimension {embedding_dim}")
                if embedding_dim != EMB_DIM:
                    print("⚠️ Warning: wrong embeddings dimension detected")
                    if utility.has_collection(collection_name):
                        utility.drop_collection(collection_name)
                        print("🗑️ Dropped existing collection")
                        
                        collection = create_collection(name=collection_name, dim=embedding_dim)
                        collection.load()
                        if os.path.exists(EMBEDDING_SAVE_PATH):
                            data = np.load(EMBEDDING_SAVE_PATH, allow_pickle=False)
                            embeddings = data["embeddings"]
                            success_idx = data["indices"].tolist()

                            df = pd.read_csv(COMPLETE_CSV_PATH)
                            article_ids = df.loc[success_idx, "article_id"].tolist()

                            if len(article_ids) != embeddings.shape[0]:
                                raise RuntimeError(f"ID/embedding count mismatch: {len(article_ids)} IDs vs {embeddings.shape[0]} vectors")

                            insert_embeddings(collection, article_ids, embeddings)
                            print(f"✅ Inserted {len(article_ids)} embeddings into Milvus collection")
                        else:
                            print("⚠️ No embeddings found - run the pipeline first")

        app.state.milvus_collection = collection
        app.state.model = model
        app.state.processor = processor
        app.state.embedding_dim = embedding_dim
        app.state.embedding_type = embedding_type

    except Exception as e:
        print(f"❌ Startup failed: {e}")
        traceback.print_exc()
        
        
app = FastAPI(lifespan=startup)
app.mount("/images", StaticFiles(directory="data/images"), name="images")

@app.get("/")
def root():
    return {"status": "Backend running"}


@app.post("/pipeline/")
def full_pipeline(options: PipelineOptions):
    result = {}

    if options.run_cleaning:
        try:
            count = clean_csv()
            result["cleaning"] = f"OK ({count} rows)"
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="CSV not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cleaning failed: {e}")
    else:
        result["cleaning"] = "skipped"

    if options.run_captioning:
        try:
            generate_and_save_captions()
            result["captioning"] = "OK"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Captioning failed: {e}")
    else:
        result["captioning"] = "skipped"

    if options.run_embedding:
        try:
            run_embedding_pipeline()
            result["embedding"] = "OK"
                        
            data = np.load(EMBEDDING_SAVE_PATH, allow_pickle=False)
            new_dim = data["embeddings"].shape[1]
            embedding_type = data.get("embedding_type", "unknown")
            
            app.state.embedding_dim = new_dim
            app.state.embedding_type = str(embedding_type)
            
            print(f"📊 New embeddings: {embedding_type}, dimension: {new_dim}")
            
        except Exception as e:
            print(f"❌ Embedding pipeline error: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
    else:
        result["embedding"] = "skipped"

    if options.run_db_insertion:
        try:
            df = pd.read_csv(COMPLETE_CSV_PATH)

            embedding_dim = getattr(app.state, "embedding_dim", EMB_DIM)
            
            if os.path.exists(EMBEDDING_SAVE_PATH):
                data = np.load(EMBEDDING_SAVE_PATH, allow_pickle=False)
                actual_dim = data["embeddings"].shape[1]
                if actual_dim != embedding_dim:
                    print(f"📊 Updating dimension from {embedding_dim} to {actual_dim}")
                    embedding_dim = actual_dim

            connect_milvus()
            collection_name = "articles"
            
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                print("🗑️ Dropped existing collection for dimension update")

            collection = create_collection(name=collection_name, dim=embedding_dim)
            collection.load()
            app.state.milvus_collection = collection
            app.state.embedding_dim = embedding_dim

            data = np.load(EMBEDDING_SAVE_PATH, allow_pickle=False)
            embeddings = data["embeddings"]
            success_idx = data["indices"].tolist()

            article_ids = df.loc[success_idx, "article_id"].tolist()
            if len(article_ids) != embeddings.shape[0]:
                raise HTTPException(
                    status_code=500,
                    detail=f"ID/embedding count mismatch: {len(article_ids)} IDs vs {embeddings.shape[0]} vectors"
                )

            insert_embeddings(collection, article_ids, embeddings)
            result["db_insertion"] = "OK"
            print(f"✅ Inserted {len(article_ids)} embeddings (dim: {embedding_dim})")

        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"DB insertion failed: {e}")
    else:
        result["db_insertion"] = "skipped"

    return {"status": "complete", "details": result}


@app.post("/search/")
def search_items(request: SearchRequest, http_request: Request):
    try:
        model = getattr(app.state, "model", None)
        processor = getattr(app.state, "processor", None)
        if model is None or processor is None:
            model, processor = load_clip_model_and_processor()
            app.state.model = model
            app.state.processor = processor

        print(f"[QUERY] Original: {request.query}")
        transformed_query = transform_query_with_ollama(request.query)
        print(f"[QUERY] Transformed: {transformed_query}")
        
        summary = summarize_transformed_query(transformed_query)

        query_embedding = embed_text_query(model, processor, transformed_query)
        
        # Debug: Check embedding dimension
        embedding_dim = len(query_embedding)
        expected_dim = getattr(app.state, "embedding_dim", 512)
        
        if embedding_dim != expected_dim:
            print(f"⚠️ Warning: Query embedding dim ({embedding_dim}) != stored dim ({expected_dim})")

        collection = getattr(app.state, "milvus_collection", None)
        if collection is None:
            connect_milvus()
            collection = Collection("articles")
            collection.load()
            app.state.milvus_collection = collection

        search_results = search_similar_items(collection, query_embedding, top_k=request.top_k)

        base_url = str(http_request.base_url)
        for item in search_results:
            if item["image"] is not None:
                url_path = "/" + item["image"].replace("data/images", "images").replace("\\", "/")
                item["image"] = base_url.rstrip("/") + url_path

        # Debug info
        print(f"[SEARCH] Found {len(search_results)} results")
        if search_results:
            print(f"[SEARCH] Top score: {search_results[0]['score']:.4f}")

        return {
            "original_query": request.query,
            "transformed_query": transformed_query,  
            "summary": summary,
            "results": search_results,
            "debug": {
                "query_embedding_dim": embedding_dim,
                "stored_embedding_dim": expected_dim,
                "embedding_type": getattr(app.state, "embedding_type", "unknown")
            }
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")
    
# In your FastAPI application file (e.g., main.py)
# Add this new endpoint alongside your existing /search/ endpoint.

@app.post("/search/baseline/")
def search_items_baseline(request: SearchRequest, http_request: Request):
    """
    Baseline search endpoint that uses the original query without LLM transformation.
    This serves as SYSTEM B in our A/B test.
    """
    try:
        model = getattr(app.state, "model", None)
        processor = getattr(app.state, "processor", None)
        if model is None or processor is None:
            model, processor = load_clip_model_and_processor()
            app.state.model = model
            app.state.processor = processor

        # --- Key Difference: Use the original query directly ---
        original_query = request.query
        print(f"[BASELINE QUERY] Original: {original_query}")

        # --- Embed the original query ---
        query_embedding = embed_text_query(model, processor, original_query)
        
        collection = getattr(app.state, "milvus_collection", None)
        if collection is None:
            connect_milvus()
            collection = Collection("articles")
            collection.load()
            app.state.milvus_collection = collection

        search_results = search_similar_items(collection, query_embedding, top_k=request.top_k)

        base_url = str(http_request.base_url)
        for item in search_results:
            if item["image"] is not None:
                url_path = "/" + item["image"].replace("data/images", "images").replace("\\", "/")
                item["image"] = base_url.rstrip("/") + url_path

        print(f"[BASELINE SEARCH] Found {len(search_results)} results")

        return {
            "original_query": original_query,
            "results": search_results,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Baseline search failed: {e}")

@app.get("/debug/status")
def debug_status():
    try:
        embedding_info = {}
        if os.path.exists(EMBEDDING_SAVE_PATH):
            data = np.load(EMBEDDING_SAVE_PATH, allow_pickle=False)
            embedding_info = {
                "embeddings_shape": data["embeddings"].shape,
                "indices_count": len(data["indices"]),
                "embedding_type": str(data.get("embedding_type", "unknown"))
            }
        
        return {
            "embeddings_file_exists": os.path.exists(EMBEDDING_SAVE_PATH),
            "complete_csv_exists": os.path.exists(COMPLETE_CSV_PATH),
            "embedding_info": embedding_info,
            "app_state": {
                "has_model": hasattr(app.state, "model"),
                "has_collection": hasattr(app.state, "milvus_collection"),
                "embedding_dim": getattr(app.state, "embedding_dim", "not_set"),
                "embedding_type": getattr(app.state, "embedding_type", "not_set")
            }
        }
    except Exception as e:
        return {"error": str(e)}