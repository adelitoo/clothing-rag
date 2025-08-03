import config
import re
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles

from typing import Dict, Any
from llama_index.core.tools import FunctionTool

# ✅ This is the correct import path based on the new library structure
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama

from models.search_request import SearchRequest
from llm.query_enhancer import LLMQueryEnhancer
from milvus.vector_db_client import VectorDBClient
from utils.model_utils import load_clip_model_and_processor
from redis_client.redis_db_client import RedisDBClient
from services.redis_search_service import RedisSearchService
from utils.response_utils import enrich_search_results


PROMPTS_DIR = Path(__file__).parent / "prompts"


def create_fashion_agent(
    search_service_instance: RedisSearchService, llm: Ollama
) -> ReActAgent:
    """Initializes and returns a ReActAgent for fashion recommendations."""
    print("🤖 Initializing fashion agent...")

    def search_clothing_item(category: str, description: str) -> str:
        """Searches for a single clothing item in a specific category."""
        print(
            f"\n👕 TOOL CALLED: Searching for Category='{category}', Description='{description}'"
        )
        try:
            # Use the regular search method instead of search_baseline for richer data
            results = search_service_instance.search_baseline(
                query=description, top_k=3
            )
            milvus_hits = results.get("milvus_results")

            if not milvus_hits:
                return f"No items found for category '{category}' with description '{description}'."

            formatted_items = []
            for hit in milvus_hits:
                article_id = hit.get("article_id", "Unknown")
                score = hit.get("score", 0.0)
                formatted_items.append(
                    f"  - Article ID: {article_id} (Relevance: {score:.2f})"
                )

            response = (
                f"Found {len(formatted_items)} {category} items matching '{description}':\n"
                + "\n".join(formatted_items)
            )
            return response
        except Exception as e:
            print(f"❌ Error in search_clothing_item: {e}")
            return f"Error searching for {category}: {str(e)}"

    clothing_tool = FunctionTool.from_defaults(
        fn=search_clothing_item,
        name="fashion_item_search",
        description="Use this tool to search for a specific category of clothing (e.g., 'pants', 'jeans', 'shirt', 'shoes') based on a detailed description. Always specify both category and description.",
    )

    # This works with version 0.11.23
    agent = ReActAgent.from_tools(tools=[clothing_tool], llm=llm, verbose=True)

    print("✅ Agent is ready.")
    return agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server starting up...")
    try:
        db_client = VectorDBClient(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        db_client.set_collection("articles", dim=config.EMB_DIM)
        app.state.db_client = db_client

        app.state.llm_enhancer = LLMQueryEnhancer(
            model=config.LLM_JUDGE_MODEL, prompt_dir=PROMPTS_DIR
        )

        app.state.redis_client = RedisDBClient(
            host=config.REDIS_HOST, port=config.REDIS_PORT
        )

        model, processor = load_clip_model_and_processor()
        app.state.clip_model = model
        app.state.clip_processor = processor

        app.state.search_service = RedisSearchService(
            redis_client=app.state.redis_client,
            db_client=app.state.db_client,
            llm_enhancer=app.state.llm_enhancer,
            model=app.state.clip_model,
            processor=app.state.clip_processor,
        )

        # Create Ollama instance with compatible settings
        llm_for_agent = Ollama(
            model=config.LLM_JUDGE_MODEL,
            request_timeout=120.0,
            # Disable features that might cause compatibility issues
            temperature=0.1,
        )
        app.state.fashion_agent = create_fashion_agent(
            app.state.search_service, llm_for_agent
        )

        print("✅ All services are loaded and ready.")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        traceback.print_exc()
    yield
    print("🔌 Server shutting down...")


app = FastAPI(lifespan=lifespan)
app.mount("/images", StaticFiles(directory=config.IMAGE_BASE_DIR), name="images")


@app.get("/")
def root():
    return {"status": "Backend running"}


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "agent": hasattr(app.state, "fashion_agent")
            and app.state.fashion_agent is not None
        },
    }


@app.post("/search/")
def search(request: SearchRequest, http_request: Request):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    search_service: RedisSearchService = http_request.app.state.search_service
    search_data = search_service.search(request.query, request.top_k)
    final_results = enrich_search_results(search_data["milvus_results"], http_request)

    return {
        "original_query": request.query,
        "transformed_query": search_data["transformed_query"],
        "summary": search_data["summary"],
        "results": final_results,
        "source": search_data.get("source", "live"),
    }


@app.post("/search/baseline/")
def search_baseline(request: SearchRequest, http_request: Request):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    search_service: RedisSearchService = http_request.app.state.search_service
    search_data = search_service.search_baseline(request.query, request.top_k)
    final_results = enrich_search_results(search_data["milvus_results"], http_request)

    return {
        "original_query": request.query,
        "results": final_results,
    }


@app.post("/agent/recommend/")
def agent_recommendation(request: SearchRequest, http_request: Request):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # --- Step 1: Get the agent, and fail early if it's missing. ---
    try:
        fashion_agent: ReActAgent = http_request.app.state.fashion_agent
        search_service: RedisSearchService = http_request.app.state.search_service
    except AttributeError:
        # This now correctly reports that a required service is not available.
        raise HTTPException(
            status_code=503,
            detail="A required service is not available. Check server startup logs.",
        )

    # --- Step 2: Run the main logic in a separate try block. ---
    try:
        agent_response = fashion_agent.chat(request.query)
        agent_summary_text = str(agent_response)

        article_ids = re.findall(r"(?:Article ID|Item ID):\s*(\d+)", agent_summary_text)

        final_results = []
        if article_ids:
            milvus_results = []
            for article_id in article_ids:
                item_data = search_service.redis_client.get_json(
                    f"article:{article_id}"
                )
                if item_data:
                    # Create a dictionary that mimics the structure enrich_search_results expects
                    milvus_results.append(
                        {"article_id": item_data["article_id"], "score": 0.99}
                    )

            final_results = enrich_search_results(milvus_results, http_request)

        return {
            "summary": agent_summary_text,
            "results": final_results,
            "source": "agent",
        }
    # --- Step 3: Catch all other errors and report them accurately. ---
    except Exception as e:
        print(f"Agent processing error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail="Agent failed to process the query after running."
        )
