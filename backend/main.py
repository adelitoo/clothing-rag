import config
import re
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles

from typing import Dict, Any, List
from llama_index.core.tools import FunctionTool
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

    # Store all search results across multiple tool calls
    search_results_storage = {"items": []}

    def search_clothing_item(category: str, description: str) -> str:
        """Searches for a single clothing item in a specific category."""
        print(
            f"\n👕 TOOL CALLED: Searching for Category='{category}', Description='{description}'"
        )
        try:
            combined_query = f"{description} {category}"
            print(f"  -> Combined query for embedding: '{combined_query}'")

            results = search_service_instance.search_baseline(
                query=combined_query, top_k=12
            )
            milvus_hits = results.get("milvus_results")

            if not milvus_hits:
                return f"No items found for category '{category}' with description '{description}'."

            # Store results for later use
            for hit in milvus_hits:
                article_id = hit.get("article_id", "Unknown")
                score = hit.get("score", 0.0)
                search_results_storage["items"].append(
                    {
                        "article_id": article_id,
                        "score": score,
                        "category": category,
                        "description": description,
                    }
                )

            formatted_items = []
            for hit in milvus_hits:
                article_id = hit.get("article_id", "Unknown")
                score = hit.get("score", 0.0)
                formatted_items.append(
                    f"  - Article ID: {article_id} (Relevance: {score:.2f})"
                )

            response = (
                f"✅ FOUND {len(formatted_items)} {category} items matching '{description}':\n"
                + "\n".join(formatted_items)
                + f"\n\n📋 TOTAL ITEMS COLLECTED SO FAR: {len(search_results_storage['items'])}"
                + "\n💡 NEXT: Continue searching for other clothing categories if needed for a complete outfit, "
                + "or provide your final recommendation if you have enough items."
            )
            return response
        except Exception as e:
            print(f"❌ Error in search_clothing_item: {e}")
            return f"Error searching for {category}: {str(e)}"

    def get_all_collected_items() -> str:
        """Returns all items collected from previous searches."""
        if not search_results_storage["items"]:
            return "No items have been collected yet. Use the search tool first."

        items_by_category = {}
        for item in search_results_storage["items"]:
            category = item["category"]
            if category not in items_by_category:
                items_by_category[category] = []
            items_by_category[category].append(item)

        result = f"📦 ALL COLLECTED ITEMS ({len(search_results_storage['items'])} total):\n\n"
        for category, items in items_by_category.items():
            result += f"🏷️ {category.upper()}:\n"
            for item in items:
                result += f"  - Article ID: {item['article_id']} (Relevance: {item['score']:.2f})\n"
            result += "\n"

        return result

    # Enhanced tool descriptions
    search_tool = FunctionTool.from_defaults(
        fn=search_clothing_item,
        name="search_clothing_item",
        description=(
            "Search for clothing items in a specific category. "
            "For complex outfit requests, call this tool multiple times for different categories "
            "(e.g., 'shirt', 'pants', 'shoes', 'jacket'). "
            "Each call adds items to your collection."
        ),
    )

    collection_tool = FunctionTool.from_defaults(
        fn=get_all_collected_items,
        name="get_collected_items",
        description=(
            "Get a summary of all clothing items collected from previous searches. "
            "Use this before providing your final outfit recommendation."
        ),
    )

    # Create agent with enhanced system prompt
    system_prompt = """You are a fashion recommendation expert. Your goal is to help users find clothing items or complete outfits.

FOR SIMPLE REQUESTS (single item like "black jeans"):
- Use search_clothing_item once
- Provide the results directly

FOR COMPLEX REQUESTS (complete outfits like "elegant dinner outfit"):
- Break down the outfit into categories (shirt, pants, shoes, etc.)
- Use search_clothing_item for each category
- Use get_collected_items to review all findings
- Provide a comprehensive recommendation with ALL article IDs

IMPORTANT RULES:
1. Always include ALL Article IDs in your final answer
2. For multi-step searches, continue until you have searched all necessary categories
3. Don't stop after just one search if the user wants a complete outfit
4. If you're building an outfit, search for at least 3-4 categories (top, bottom, shoes, accessories)

CRITICAL FORMATTING REQUIREMENT:
In your final answer, list each Article ID on a separate line like this:
- Article ID: 123456789 (Relevance: 0.85)
- Article ID: 987654321 (Relevance: 0.82)

NEVER group multiple IDs together on one line. Each ID must be on its own line with this exact format.
Your final answer should ALWAYS contain specific Article ID numbers, never just action plans."""

    agent = ReActAgent.from_tools(
        tools=[search_tool, collection_tool],
        llm=llm,
        verbose=True,
        system_prompt=system_prompt,
        max_iterations=40,  # Allow more iterations for complex queries
    )

    # Clear storage for each new conversation
    def reset_storage():
        search_results_storage["items"].clear()

    agent._reset_storage = reset_storage
    print("✅ Enhanced agent is ready.")
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

        llm_for_agent = Ollama(
            model=config.LLM_JUDGE_MODEL,
            request_timeout=120.0,
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


def extract_article_ids_robust(text: str) -> List[tuple]:
    """
    Enhanced regex parsing that handles multiple response formats.
    Returns list of (article_id, score) tuples.
    """
    found_items = []

    # Pattern 1: Standard format - "Article ID: 123 (Relevance: 0.85)"
    pattern1 = re.findall(
        r"Article\s+ID\s*:?\s*(\d+).*?Relevance\s*:?\s*([\d\.]+)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    found_items.extend(pattern1)

    # Pattern 2: Alternative format - "ID: 123, Score: 0.85"
    pattern2 = re.findall(
        r"ID\s*:?\s*(\d+).*?(?:Score|Relevance)\s*:?\s*([\d\.]+)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    found_items.extend(pattern2)

    # Pattern 3: Just article numbers with scores in parentheses
    pattern3 = re.findall(r"(\d{6,})\s*\([^\)]*?([\d\.]+)[^\)]*?\)", text)
    found_items.extend(pattern3)

    # Pattern 4: Article ID with any text in parentheses (like "Very relevant", "All quite relevant")
    pattern4 = re.findall(r"Article\s+ID\s*:?\s*(\d+)\s*\([^)]+\)", text, re.IGNORECASE)
    for article_id in pattern4:
        found_items.append((article_id, "1.0"))  # Default score

    # Pattern 5: Multiple article IDs in one line (comma-separated)
    # Example: "Article ID: 690108004, 540395010, 690449036, 690449022"
    pattern5 = re.findall(
        r"Article\s+ID\s*:?\s*((?:\d+(?:\s*,\s*\d+)*)+)", text, re.IGNORECASE
    )
    for ids_string in pattern5:
        # Split by comma and extract individual IDs
        individual_ids = re.findall(r"\d+", ids_string)
        for article_id in individual_ids:
            found_items.append((article_id, "1.0"))  # Default score

    # Pattern 6: Any sequence of 6+ digits that looks like an article ID
    # (fallback pattern for when formatting is inconsistent)
    pattern6 = re.findall(r"\b(\d{6,})\b", text)
    for article_id in pattern6:
        found_items.append((article_id, "0.8"))  # Lower default score for fallback

    # Remove duplicates while preserving order
    seen = set()
    unique_items = []
    for item_id, score in found_items:
        if item_id not in seen:
            seen.add(item_id)
            unique_items.append((item_id, score))

    print(f"🔍 Extracted {len(unique_items)} unique article IDs from agent response")
    if unique_items:
        print(f"🔍 First few IDs: {unique_items[:3]}")
    return unique_items


@app.post("/agent/recommend/")
def agent_recommendation(request: SearchRequest, http_request: Request):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        fashion_agent: ReActAgent = http_request.app.state.fashion_agent
        redis_client: RedisDBClient = http_request.app.state.redis_client
    except AttributeError:
        raise HTTPException(
            status_code=503,
            detail="A required service is not available. Check server startup logs.",
        )

    cache_key = f"cache:agent:{request.query.strip().lower()}"
    cached_response = redis_client.get_json(cache_key)
    if cached_response:
        print(f"✅ Agent Cache HIT for query: '{request.query}'")
        cached_response.setdefault("results", [])
        return {**cached_response, "source": "agent_cache"}

    print(f"❌ Agent Cache MISS for query: '{request.query}'")

    try:
        # Reset storage for new query
        if hasattr(fashion_agent, "_reset_storage"):
            fashion_agent._reset_storage()

        agent_response = fashion_agent.chat(request.query)
        agent_summary_text = str(agent_response)

        print(f"🤖 Agent response length: {len(agent_summary_text)}")
        print(f"🤖 Agent response preview: {agent_summary_text[:200]}...")

        # Enhanced parsing
        found_items = extract_article_ids_robust(agent_summary_text)

        final_results = []
        if not found_items:
            print(
                "⚠️ No article IDs found in agent response. Checking for error patterns..."
            )

            # Check if response contains action patterns (indicating the agent got stuck)
            if (
                "Action:" in agent_summary_text
                and "Action Input:" in agent_summary_text
            ):
                print(
                    "🚨 DETECTED: Agent got stuck and returned an action instead of results"
                )
                error_msg = (
                    "The agent planned a multi-step search but didn't complete it. "
                    "This might be due to the complexity of the query. "
                    "Try asking for specific items instead of complete outfits."
                )
                return {
                    "summary": error_msg,
                    "results": [],
                    "source": "agent",
                    "error": "agent_incomplete_execution",
                }

        # Process found items
        if found_items:
            results_with_details = []
            for article_id_str, score_str in found_items:
                item_data = redis_client.get_json(f"article:{article_id_str}")

                if item_data:
                    try:
                        score = float(score_str)
                    except (ValueError, TypeError):
                        score = 0.0

                    item_data["score"] = score
                    results_with_details.append(item_data)
                else:
                    print(f"⚠️ Data for article ID {article_id_str} not found in Redis.")

            final_results = enrich_search_results(results_with_details, http_request)

        response_data = {
            "summary": agent_summary_text,
            "results": final_results,
            "total_items_found": len(final_results),
        }

        # Only cache successful responses with results
        if final_results:
            redis_client.set_json(cache_key, response_data, ttl=3600)

        return {**response_data, "source": "agent"}

    except Exception as e:
        print(f"Agent processing error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Agent failed to process the query: {e}"
        )
