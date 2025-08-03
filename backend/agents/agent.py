import logging
import sys
import asyncio
from typing import Dict, Any

from services.redis_search_service import RedisSearchService
from redis_client.redis_db_client import RedisDBClient
from milvus.vector_db_client import VectorDBClient
from llm.query_enhancer import LLMQueryEnhancer
from utils.model_utils import load_clip_model_and_processor

from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
import config

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

print("🔄 Initializing services...")
redis_client = RedisDBClient(host=config.REDIS_HOST, port=config.REDIS_PORT)
db_client = VectorDBClient(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
db_client.set_collection(name="articles", dim=config.EMB_DIM)
llm_enhancer = LLMQueryEnhancer(
    model=config.LLM_JUDGE_MODEL, prompt_dir=config.PROMPTS_DIR
)
model, processor = load_clip_model_and_processor()

search_service = RedisSearchService(
    redis_client=redis_client,
    db_client=db_client,
    llm_enhancer=llm_enhancer,
    model=model,
    processor=processor,
)
print("✅ Services initialized.")


def search_clothing_item(category: str, description: str) -> str:
    print(
        f"\n👕 TOOL CALLED: Searching for Category='{category}', Description='{description}'"
    )
    results = search_service.search_baseline(query=description, top_k=3)
    return str(results)


clothing_tool = FunctionTool.from_defaults(
    fn=search_clothing_item,
    name="fashion_item_search",
    description=(
        "Use this tool to search for a specific category of clothing "
        "(e.g., 'pants', 'shirt', 'shoes') based on a detailed description."
    ),
)

print("🤖 Initializing agent...")
llm = Ollama(model="llama3.1:8b", request_timeout=120.0)

agent = None
try:
    agent = ReActAgent.from_tools([clothing_tool], llm=llm, verbose=True)
    print("✅ Agent created using from_tools method")
except Exception as e:
    print(f"❌ from_tools method failed: {e}")
    try:
        agent = ReActAgent(tools=[clothing_tool], llm=llm, verbose=True)
        print("✅ Agent created using direct instantiation")
    except Exception as e2:
        print(f"❌ Direct instantiation failed: {e2}")
        raise Exception("Could not create agent with either method")

print("✅ Agent is ready.")


async def run_agent_async(query: str):
    """Run the agent asynchronously"""
    try:
        result = await agent.run(query)
        return result
    except Exception as e:
        print(f"❌ Async run failed: {e}")
        try:
            if hasattr(agent, "chat"):
                return agent.chat(query)
            elif hasattr(agent, "query"):
                return agent.query(query)
            else:
                raise Exception("No working method found")
        except Exception as e2:
            print(f"❌ Sync methods also failed: {e2}")
            raise


def run_agent_sync(query: str):
    try:
        if hasattr(agent, "chat"):
            return agent.chat(query)
        elif hasattr(agent, "query"):
            return agent.query(query)
        else:
            # Fall back to async in sync context
            return asyncio.run(agent.run(query))
    except Exception as e:
        print(f"❌ Sync execution failed: {e}")
        raise


async def main():
    query = "Can you recommend an elegant outfit for a very fancy dinner out for men in Como this weekend?"
    print(f"\n💬 USER QUERY: {query}\n")

    response = None

    try:
        print("🔄 Trying async execution...")
        response = await run_agent_async(query)
        print("✅ Async execution successful!")
    except Exception as e:
        print(f"❌ Async execution failed: {e}")

        try:
            print("🔄 Trying sync execution...")
            response = run_agent_sync(query)
            print("✅ Sync execution successful!")
        except Exception as e2:
            print(f"❌ Both async and sync execution failed")
            print(f"Async error: {e}")
            print(f"Sync error: {e2}")
            return

    if response:
        print("\n\n--- FINAL RESPONSE ---")
        print(str(response))
    else:
        print("❌ No response received")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ Asyncio failed: {e}")
        print("🔄 Trying direct sync execution...")

        query = "Can you recommend an elegant outfit for a very fancy dinner out for men in Como this weekend?"
        print(f"\n💬 USER QUERY: {query}\n")

        try:
            response = run_agent_sync(query)
            print("\n\n--- FINAL RESPONSE ---")
            print(str(response))
        except Exception as e2:
            print(f"❌ All execution methods failed: {e2}")
