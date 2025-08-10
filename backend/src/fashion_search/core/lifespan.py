import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI
from llama_index.llms.ollama import Ollama

from .config import settings
from ..agents.orchestrator import MultiFashionAgent
from ..llm.query_enhancer import LLMQueryEnhancer
from ..milvus_client.vector_db_client import VectorDBClient
from ..redis_client.redis_db_client import RedisDBClient
from ..services.redis_search_service import RedisSearchService
from .model_loader import load_clip_model_and_processor
from ..memory.conversation_manager import ConversationManager
from ..agents.refiner import QueryRefinementAgent

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.db_client = VectorDBClient(
            host=settings.MILVUS_HOST, port=settings.MILVUS_PORT
        )

        app.state.db_client.set_collection("articles")

        app.state.redis_client = RedisDBClient(
            host=settings.REDIS_HOST, port=int(settings.REDIS_PORT)
        )

        model, processor = load_clip_model_and_processor()
        app.state.clip_model = model
        app.state.clip_processor = processor

        app.state.llm_enhancer = LLMQueryEnhancer(
            model=settings.LLM_JUDGE_MODEL, prompt_dir=settings.PROMPTS_DIR
        )

        app.state.search_service = RedisSearchService(
            redis_client=app.state.redis_client,
            db_client=app.state.db_client,
            llm_enhancer=app.state.llm_enhancer,
            model=app.state.clip_model,
            processor=app.state.clip_processor,
        )

        llm_for_agent = Ollama(
            model=settings.LLM_JUDGE_MODEL, request_timeout=120.0, temperature=0.1
        )

        app.state.conversation_manager = ConversationManager(
            redis_client=app.state.redis_client
        )

        app.state.query_refiner = QueryRefinementAgent(llm=llm_for_agent)

        app.state.multi_fashion_agent = MultiFashionAgent(
            search_service=app.state.search_service, 
            llm=llm_for_agent,
            conversation_manager=app.state.conversation_manager, # Pass instance
            query_refiner=app.state.query_refiner             # Pass instance
        )

        print("✅ All services initialized, including conversational memory.")
        yield

    except Exception as e:
        print(f"❌ Startup failed: {e}")
        traceback.print_exc()

    print("🔌 Server shutting down...")