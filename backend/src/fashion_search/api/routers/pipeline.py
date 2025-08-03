from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from ...schemas.api_schemas import PipelineOptions
from ...pipeline.steps import CleanupStep, CaptioningStep, EmbeddingStep, DbInsertionStep
from ...milvus_client.vector_db_client import VectorDBClient

router = APIRouter(prefix="/pipeline", tags=["Data Processing Pipeline"])

@router.post("/")
async def run_data_pipeline(options: PipelineOptions, request: Request, background_tasks: BackgroundTasks):
    if not any(vars(options).values()):
        raise HTTPException(
            status_code=400,
            detail="No pipeline step was selected."
        )

    try:
        db_client: VectorDBClient = request.app.state.db_client
    except AttributeError:
        raise HTTPException(status_code=503, detail="Database client not available.")

    step_map = {
        "run_cleanup": CleanupStep(),
        "run_captioning": CaptioningStep(),
        "run_embeddings": EmbeddingStep(),
        "run_db_insertion": DbInsertionStep(db_client=db_client),
    }

    for option, step in step_map.items():
        if getattr(options, option):
            background_tasks.add_task(step.run)
            print(f"🚀 Task '{step.__class__.__name__}' added to background queue.")

    return {
        "message": "Pipeline tasks have been successfully triggered in the background.",
        "options_received": options.model_dump()
    }