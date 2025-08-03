from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from ..core.lifespan import lifespan
from ..core.config import settings
from .routers import recommendation, pipeline, search 

app = FastAPI(
    title="Fashion Search API",
    description="An advanced fashion search engine using a multi-agent system.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/images", StaticFiles(directory=settings.IMAGE_BASE_DIR), name="images")

app.include_router(recommendation.router)
app.include_router(pipeline.router)
app.include_router(search.router)


@app.get("/", tags=["Health Check"])
def root():
    return {"status": "Backend is running"}