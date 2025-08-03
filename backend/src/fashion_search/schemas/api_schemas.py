from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    query: str
    top_k: int = 20


class PipelineOptions(BaseModel):
    run_cleanup: bool = Field(
        default=False, 
        description="Run the dataset cleanup step to remove entries with missing images."
    )
    run_captioning: bool = Field(
        default=False, 
        description="Run the image captioning pipeline to generate descriptive text for images."
    )
    run_embeddings: bool = Field(
        default=False, 
        description="Run the text embedding generation pipeline."
    )
    run_db_insertion: bool = Field(
        default=False, 
        description="Insert the generated embeddings into the Milvus vector database."
    )