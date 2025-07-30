from pydantic import BaseModel

class PipelineOptions(BaseModel):
    run_cleaning: bool = True
    run_captioning: bool = True
    run_embedding: bool = True
    run_db_insertion: bool = True
