from pydantic import BaseModel

class RedisJSON(BaseModel):
    usr_query: str
    transformed_usr_query: str
    transformed_usr_query_embedding: str
    milvus_result: list[float] 
    cached_at: str
