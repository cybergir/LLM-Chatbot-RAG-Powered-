from pydantic import BaseModel, Field
from typing import List

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query / question")
    top_k: int = Field(5, description="Number of retrieved passages to use")

class IngestRequest(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to ingest")
