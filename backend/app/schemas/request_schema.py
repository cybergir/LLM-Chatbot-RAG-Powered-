from pydantic import BaseModel, Field, validator, HttpUrl
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User query / question")
    top_k: int = Field(5, ge=1, le=20, description="Number of retrieved passages to use")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters for retrieval")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    stream: bool = Field(False, description="Whether to stream the response")

    @validator('query')
    def query_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty or whitespace')
        return v.strip()

# Add these missing classes
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User query / question")
    top_k: int = Field(5, ge=1, le=20, description="Number of retrieved passages to use")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters for retrieval")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")

class StreamingChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None

class IngestRequest(BaseModel):
    urls: List[HttpUrl] = Field(..., description="List of URLs to ingest")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for ingested content")
    chunk_size: int = Field(1000, ge=100, le=2000, description="Target chunk size in characters")
    chunk_overlap: int = Field(100, ge=0, le=500, description="Chunk overlap in characters")

class BatchIngestConfig(BaseModel):
    batch_size: int = Field(10, ge=1, le=50)
    max_pages_per_url: int = Field(5, ge=1, le=20)
    follow_links: bool = Field(True)
    respect_robots_txt: bool = Field(True)
    chunk_strategy: str = Field("semantic", pattern="^(semantic|fixed|paragraph)$")

class BatchIngestRequest(BaseModel):
    urls: List[HttpUrl] = Field(..., description="List of URLs to ingest")
    config: Optional[BatchIngestConfig] = Field(None, description="Batch ingestion configuration")

class IngestFromTextRequest(BaseModel):
    texts: List[Dict[str, Any]] = Field(..., description="List of text objects with content and metadata")
    
    @validator('texts')
    def validate_texts(cls, v):
        for text_obj in v:
            if 'text' not in text_obj or not text_obj['text'].strip():
                raise ValueError('Each text object must contain non-empty "text" field')
        return v