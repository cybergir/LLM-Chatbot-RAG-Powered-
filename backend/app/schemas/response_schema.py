from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class SourceItem(BaseModel):
    source: str
    score: float = Field(..., ge=0, le=1)
    content_preview: Optional[str] = None
    chunk_index: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = []
    conversation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = True
    processing_time: Optional[float] = None

class IngestResponse(BaseModel):
    job_id: str
    message: str
    status: str = Field(..., regex="^(processing|completed|failed)$")
    urls_received: int
    estimated_time: Optional[str] = None

class BatchIngestResponse(BaseModel):
    job_id: str
    message: str
    status: str
    total_urls: int
    batch_size: int

class IngestStatusResponse(BaseModel):
    job_id: str
    status: str
    urls_processed: int
    total_urls: int
    chunks_created: int
    started_at: float
    completed_at: Optional[float] = None
    error: Optional[str] = None
    progress: float = Field(0, ge=0, le=100)

class RAGStatsResponse(BaseModel):
    document_count: int
    vector_dimensions: int
    initialized: bool
    google_ai_available: bool
    cache_available: bool
    success: bool = True

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str
    services: Dict[str, str]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)