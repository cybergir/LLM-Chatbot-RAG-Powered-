from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

class IngestionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class IngestionJob:
    id: str
    urls: List[str]
    status: IngestionStatus
    submitted_by: str
    created_at: float
    config: Optional[Dict[str, Any]] = None
    urls_processed: int = 0
    chunks_created: int = 0
    completed_at: Optional[float] = None
    error: Optional[str] = None
    
    def dict(self):
        return {
            "id": self.id,
            "urls": self.urls,
            "status": self.status,
            "submitted_by": self.submitted_by,
            "created_at": self.created_at,
            "config": self.config,
            "urls_processed": self.urls_processed,
            "chunks_created": self.chunks_created,
            "completed_at": self.completed_at,
            "error": self.error
        }