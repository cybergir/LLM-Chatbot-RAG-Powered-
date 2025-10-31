from pydantic import BaseModel
from typing import List, Dict, Any

class SourceItem(BaseModel):
    source: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = []
