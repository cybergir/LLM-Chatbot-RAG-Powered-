from typing import Any
from fastapi import APIRouter, HTTPException
from app.schemas.request_schema import QueryRequest
from app.schemas.response_schema import ChatResponse
from app.services.rag_service import rag_service

router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(req: QueryRequest) -> ChatResponse:
   
    # Query the RAG system and return a source-backed answer.
    
    try:
        answer, sources = await rag_service.get_response(req.query, top_k=req.top_k)
        return ChatResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
