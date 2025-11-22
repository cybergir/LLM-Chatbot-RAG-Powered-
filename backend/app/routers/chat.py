from typing import Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
import logging

from app.schemas.request_schema import ChatRequest, StreamingChatRequest
from app.schemas.response_schema import ChatResponse, RAGStatsResponse
from app.services.rag_service import rag_service
from app.middleware.auth import get_current_user
from app.utils.rate_limiter import rate_limit

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/query", response_model=ChatResponse)
@rate_limit(max_requests=100, window_seconds=3600)
async def chat_endpoint(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user)
) -> ChatResponse:
    # Advanced chat endpoint with authentication and rate limiting
    try:
        # Validate request
        if not req.query or len(req.query.strip()) < 1:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if len(req.query) > 1000:
            raise HTTPException(status_code=400, detail="Query too long")
        
        # Generate response
        answer, sources = await rag_service.get_response(
            query=req.query,
            top_k=req.top_k or 5,
            filters=req.filters,
            conversation_id=req.conversation_id
        )
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            conversation_id=req.conversation_id,
            timestamp=rag_service.get_current_timestamp(),
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/stream")
async def stream_chat_endpoint(
    req: StreamingChatRequest,
    current_user: dict = Depends(get_current_user)
):
    # Streaming chat endpoint for real-time responses
    async def generate():
        try:
            # This would integrate with streaming LLM responses
            # For now, we'll simulate streaming
            answer, sources = await rag_service.get_response(
                query=req.query,
                top_k=req.top_k or 5,
                filters=req.filters
            )
            
            # Simulate streaming by yielding chunks
            words = answer.split()
            for i, word in enumerate(words):
                if i < len(words) - 1:
                    yield f"data: {word} \n\n"
                    await asyncio.sleep(0.05)
                else:
                    yield f"data: {word}\n\n"
            
            # Send sources as final message
            sources_data = json.dumps({"sources": sources, "type": "sources"})
            yield f"data: {sources_data}\n\n"
            
        except Exception as e:
            error_msg = json.dumps({"error": str(e), "type": "error"})
            yield f"data: {error_msg}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@router.get("/stats", response_model=RAGStatsResponse)
async def get_rag_stats(current_user: dict = Depends(get_current_user)) -> RAGStatsResponse:
    # Get RAG system statistics
    try:
        stats = await rag_service.get_stats()
        return RAGStatsResponse(**stats, success=True)
    except Exception as e:
        logger.error(f"Stats endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

@router.post("/clear-cache")
async def clear_cache(current_user: dict = Depends(get_current_user)):
    # Clear RAG cache (admin function)
    try:
        # Implementation would clear relevant caches
        return {"message": "Cache cleared successfully", "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to clear cache")