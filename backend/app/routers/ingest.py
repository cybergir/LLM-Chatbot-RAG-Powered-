from typing import List
from fastapi import APIRouter, HTTPException
from app.schemas.request_schema import IngestRequest
from app.services.ingest_service import ingest_service

router = APIRouter()

@router.post("/urls")
async def ingest_urls(req: IngestRequest):
    # Ingest a list of URLs (scrape, chunk, embed, and store vectors).
    try:
        result = await ingest_service.ingest_urls(req.urls)
        return {"status": "ok", "ingested": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
