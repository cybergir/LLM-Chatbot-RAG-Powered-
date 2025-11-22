from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
import uuid
import time
import logging

from app.schemas.request_schema import IngestRequest, BatchIngestRequest, IngestFromTextRequest
from app.schemas.response_schema import IngestResponse, BatchIngestResponse, IngestStatusResponse
from app.services.ingest_service import ingest_service
from app.middleware.auth import get_current_user, require_role
from app.utils.rate_limiter import rate_limit
from app.models.ingest_models import IngestionJob, IngestionStatus

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory job tracking (in production, use Redis or database)
ingestion_jobs: Dict[str, IngestionJob] = {}

@router.post("/urls", response_model=IngestResponse)
@rate_limit(max_requests=50, window_seconds=3600)
async def ingest_urls(
    req: IngestRequest,
    background_tasks: BackgroundTasks,
    max_pages: int = Query(5, description="Maximum pages to scrape per URL"),
    current_user: dict = Depends(get_current_user)
) -> IngestResponse:
    """
    Ingest content from URLs with background processing
    """
    try:
        # Validate URLs
        if not req.urls:
            raise HTTPException(status_code=400, detail="URL list cannot be empty")
        
        if len(req.urls) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 URLs per request")
        
        # Create ingestion job
        job_id = str(uuid.uuid4())
        job = IngestionJob(
            id=job_id,
            urls=req.urls,
            status=IngestionStatus.PROCESSING,
            submitted_by=current_user.get("user_id", "unknown"),
            created_at=time.time()
        )
        ingestion_jobs[job_id] = job
        
        # Process in background
        background_tasks.add_task(
            process_ingestion_job, 
            job_id, 
            req.urls, 
            max_pages,
            req.metadata or {}
        )
        
        return IngestResponse(
            job_id=job_id,
            message="Ingestion started",
            status="processing",
            urls_received=len(req.urls),
            estimated_time="30-60 seconds"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@router.post("/urls/batch", response_model=BatchIngestResponse)
@require_role("admin")
async def batch_ingest_urls(
    req: BatchIngestRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
) -> BatchIngestResponse:
    """
    Batch ingest URLs with advanced configuration
    """
    try:
        job_id = str(uuid.uuid4())
        job = IngestionJob(
            id=job_id,
            urls=req.urls,
            status=IngestionStatus.PROCESSING,
            submitted_by=current_user.get("user_id", "unknown"),
            created_at=time.time(),
            config=req.config.dict() if req.config else {}
        )
        ingestion_jobs[job_id] = job
        
        background_tasks.add_task(
            process_batch_ingestion, 
            job_id, 
            req.urls, 
            req.config
        )
        
        return BatchIngestResponse(
            job_id=job_id,
            message="Batch ingestion started",
            status="processing",
            total_urls=len(req.urls),
            batch_size=req.config.batch_size if req.config else 10
        )
        
    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/texts")
@rate_limit(max_requests=100, window_seconds=3600)
async def ingest_from_texts(
    req: IngestFromTextRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Ingest content from raw text
    """
    try:
        if not req.texts:
            raise HTTPException(status_code=400, detail="Texts cannot be empty")
        
        results = await ingest_service.ingest_texts(req.texts)
        
        return {
            "status": "success",
            "ingested_texts": len(req.texts),
            "total_chunks": results["total_chunks"],
            "average_chunk_size": results["average_chunk_size"]
        }
        
    except Exception as e:
        logger.error(f"Text ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}", response_model=IngestStatusResponse)
async def get_ingestion_status(
    job_id: str,
    current_user: dict = Depends(get_current_user)
) -> IngestStatusResponse:
    """
    Check status of an ingestion job
    """
    job = ingestion_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Check if user has permission to view this job
    if job.submitted_by != current_user.get("user_id") and not current_user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return IngestionStatusResponse(
        job_id=job.id,
        status=job.status,
        urls_processed=job.urls_processed,
        total_urls=len(job.urls),
        chunks_created=job.chunks_created,
        started_at=job.created_at,
        completed_at=job.completed_at,
        error=job.error
    )

@router.get("/jobs")
async def list_ingestion_jobs(
    skip: int = 0,
    limit: int = 20,
    current_user: dict = Depends(require_role("admin"))
):
    """
    List all ingestion jobs (admin only)
    """
    jobs = list(ingestion_jobs.values())
    jobs.sort(key=lambda x: x.created_at, reverse=True)
    
    paginated_jobs = jobs[skip:skip + limit]
    
    return {
        "jobs": [job.dict() for job in paginated_jobs],
        "total": len(jobs),
        "skip": skip,
        "limit": limit
    }

async def process_ingestion_job(job_id: str, urls: List[str], max_pages: int, metadata: Dict):
    """Background task to process ingestion job"""
    job = ingestion_jobs[job_id]
    
    try:
        logger.info(f"Processing ingestion job {job_id} with {len(urls)} URLs")
        
        # Update job status
        job.status = IngestionStatus.PROCESSING
        
        # Process URLs
        results = await ingest_service.ingest_urls(urls, max_pages_per_url=max_pages)
        
        # Update job with results
        job.status = IngestionStatus.COMPLETED
        job.urls_processed = results["successful_urls"]
        job.chunks_created = results["total_chunks"]
        job.completed_at = time.time()
        
        logger.info(f"Completed ingestion job {job_id}: {job.chunks_created} chunks created")
        
    except Exception as e:
        logger.error(f"Background ingestion failed for job {job_id}: {e}")
        job.status = IngestionStatus.FAILED
        job.error = str(e)
        job.completed_at = time.time()

async def process_batch_ingestion(job_id: str, urls: List[str], config: Any):
    """Process batch ingestion with advanced configuration"""
    # Implementation for batch processing with progress tracking
    pass