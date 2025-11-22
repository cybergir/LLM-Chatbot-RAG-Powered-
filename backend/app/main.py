from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from prometheus_fastapi_instrumentator import Instrumentator
import time
import logging

from app.routers import chat, ingest, admin, monitoring
from app.utils.logger import setup_logger
from app.utils.config import settings
from app.services.rag_service import rag_service
from app.middleware.exception_handlers import setup_exception_handlers

logger = setup_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting up: initializing RAG service...")
    startup_time = time.time()
    
    try:
        await rag_service.initialize()
        startup_duration = time.time() - startup_time
        logger.info(f"✅ Startup complete in {startup_duration:.2f}s")
        
        # Log service health
        stats = await rag_service.get_stats()
        logger.info(f"📊 Service stats: {stats}")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield  # App runs here
    
    # Shutdown
    logger.info("🛑 Shutting down RAG service...")
    # Add any cleanup logic here

def create_app() -> FastAPI:
    app = FastAPI(
        title="Enterprise RAG Chatbot API",
        description="Production-ready RAG-powered LLM Chatbot with advanced features",
        version="1.0.0",
        docs_url="/docs" if settings.ENABLE_DOCS else None,
        redoc_url="/redoc" if settings.ENABLE_DOCS else None,
        openapi_url="/openapi.json" if settings.ENABLE_DOCS else None,
        lifespan=lifespan
    )

    # Security Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time"]
    )
    
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Custom middleware for request tracking
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        request_id = request.headers.get('X-Request-ID', f"req_{int(time.time()*1000)}")
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        
        # Log request details
        logger.info(
            f"Request {request_id}: {request.method} {request.url} "
            f"completed in {process_time:.3f}s "
            f"status={response.status_code}"
        )
        
        return response

    # Setup exception handlers
    setup_exception_handlers(app)
    
    # Include routers
    app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["Data Ingestion"])
    app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
    app.include_router(monitoring.router, prefix="/api/v1/monitoring", tags=["Monitoring"])
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["Administration"])
    
    # Prometheus metrics
    if settings.ENABLE_METRICS:
        Instrumentator().instrument(app).expose(app)
    
    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Comprehensive health check"""
        try:
            stats = await rag_service.get_stats()
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "version": "1.0.0",
                "services": {
                    "rag_service": "healthy",
                    "vector_store": "healthy" if stats["document_count"] >= 0 else "degraded"
                }
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "error": str(e)}
            )
    
    @app.get("/", tags=["Root"])
    async def read_root():
        return {
            "message": "Enterprise RAG Chatbot API",
            "version": "1.0.0",
            "status": "operational",
            "docs": "/docs" if settings.ENABLE_DOCS else "disabled"
        }

    return app

app = create_app()