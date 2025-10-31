from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import chat, ingest
from app.utils.logger import setup_logger
from app.utils.config import settings
from app.services.rag_service import rag_service

logger = setup_logger()

def create_app() -> FastAPI:
    app = FastAPI(
        title="LLM Chatbot (RAG-Powered) API",
        description="FastAPI backend for RAG-powered LLM Chatbot",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ingest.router, prefix="/ingest", tags=["Ingest"])
    app.include_router(chat.router, prefix="/chat", tags=["Chat"])

    @app.get("/")
    async def read_root():
        return {"message": "LLM Chatbot (RAG-Powered) is running"}

    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting up: initializing RAG service...")
        await rag_service.initialize()
        logger.info("Startup complete.")

    return app


app = create_app()
