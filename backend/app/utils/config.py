import os
from typing import List, Optional
from pydantic import BaseSettings, AnyHttpUrl

class Settings(BaseSettings):
    # API Configuration
    ENVIRONMENT: str = "development"
    CORS_ORIGINS: List[AnyHttpUrl] = ["http://localhost:3000"]
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    ENABLE_DOCS: bool = True
    ENABLE_METRICS: bool = True
    
    # API Keys
    GOOGLE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    
    # Embedding Model
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    # Redis
    REDIS_URL: Optional[str] = None
    CACHE_TTL: int = 3600
    
    # Security
    JWT_SECRET_KEY: str = "your-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    
    # Performance
    USE_GPU: bool = False
    MAX_WORKERS: int = 4
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()