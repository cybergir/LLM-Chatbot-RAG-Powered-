from __future__ import annotations
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import json
from dataclasses import dataclass
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import redis.asyncio as redis
import hashlib

from app.utils.config import settings
from app.utils.logger import setup_logger
from app.schemas.response_schema import SearchResult

logger = setup_logger()

@dataclass
class RAGConfig:
    top_k: int = 5
    similarity_threshold: float = 0.7
    max_context_length: int = 4000
    enable_hybrid_search: bool = True
    enable_reranking: bool = True

class AsyncVectorStore:
    # Production-ready vector store with persistence and advanced search
    
    def __init__(self, persist_path: str = "./vector_store"):
        self.persist_path = persist_path
        self.embeddings: List[np.ndarray] = []
        self.texts: List[str] = []
        self.metadata: List[Dict] = []
        self._load_from_disk()
    
    def _load_from_disk(self):
        # Load vector store from disk
        try:
            with open(f"{self.persist_path}/vectors.npy", "rb") as f:
                self.embeddings = list(np.load(f))
            with open(f"{self.persist_path}/data.json", "r") as f:
                data = json.load(f)
                self.texts = data["texts"]
                self.metadata = data["metadata"]
            logger.info(f"Loaded {len(self.embeddings)} vectors from disk")
        except FileNotFoundError:
            logger.info("No existing vector store found, starting fresh")
    
    def _persist_to_disk(self):
        """Persist vector store to disk"""
        import os
        os.makedirs(self.persist_path, exist_ok=True)
        
        with open(f"{self.persist_path}/vectors.npy", "wb") as f:
            np.save(f, np.array(self.embeddings))
        
        with open(f"{self.persist_path}/data.json", "w") as f:
            json.dump({
                "texts": self.texts,
                "metadata": self.metadata,
                "updated_at": datetime.now().isoformat()
            }, f, indent=2)
    
    async def add_embeddings_batch(self, embeddings: List[np.ndarray], texts: List[str], metadata: List[Dict]):
        """Add embeddings in batch with persistence"""
        self.embeddings.extend(embeddings)
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        self._persist_to_disk()
    
    async def similarity_search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """Advanced similarity search with filtering and scoring"""
        if not self.embeddings:
            return []
        
        # Convert to numpy for efficient computation
        embeddings_array = np.array(self.embeddings)
        
        # Cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        embeddings_norm = embeddings_array / (np.linalg.norm(embeddings_array, axis=1, keepdims=True) + 1e-10)
        similarities = np.dot(embeddings_norm, query_norm)
        
        # Apply filters if provided
        indices = np.arange(len(similarities))
        if filters:
            filtered_indices = []
            for idx in indices:
                metadata = self.metadata[idx]
                if self._matches_filters(metadata, filters):
                    filtered_indices.append(idx)
            indices = np.array(filtered_indices)
            similarities = similarities[indices]
        
        # Get top results
        top_indices = indices[np.argsort(-similarities)[:top_k]]
        
        results = []
        for idx in top_indices:
            results.append(SearchResult(
                text=self.texts[idx],
                metadata=self.metadata[idx],
                score=float(similarities[idx]),
                id=f"doc_{idx}"
            ))
        
        return results
    
    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        # Check if metadata matches all filters
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True

class GoogleAIService:
    # Service for Google Generative AI integration
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def generate_content(self, prompt: str, context: str, query: str) -> str:
        # Generate content using Google Generative AI
        import google.generativeai as genai
        
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        full_prompt = f"""
        You are a helpful AI assistant. Use ONLY the provided context to answer the user's question.
        
        CONTEXT:
        {context}
        
        USER QUESTION: {query}
        
        INSTRUCTIONS:
        1. Answer based ONLY on the provided context
        2. If the context doesn't contain relevant information, say "I don't have enough information to answer this question based on the provided sources."
        3. Be concise and accurate
        4. Cite sources when relevant by mentioning the document or source
        5. If the question is ambiguous, ask for clarification but try to provide the best answer based on context
        
        ANSWER:
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: model.generate_content(full_prompt)
            )
            return response.text
        except Exception as e:
            logger.error(f"Google AI API error: {e}")
            raise

class RAGService:
    # Production-ready RAG service with advanced features
    
    def __init__(self):
        self.vector_store = AsyncVectorStore()
        self.embedder = None
        self.google_ai = None
        self.redis_client = None
        self.config = RAGConfig()
        self._initialized = False
    
    async def initialize(self):
        # Async initialization with proper error handling
        if self._initialized:
            return
            
        try:
            # Initialize embedding model
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model...")
            self.embedder = SentenceTransformer(
                settings.EMBEDDING_MODEL_NAME,
                device='cuda' if settings.USE_GPU else 'cpu'
            )
            
            # Initialize Google AI
            if settings.GOOGLE_API_KEY:
                self.google_ai = GoogleAIService(settings.GOOGLE_API_KEY)
                logger.info("Google AI service initialized")
            
            # Initialize Redis for caching
            if settings.REDIS_URL:
                self.redis_client = redis.from_url(settings.REDIS_URL)
                logger.info("Redis client initialized")
            
            self._initialized = True
            logger.info("RAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise
    
    async def add_documents_batch(self, documents: List[Dict[str, Any]]) -> int:
        # Add documents in batch for better performance
        if not self._initialized:
            await self.initialize()
        
        texts = [doc["text"] for doc in documents]
        metadata_list = [doc["metadata"] for doc in documents]
        
        # Batch encode for better performance
        logger.info(f"Encoding {len(texts)} documents...")
        embeddings = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.embedder.encode(texts, convert_to_numpy=True)
        )
        
        await self.vector_store.add_embeddings_batch(
            list(embeddings), texts, metadata_list
        )
        
        # Invalidate relevant caches
        if self.redis_client:
            await self._invalidate_cache_pattern("rag_search:*")
        
        return len(texts)
    
    async def get_response(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[Dict] = None,
        conversation_id: Optional[str] = None
    ) -> Tuple[str, List[Dict]]:
        
        # Advanced RAG response generation with caching and conversation context
        
        if not self._initialized:
            await self.initialize()
        
        # Check cache first
        cache_key = await self._generate_cache_key(query, top_k, filters)
        if cached := await self._get_cached_response(cache_key):
            return cached
        
        # Generate query embedding
        query_embedding = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.embedder.encode(query, convert_to_numpy=True)
        )
        
        # Search vector store
        search_results = await self.vector_store.similarity_search(
            query_embedding, top_k=top_k, filters=filters
        )
        
        if not search_results:
            response = ("I couldn't find relevant information in the knowledge base to answer your question. "
                       "Please try rephrasing or ask about something else.", [])
            await self._cache_response(cache_key, response)
            return response
        
        # Build context and generate response
        context = self._build_context(search_results)
        answer = await self._generate_answer(query, context, conversation_id)
        
        # Prepare sources
        sources = [
            {
                "source": result.metadata.get("source", "Unknown"),
                "chunk_index": result.metadata.get("chunk_index", 0),
                "score": result.score,
                "content_preview": result.text[:200] + "..." if len(result.text) > 200 else result.text
            }
            for result in search_results
        ]
        
        response = (answer, sources)
        await self._cache_response(cache_key, response)
        
        return response
    
    def _build_context(self, search_results: List[SearchResult]) -> str:
        # Build context from search results
        context_parts = []
        for i, result in enumerate(search_results):
            context_parts.append(
                f"[Document {i+1} from {result.metadata.get('source', 'Unknown')}]:\n"
                f"{result.text}\n"
            )
        return "\n".join(context_parts)
    
    async def _generate_answer(self, query: str, context: str, conversation_id: Optional[str]) -> str:
        # Generate answer using Google AI or fallback
        if self.google_ai:
            try:
                return await self.google_ai.generate_content(
                    prompt="You are a helpful AI assistant that answers questions based on provided context.",
                    context=context,
                    query=query
                )
            except Exception as e:
                logger.error(f"Google AI generation failed: {e}")
                # Fall through to fallback
        
        # Fallback: return top result with citation
        return self._fallback_answer(context, query)
    
    def _fallback_answer(self, context: str, query: str) -> str:
        # Fallback answer when AI service is unavailable
        lines = context.split('\n')
        if lines:
            main_source = lines[0] if len(lines) > 0 else "the provided sources"
            return (f"Based on {main_source}, here's the relevant information for your question '{query}':\n\n"
                   f"{context[:2000]}...\n\n[Note: This is a direct excerpt from the source material]")
        return "I found some relevant information but couldn't process it properly. Please try again."
    
    async def _generate_cache_key(self, query: str, top_k: int, filters: Optional[Dict]) -> str:
        # Generate cache key for query
        key_data = f"{query}:{top_k}:{json.dumps(filters or {})}"
        return f"rag_search:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    async def _get_cached_response(self, cache_key: str) -> Optional[Tuple[str, List[Dict]]]:
        # Get cached response from Redis
        if not self.redis_client:
            return None
        
        try:
            cached = await self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
        return None
    
    async def _cache_response(self, cache_key: str, response: Tuple[str, List[Dict]]):
        # Cache response in Redis
        if not self.redis_client:
            return
        
        try:
            await self.redis_client.setex(
                cache_key, 
                settings.CACHE_TTL, 
                json.dumps(response)
            )
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
    
    async def _invalidate_cache_pattern(self, pattern: str):
        # Invalidate cache entries matching pattern
        if not self.redis_client:
            return
        
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        # Get RAG service statistics
        return {
            "document_count": len(self.vector_store.texts),
            "vector_dimensions": self.vector_store.embeddings[0].shape[0] if self.vector_store.embeddings else 0,
            "initialized": self._initialized,
            "google_ai_available": self.google_ai is not None,
            "cache_available": self.redis_client is not None
        }

# Singleton instance
rag_service = RAGService()