from __future__ import annotations
from typing import List, Tuple, Optional
import os
import numpy as np
import asyncio
import logging

from app.utils.config import settings
from app.utils.logger import setup_logger

# simple in-memory vector store
class InMemoryVectorStore:
    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.texts: List[str] = []
        self.meta: List[dict] = []

    def add(self, emb: np.ndarray, text: str, meta: dict):
        self.embeddings.append(emb)
        self.texts.append(text)
        self.meta.append(meta)

    def is_empty(self) -> bool:
        return len(self.embeddings) == 0

    def query(self, emb: np.ndarray, top_k: int = 5) -> List[Tuple[str, dict, float]]:
        if self.is_empty():
            return []
        mats = np.stack(self.embeddings, axis=0)  # (N, dim)
        # cosine similarity
        emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
        mats_norm = mats / (np.linalg.norm(mats, axis=1, keepdims=True) + 1e-10)
        sims = mats_norm.dot(emb_norm)
        top_idx = np.argsort(-sims)[:top_k]
        results = [(self.texts[i], self.meta[i], float(sims[i])) for i in top_idx]
        return results

logger = setup_logger()

class RAGService:
    def __init__(self):
        self.store = InMemoryVectorStore()
        self.embedder = None  # sentence-transformers model set during initialize
        self.openai_key = os.getenv("OPENAI_API_KEY")

    async def initialize(self):
        # load embedder lazily (this can take time)
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model (sentence-transformers)...")
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        logger.info("Embedder loaded.")

    async def add_document(self, text: str, meta: dict):
        emb = self.embedder.encode(text, convert_to_numpy=True)
        self.store.add(emb, text, meta)

    async def get_response(self, query: str, top_k: int = 5) -> Tuple[str, List[dict]]:
        """
        Return (answer_text, sources_list)
        """
        if self.embedder is None:
            raise RuntimeError("RAGService not initialized")

        query_emb = self.embedder.encode(query, convert_to_numpy=True)
        results = self.store.query(query_emb, top_k=top_k)
        if not results:
            return ("I couldn't find relevant information in the indexed sources.", [])

        # build context
        context_texts = []
        sources = []
        for txt, meta, score in results:
            snippet = txt.strip()
            context_texts.append(snippet)
            sources.append({"source": meta.get("source"), "score": score})

        # If OPENAI_API_KEY is provided, call OpenAI with context; else return concatenated snippets
        if self.openai_key:
            try:
                answer = await asyncio.get_event_loop().run_in_executor(
                    None, self._call_openai_completion, query, context_texts
                )
                return (answer, sources)
            except Exception as e:
                logger.exception("OpenAI call failed, returning raw context.")
                return ("\n\n".join(context_texts), sources)
        else:
            # fallback: return top contexts as the answer with citations
            combined = "\n\n".join([f"[Source: {s['source']}] {t}" for t, s, _ in ((r[0], r[1], r[2]) for r in results)])
            return (combined, sources)

    def _call_openai_completion(self, query: str, contexts: List[str]) -> str:
        """
        Synchronous wrapper for OpenAI text generation (so we can run it in executor).
        """
        import openai
        openai.api_key = self.openai_key

        prompt = "Use ONLY the following context to answer the question. If the answer cannot be found, say so.\n\n"
        for i, c in enumerate(contexts):
            prompt += f"Context [{i+1}]: {c}\n\n"
        prompt += f"Question: {query}\n\nAnswer with a concise and source-backed response; include which context(s) you used."

        response = openai.Completion.create(
            engine=settings.OPENAI_COMPLETION_ENGINE,
            prompt=prompt,
            max_tokens=512,
            temperature=0.0,
            n=1,
            stop=None,
        )
        text = response.choices[0].text.strip()
        return text

# singleton service to be imported by routers
rag_service = RAGService()
