from __future__ import annotations
from typing import List, Dict, Any
import asyncio
import re
import textwrap
import logging

import requests
from bs4 import BeautifulSoup

from app.services.rag_service import rag_service

logger = logging.getLogger("ingest_service")

def clean_text(s: str) -> str:
    # basic cleaning
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    # naive chunker by sentences
    paragraphs = text.split("\n")
    chunks = []
    current = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(current) + len(p) + 1 <= max_chars:
            current = (current + " " + p).strip()
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    # further shorten long chunks
    final = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            for i in range(0, len(c), max_chars):
                final.append(c[i:i+max_chars])
    return final

class IngestService:
    def __init__(self):
        pass

    async def ingest_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        For each url: scrape text, chunk, embed (via rag_service), and store.
        Returns list of ingested metadata.
        """
        results = []
        # fetch in parallel
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(None, self._fetch_and_parse, url) for url in urls]
        pages = await asyncio.gather(*tasks)
        for url, text in pages:
            if not text:
                logger.warning(f"No text extracted from {url}")
                continue
            chunks = chunk_text(text, max_chars=1000)
            for i, chunk in enumerate(chunks):
                meta = {"source": url, "chunk_index": i}
                await rag_service.add_document(chunk, meta)
            results.append({"url": url, "chunks": len(chunks)})
        return results

    def _fetch_and_parse(self, url: str) -> (str, str):
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            # remove scripts/styles
            for t in soup(["script", "style", "header", "footer", "nav", "aside"]):
                t.decompose()
            texts = [p.get_text(separator=" ", strip=True) for p in soup.find_all(["p", "li", "h1", "h2", "h3"])]
            combined = "\n".join(texts)
            cleaned = clean_text(combined)
            return url, cleaned
        except Exception as e:
            logger.exception(f"Failed to fetch {url}: {e}")
            return url, ""

ingest_service = IngestService()
