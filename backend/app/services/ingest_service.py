from __future__ import annotations
import asyncio
import aiohttp
import re
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken
from bs4 import BeautifulSoup

from app.services.rag_service import rag_service
from app.utils.config import settings

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]
    tokens: int

class AdvancedTextProcessor:
    # Advanced text processing with semantic chunking
    
    def __init__(self):
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.min_chunk_size = 200  # tokens
        self.max_chunk_size = 1000  # tokens
        self.overlap = 50  # tokens
    
    def count_tokens(self, text: str) -> int:
        # Count tokens in text
        return len(self.encoder.encode(text))
    
    def semantic_chunk(self, text: str, metadata: Dict) -> List[Chunk]:
        # Advanced semantic chunking preserving context
        # Split into paragraphs first
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            # If paragraph itself is too large, split by sentences
            if para_tokens > self.max_chunk_size:
                sentences = re.split(r'[.!?]+', para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    sent_tokens = self.count_tokens(sentence)
                    self._add_to_chunk(sentence, sent_tokens, metadata, chunks, current_chunk, current_tokens)
            else:
                self._add_to_chunk(para, para_tokens, metadata, chunks, current_chunk, current_tokens)
        
        # Add final chunk
        if current_chunk:
            chunks.append(Chunk(
                text='\n'.join(current_chunk),
                metadata=metadata.copy(),
                tokens=current_tokens
            ))
        
        return chunks
    
    def _add_to_chunk(self, text: str, tokens: int, metadata: Dict, 
                     chunks: List[Chunk], current_chunk: List[str], current_tokens: int):
        # Add text to current chunk or create new chunk
        if current_tokens + tokens > self.max_chunk_size and current_chunk:
            # Save current chunk and start new one with overlap
            chunks.append(Chunk(
                text='\n'.join(current_chunk),
                metadata=metadata.copy(),
                tokens=current_tokens
            ))
            
            # Keep last few sentences for overlap
            if len(current_chunk) > 2:
                current_chunk = current_chunk[-2:]
                current_tokens = sum(self.count_tokens(t) for t in current_chunk)
            else:
                current_chunk = []
                current_tokens = 0
        
        current_chunk.append(text)
        current_tokens += tokens

class AsyncWebScraper:
    # Async web scraper with rate limiting and politeness
    
    def __init__(self):
        self.session = None
        self.processor = AdvancedTextProcessor()
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Mozilla/5.0 (compatible; RAG-Bot/1.0)'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def scrape_url(self, url: str, max_pages: int = 10) -> List[Chunk]:
        # Scrape URL and internal links with depth control
        chunks = []
        visited = set()
        
        await self._scrape_recursive(url, max_pages, 0, visited, chunks)
        return chunks
    
    async def _scrape_recursive(self, url: str, max_pages: int, depth: int, 
                               visited: set, chunks: List[Chunk]):
        # Recursively scrape pages
        if depth >= 2 or len(visited) >= max_pages or url in visited:
            return
        
        visited.add(url)
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract main content
                title = soup.find('title')
                title_text = title.get_text().strip() if title else "No Title"
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    element.decompose()
                
                # Try to find main content area
                main_content = self._extract_main_content(soup)
                if not main_content:
                    main_content = soup
                
                text = main_content.get_text(separator='\n', strip=True)
                cleaned_text = self._clean_text(text)
                
                if cleaned_text:
                    # Chunk the content
                    metadata = {
                        "source": url,
                        "title": title_text,
                        "depth": depth,
                        "scraped_at": asyncio.get_event_loop().time()
                    }
                    
                    content_chunks = self.processor.semantic_chunk(cleaned_text, metadata)
                    chunks.extend(content_chunks)
                    
                    logger.info(f"Scraped {url} - {len(content_chunks)} chunks")
                    
                    # Recursively scrape internal links
                    if depth < 1:  # Limit recursion depth
                        internal_links = self._get_internal_links(url, soup)
                        for link in internal_links[:5]:  # Limit to 5 links per page
                            if len(visited) < max_pages:
                                await self._scrape_recursive(link, max_pages, depth + 1, visited, chunks)
                
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
    
    def _extract_main_content(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        # Extract main content using common patterns
        # Try common main content selectors
        selectors = [
            'main', 'article', '[role="main"]', 
            '.content', '.main-content', '#content',
            '.post-content', '.article-content'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element
        
        # Fallback: try to find the largest text container
        candidates = soup.find_all(['div', 'section'])
        if candidates:
            return max(candidates, key=lambda x: len(x.get_text(strip=True)))
        
        return None
    
    def _get_internal_links(self, base_url: str, soup: BeautifulSoup) -> List[str]:
        # Extract internal links from page
        internal_links = set()
        base_domain = urlparse(base_url).netloc
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # Check if it's internal and not a fragment
            if (urlparse(full_url).netloc == base_domain and 
                '#' not in full_url and
                not any(ext in full_url.lower() for ext in ['.pdf', '.jpg', '.png'])):
                internal_links.add(full_url)
        
        return list(internal_links)
    
    def _clean_text(self, text: str) -> str:
        # Clean and normalize text
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common boilerplate
        boilerplate_phrases = [
            'cookie policy', 'privacy policy', 'terms of service',
            'skip to content', 'menu', 'navigation'
        ]
        for phrase in boilerplate_phrases:
            text = re.sub(phrase, '', text, flags=re.IGNORECASE)
        return text.strip()

class IngestService:
    # Production-ready ingestion service
    
    def __init__(self):
        self.processor = AdvancedTextProcessor()
        self.scraper = AsyncWebScraper()
    
    async def ingest_urls(
        self, 
        urls: List[str], 
        max_pages_per_url: int = 5,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        
        # Advanced URL ingestion with progress tracking
        
        logger.info(f"Starting ingestion of {len(urls)} URLs")
        
        all_chunks = []
        stats = {
            "total_urls": len(urls),
            "successful_urls": 0,
            "failed_urls": 0,
            "total_chunks": 0,
            "failed_urls_list": []
        }
        
        async with self.scraper as scraper:
            for i in range(0, len(urls), batch_size):
                batch = urls[i:i + batch_size]
                batch_tasks = [
                    scraper.scrape_url(url, max_pages_per_url) 
                    for url in batch
                ]
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for url, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to ingest {url}: {result}")
                        stats["failed_urls"] += 1
                        stats["failed_urls_list"].append(url)
                    else:
                        all_chunks.extend(result)
                        stats["successful_urls"] += 1
                        logger.info(f"Successfully ingested {url} - {len(result)} chunks")
        
        # Batch add to vector store
        if all_chunks:
            documents = [
                {
                    "text": chunk.text,
                    "metadata": {
                        **chunk.metadata,
                        "tokens": chunk.tokens,
                        "ingestion_id": f"ingest_{hash(url)}"
                    }
                }
                for chunk in all_chunks
            ]
            
            added_count = await rag_service.add_documents_batch(documents)
            stats["total_chunks"] = added_count
        
        logger.info(f"Ingestion completed: {stats}")
        return stats
    
    async def ingest_texts(self, texts: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Ingest raw texts with metadata
        all_chunks = []
        
        for text_data in texts:
            text = text_data["text"]
            metadata = text_data.get("metadata", {})
            
            chunks = self.processor.semantic_chunk(text, metadata)
            all_chunks.extend(chunks)
        
        if all_chunks:
            documents = [
                {
                    "text": chunk.text,
                    "metadata": {
                        **chunk.metadata,
                        "tokens": chunk.tokens,
                        "source": "direct_upload"
                    }
                }
                for chunk in all_chunks
            ]
            
            added_count = await rag_service.add_documents_batch(documents)
            
            return {
                "total_texts": len(texts),
                "total_chunks": added_count,
                "average_chunk_size": sum(chunk.tokens for chunk in all_chunks) / len(all_chunks)
            }
        
        return {"total_texts": 0, "total_chunks": 0, "average_chunk_size": 0}

# Singleton instance
ingest_service = IngestService()