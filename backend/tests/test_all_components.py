import asyncio
import sys

async def test_components():
    print("🧪 Testing RAG System Components...\n")
    
    # Test 1: Basic imports
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI")
    except Exception as e:
        print(f"Google AI: {e}")
        return False

    # Test 2: Embeddings
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers")
    except Exception as e:
        print(f"Sentence Transformers: {e}")
        return False

    # Test 3: Vector Database
    try:
        import chromadb
        print("✅ ChromaDB")
    except Exception as e:
        print(f"ChromaDB: {e}")
        return False

    # Test 4: Web Scraping
    try:
        from bs4 import BeautifulSoup
        import requests
        print("✅ Web Scraping (BeautifulSoup + Requests)")
    except Exception as e:
        print(f"Web Scraping: {e}")
        return False

    # Test 5: FastAPI
    try:
        from fastapi import FastAPI
        import uvicorn
        print("✅ FastAPI + Uvicorn")
    except Exception as e:
        print(f"FastAPI: {e}")
        return False

    # Test 6: Async operations
    try:
        import aiohttp
        print("✅ Async HTTP (aiohttp)")
    except Exception as e:
        print(f"Async HTTP: {e}")
        return False

    print("\n🎉 All critical components are ready!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_components())
    sys.exit(0 if success else 1)