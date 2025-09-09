# In main.py

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

# --- ADD THESE IMPORTS FOR YOUR SERVICES ---
from app.services.content_processor import ContentProcessor
from app.services.chunker import ImprovedTextChunker
from app.services.llm_processor import ImprovedLLMProcessor
from app.services.vector_store import EnhancedHybridVectorStore
# -----------------------------------------

from app.routes import endpoints
from app.utils.logger import logger  # Corrected logger import
from sentence_transformers import SentenceTransformer
import pinecone
import google.generativeai as genai

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")
    
    # Load models and clients and attach them to the app's state
    logger.info("Loading embedding model...")
    app.state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    logger.info("Initializing Pinecone...")
    pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX")
    app.state.pinecone_index = pc.Index(index_name)

    logger.info("Initializing Gemini...")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    # Also initialize your services here and attach them to the state
    app.state.content_processor = ContentProcessor()
    app.state.text_chunker = ImprovedTextChunker()
    app.state.llm_processor = ImprovedLLMProcessor()
    app.state.vector_store = EnhancedHybridVectorStore(
        embedding_model=app.state.embedding_model,
        pinecone_index=app.state.pinecone_index
    )

    yield
    
    logger.info("Application shutdown...")

app = FastAPI(title="HackRx RAG API", lifespan=lifespan)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "null", 
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8080", 
        "http://localhost:8080",
        "http://localhost:8081",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(endpoints.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)