from fastapi import FastAPI
from app.routes import endpoints
from app.config import embedding_model, pinecone_client, pinecone_index, logger
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
import pinecone
import os
import google.generativeai as genai
from app.services.content_processor import ContentProcessor
from app.services.chunker import ImprovedTextChunker
from app.services.vector_store import EnhancedHybridVectorStore
from app.services.llm_processor import ImprovedLLMProcessor
from app.utils.logger import logger  # Use this logger everywhere


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, pinecone_client, pinecone_index
    try:
        logger.info("Loading embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        logger.info("Initializing Pinecone...")
        pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX")
        pinecone_index = pc.Index(index_name)

        logger.info("Initializing Gemini...")
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        yield
    finally:
        logger.info("Shutting down services...")

app = FastAPI(title="HackRx RAG API", lifespan=lifespan)
app.include_router(endpoints.router)
