# In main.py

# --- 1. ADD THIS IMPORT ---
import uvicorn

# Original imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import endpoints
from app.config import embedding_model, pinecone_client, pinecone_index, logger
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
import pinecone
import os
import google.generativeai as genai

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

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(endpoints.router)


# --- 2. ADD THIS BLOCK AT THE END ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
