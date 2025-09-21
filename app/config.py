import os
from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
import logging

load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hackrx")

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

if not BEARER_TOKEN:
    raise ValueError("BEARER_TOKEN is not set!")

# Global models (to be initialized on startup)
embedding_model = None
pinecone_client = None
pinecone_index = None

