from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import BEARER_TOKEN
from app.services.content_processor import ContentProcessor
from app.services.chunker import ImprovedTextChunker
from app.services.vector_store import EnhancedHybridVectorStore
from app.services.llm_processor import ImprovedLLMProcessor
from app.utils.logger import log_document_content, log_chunks_preview, log_search_results
from cache_manager import get_cached_document, cache_document, cache_stats
from pydantic import BaseModel, HttpUrl
import uuid

router = APIRouter(prefix="/api/v1")
security = HTTPBearer()

# Initialize processors
content_processor = ContentProcessor()
text_chunker = ImprovedTextChunker()
hybrid_vector_store = EnhancedHybridVectorStore()
llm_processor = ImprovedLLMProcessor()

class ProcessRequest(BaseModel):
    documents: HttpUrl
    questions: list[str]

class ProcessResponse(BaseModel):
    answers: list[str]

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials

@router.post("/hackrx/run", response_model=ProcessResponse)
async def process_documents(request: ProcessRequest, credentials: HTTPAuthorizationCredentials = Depends(verify_token)):
    url = str(request.documents)
    cached_doc = get_cached_document(url)
    if cached_doc:
        document_id = cached_doc["document_id"]
        chunks = cached_doc["chunks"]
    else:
        text = content_processor.download_and_extract(url)
        log_document_content(text)
        chunks = text_chunker.chunk_text(text)
        document_id = str(uuid.uuid4())
        hybrid_vector_store.add_to_pinecone_fallback(chunks, document_id)
        cache_document(url, document_id, chunks)

    # Process questions
    all_relevant_chunks = set()
    for question in request.questions:
        relevant_chunks = hybrid_vector_store.search(question, document_id)
        log_search_results(question, relevant_chunks)
        all_relevant_chunks.update(relevant_chunks[:5])
    final_chunks = list(all_relevant_chunks)[:20]

    answers = llm_processor.generate_answers(request.questions, final_chunks)
    return ProcessResponse(answers=answers)

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.get("/cache/stats")
async def cache_stats_endpoint():
    return cache_stats()
