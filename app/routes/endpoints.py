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

router = APIRouter(prefix="/api")
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

@router.post("/run", response_model=ProcessResponse)
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

# Add these new Pydantic models with the others
class SummarizeRequest(BaseModel):
    documents: HttpUrl

class SummarizeResponse(BaseModel):
    summary: str


# Add this new endpoint function within the file
@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_document(request: SummarizeRequest, credentials: HTTPAuthorizationCredentials = Depends(verify_token)):
    """
    Downloads a document from a URL, extracts its content, and returns a summary.
    """
    url = str(request.documents)
    
    try:
        # Step 1: Download and extract the full text content.
        # We reuse the existing content_processor for this.
        full_text = content_processor.download_and_extract(url)
        log_document_content(full_text)

        # Step 2: Pass the text and the chunker instance to the new summarize method.
        # The llm_processor will handle chunking and summarization.
        summary = llm_processor.summarize_text(full_text, text_chunker)

        # Step 3: Return the final summary.
        return SummarizeResponse(summary=summary)
        
    except HTTPException as http_exc:
        # Re-raise FastAPI-specific exceptions
        raise http_exc
    except Exception as e:
        # Handle other potential errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.get("/cache/stats")
async def cache_stats_endpoint():
    return cache_stats()
