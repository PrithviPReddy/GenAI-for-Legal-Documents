# In app/routes/endpoints.py

import uuid
from typing import Optional
from fastapi import (
    APIRouter, HTTPException, Depends, UploadFile, File, Form, Response, Cookie
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl

from app.config import BEARER_TOKEN
from app.services.content_processor import ContentProcessor
from app.services.chunker import ImprovedTextChunker
from app.services.vector_store import EnhancedHybridVectorStore
from app.services.llm_processor import ImprovedLLMProcessor
from app.utils.logger import log_document_content, log_search_results
from session_manager import (
    get_session_data,
    update_session_data,
    get_or_create_session_id
)

router = APIRouter(prefix="/api/v1")
security = HTTPBearer()

# --- Initialize processors ---
content_processor = ContentProcessor()
text_chunker = ImprovedTextChunker()
hybrid_vector_store = EnhancedHybridVectorStore()
llm_processor = ImprovedLLMProcessor()

# --- Pydantic Models for New Workflow ---
class UploadResponse(BaseModel):
    message: str
    session_id: str # For debugging/reference

class QARequest(BaseModel):
    questions: list[str]

class ProcessResponse(BaseModel):
    answers: list[str]

class SummarizeResponse(BaseModel):
    summary: str

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials

# --- NEW WORKFLOW ENDPOINTS ---
@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    response: Response,
    url: Optional[HttpUrl] = Form(None),
    file: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Cookie(None),
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """
    Handles document upload and starts/resets a user session.
    """
    if not (url or file) or (url and file):
        raise HTTPException(status_code=400, detail="Provide either a URL or a file, but not both.")

    if url:
        content, content_type = content_processor.download_and_extract(str(url))
    else: # if file
        content = await file.read()
        content_type = file.content_type

    # Always process the document from scratch
    full_text = content_processor.extract_text_from_content(content, content_type)
    chunks = text_chunker.chunk_text(full_text)
    
    # Always generate a new document_id for Pinecone
    new_document_id = str(uuid.uuid4())
    hybrid_vector_store.add_to_pinecone_fallback(chunks, new_document_id)
    
    # Get the user's session ID or create a new one
    active_session_id = get_or_create_session_id(session_id)
    
    # Update the session storage with the new document's data
    update_session_data(active_session_id, new_document_id, full_text)
    
    # Set the session ID in the user's browser cookie
    response.set_cookie(key="session_id", value=active_session_id, httponly=True)
    
    return UploadResponse(message="Document processed and session is active.", session_id=active_session_id)

@router.post("/run", response_model=ProcessResponse)
async def process_documents(
    request: QARequest,
    session_id: Optional[str] = Cookie(None),
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """
    Answers questions based on the document in the current session.
    """
    if not session_id or not (session_data := get_session_data(session_id)):
        raise HTTPException(status_code=400, detail="No active session. Please upload a document first.")
    
    document_id = session_data["document_id"]
    all_relevant_chunks = set()
    for question in request.questions:
        relevant_chunks = hybrid_vector_store.search(question, document_id)
        all_relevant_chunks.update(relevant_chunks[:5])
    
    final_chunks = list(all_relevant_chunks)[:20]
    answers = llm_processor.generate_answers(request.questions, final_chunks)
        
    return ProcessResponse(answers=answers)

@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_document(
    session_id: Optional[str] = Cookie(None),
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """
    Summarizes the document in the current session.
    """
    if not session_id or not (session_data := get_session_data(session_id)):
        raise HTTPException(status_code=400, detail="No active session. Please upload a document first.")
    
    full_text = session_data["full_text"]
    summary = llm_processor.summarize_text(full_text, text_chunker)
    return SummarizeResponse(summary=summary)

# --- Utility Endpoints ---
@router.get("/health")
async def health_check():
    return {"status": "healthy"}
