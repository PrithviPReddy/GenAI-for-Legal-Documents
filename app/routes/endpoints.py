import uuid
from typing import Optional, List, Dict
from fastapi import (
    APIRouter, HTTPException, Depends, UploadFile, File, Form, Response, Cookie, Request
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl

# from app.config import BEARER_TOKEN
from session_manager import (
    get_session_data,
    update_session_data,
    get_or_create_session_id
)

router = APIRouter(prefix="/api/v1")
security = HTTPBearer()


# --- Pydantic Models for New Workflow ---

class RiskItem(BaseModel):
    risk_category: str
    explanation: str
    quote: str

class AnalyzeResponse(BaseModel):
    risks: list[RiskItem] 


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
@router.get("/")
async def root_health():
    return {"status": "ok", "message": "Gen AI API backend is running!"}

@router.post("/analyze/risks", response_model=AnalyzeResponse)
async def analyze_document_risks(
    request: Request,
    session_id: Optional[str] = Cookie(None),
    # credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """
    Analyzes the document in the current session for potential risks.
    """
    llm_processor = request.app.state.llm_processor

    if not session_id or not (session_data := get_session_data(session_id)):
        raise HTTPException(status_code=400, detail="No active session. Please upload a document first.")
    
    full_text = session_data["full_text"]
    
    found_risks = llm_processor.analyze_text_for_risks(full_text)
    
    return AnalyzeResponse(risks=found_risks)

    
@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    request: Request, # Add request to access app state
    response: Response,
    url: Optional[HttpUrl] = Form(None),
    file: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Cookie(None),
    # credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """
    Handles document upload and starts/resets a user session.
    """
    # Access the initialized services from the request's application state
    content_processor = request.app.state.content_processor
    text_chunker = request.app.state.text_chunker
    vector_store = request.app.state.vector_store
    
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
    vector_store.add_to_pinecone_fallback(chunks, new_document_id)
    
    # Get the user's session ID or create a new one
    active_session_id = get_or_create_session_id(session_id)
    
    # Update the session storage with the new document's data
    update_session_data(active_session_id, new_document_id, full_text)
    
    # Set the session ID in the user's browser cookie
    response.set_cookie(key="session_id", value=active_session_id, httponly=True)
    
    return UploadResponse(message="Document processed and session is active.", session_id=active_session_id)

@router.post("/run", response_model=ProcessResponse)
async def process_documents(
    qa_request: QARequest, # Renamed to avoid conflict with 'request'
    request: Request,      # Add request to access app state
    session_id: Optional[str] = Cookie(None),
    # credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """
    Answers questions based on the document in the current session.
    """
    # Access the initialized services from the request's application state
    vector_store = request.app.state.vector_store
    llm_processor = request.app.state.llm_processor

    if not session_id or not (session_data := get_session_data(session_id)):
        raise HTTPException(status_code=400, detail="No active session. Please upload a document first.")
    
    document_id = session_data["document_id"]
    all_relevant_chunks = set()
    for question in qa_request.questions:
        relevant_chunks = vector_store.search(question, document_id)
        all_relevant_chunks.update(relevant_chunks[:5])
    
    final_chunks = list(all_relevant_chunks)[:20]
    answers = llm_processor.generate_answers(qa_request.questions, final_chunks)
        
    return ProcessResponse(answers=answers)

@router.post("/summarize", response_model=SummarizeResponse)
async def summarize_document(
    request: Request, # Add request to access app state
    session_id: Optional[str] = Cookie(None),
    # credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """
    Summarizes the document in the current session.
    """
    # Access the initialized services from the request's application state
    llm_processor = request.app.state.llm_processor
    text_chunker = request.app.state.text_chunker

    if not session_id or not (session_data := get_session_data(session_id)):
        raise HTTPException(status_code=400, detail="No active session. Please upload a document first.")
    
    full_text = session_data["full_text"]
    summary = llm_processor.summarize_text(full_text, text_chunker)
    return SummarizeResponse(summary=summary)

# --- Utility Endpoints ---
@router.get("/health")
async def health_check():
    return {"status": "healthy"}


