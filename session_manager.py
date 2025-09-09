import uuid
from typing import Optional, Dict, Any

# This dictionary will act as our simple, in-memory session storage.
# It maps a session_id to the data for that session.
SESSION_STORAGE: Dict[str, Dict[str, Any]] = {}

def get_session_data(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves the data for a given session ID."""
    return SESSION_STORAGE.get(session_id)

def update_session_data(session_id: str, document_id: str, full_text: str):
    """Stores or updates the data for a given session ID."""
    SESSION_STORAGE[session_id] = {
        "document_id": document_id,
        "full_text": full_text
    }

def get_or_create_session_id(session_cookie: Optional[str]) -> str:
    """
    Returns the existing session ID from the cookie or creates a new one.
    """
    if session_cookie and session_cookie in SESSION_STORAGE:
        return session_cookie
    return str(uuid.uuid4())
