import hashlib
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Global in-memory cache
document_cache: Dict[str, Dict[str, Any]] = {}


def get_url_hash(url: str) -> str:
    """Create a short hash from the document URL (so long URLs don’t blow up the dict keys)."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def get_cached_document(url: str) -> Optional[Dict[str, Any]]:
    """Return cached entry if exists."""
    url_hash = get_url_hash(url)
    return document_cache.get(url_hash)


def cache_document(url: str, document_id: str, chunks: List[str]):
    """Save document ID + chunks in cache."""
    url_hash = get_url_hash(url)
    document_cache[url_hash] = {
        "document_id": document_id,
        "chunks": chunks,
        "cached_at": datetime.utcnow().isoformat()
    }
    logger.info(f"🗂️ Cached document {url} with ID {document_id}")


def cache_stats() -> Dict[str, Any]:
    """Return cache statistics."""
    return {
        "cached_documents": len(document_cache),
        "documents": list(document_cache.keys())
    }
