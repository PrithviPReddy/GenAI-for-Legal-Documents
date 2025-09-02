from typing import List
import logging

logger = logging.getLogger("hackrx")

def log_document_content(content: str, max_chars: int = 1000):
    logger.info(f"📄 Document content preview ({len(content)} chars):")
    logger.info(content[:max_chars])

def log_chunks_preview(chunks: List[str], max_chunks: int = 3):
    logger.info(f"📦 Created {len(chunks)} chunks. Preview:")
    for i, chunk in enumerate(chunks[:max_chunks]):
        logger.info(f"Chunk {i+1}: {chunk[:200]}...")

def log_search_results(question: str, chunks: List[str], max_results: int = 2):
    logger.info(f"🔍 Search results for '{question[:50]}...': {len(chunks)} chunks found")
    for i, chunk in enumerate(chunks[:max_results]):
        logger.info(f"Result {i+1}: {chunk[:150]}...")
