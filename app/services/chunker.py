from typing import List
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.utils.logger import logger


class ImprovedTextChunker:
    """Enhanced text chunking with better strategies for legal documents"""
    
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        # Improved separators for legal/constitutional documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=[
                "\n=== Page",    # Page breaks
                "\n\n",         # Paragraph breaks
                "\nArticle",    # Article breaks for constitution
                "\nSection",    # Section breaks
                "\nChapter",    # Chapter breaks
                ".\n",          # Sentence breaks
                "\n",           # Line breaks
                " ",            # Word breaks
                ""              # Character breaks
            ]
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks with improved preprocessing"""
        try:
            # Clean and preprocess text
            cleaned_text = self.preprocess_text(text)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(cleaned_text)
            
            # Post-process chunks
            processed_chunks = []
            for chunk in chunks:
                processed_chunk = self.postprocess_chunk(chunk)
                if len(processed_chunk.strip()) > 100:  # Only keep substantial chunks
                    processed_chunks.append(processed_chunk)
            
            logger.info(f" Created {len(processed_chunks)} processed chunks from text")
            return processed_chunks
            
        except Exception as e:
            logger.error(f" Failed to chunk text: {e}")
            return []
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better chunking"""
        # Remove NUL characters
        text = text.replace('\x00', '')
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Fix common OCR issues (hyphenated words across line breaks)
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        
        # Normalize spacing around articles and sections
        text = re.sub(r'\b(Article|Section|Chapter)\s+(\d+)', r'\n\1 \2', text)
        
        return text.strip()
    
    def postprocess_chunk(self, chunk: str) -> str:
        """Clean up individual chunks"""
        # Remove page markers at the start/end of chunks
        chunk = re.sub(r'^=== Page \d+ ===\s*', '', chunk)
        chunk = re.sub(r'\s*=== Page \d+ ===$', '', chunk)
        
        # Clean up whitespace
        chunk = re.sub(r'\s+', ' ', chunk)
        chunk = chunk.strip()
        
        return chunk

