import os
import tempfile
import requests
from fastapi import HTTPException
from langchain_community.document_loaders import PyPDFLoader
from app.utils.logger import logger

class ContentProcessor:
    """Handle content download and text extraction from various sources."""

    def extract_text_from_content(self, content: bytes, content_type: str) -> str:
        """
        Extracts text from in-memory content (bytes) based on its MIME type.
        This is the core of the 'data pipe'.
        """
        text = ""
        if "application/pdf" in content_type:
            logger.info("Detected PDF bytes, processing with PyPDFLoader...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                loader = PyPDFLoader(temp_file_path)
                pages = loader.load()
                text_parts = [f"\n=== Page {i + 1} ===\n{page.page_content.strip()}\n" for i, page in enumerate(pages) if page.page_content.strip()]
                text = "".join(text_parts)
            except Exception as pdf_err:
                logger.error(f"💥 Failed to parse PDF: {pdf_err}")
                raise HTTPException(status_code=422, detail="Failed to parse PDF content.")
            finally:
                os.unlink(temp_file_path)

        elif "text/plain" in content_type:
            logger.info("Detected plain text bytes, decoding directly.")
            text = content.decode("utf-8")

        else:
            logger.error(f"⚠ Unsupported content type: {content_type}")
            raise HTTPException(status_code=415, detail=f"Unsupported content type: {content_type}")

        if not text.strip():
            raise ValueError("No text could be extracted from the content.")

        cleaned_text = text.replace("\x00", "")
        logger.info(f"Extracted and cleaned {len(cleaned_text)} characters.")
        return cleaned_text.strip()

    def download_and_extract(self, url: str) -> tuple[bytes, str]:
        """
        Downloads content from a URL and returns its raw content (bytes) and content_type.
        """
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            
            content_type = response.headers.get("content-type", "").lower()
            logger.info(f"Downloaded content with type: {content_type}")
            
            # This is the crucial line that returns two values as a tuple
            return response.content, content_type

        except Exception as e:
            logger.error(f"Failed to download content: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process content from URL: {str(e)}")