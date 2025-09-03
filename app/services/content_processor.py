import os
import tempfile
import requests
from fastapi import HTTPException
from langchain_community.document_loaders import PyPDFLoader
from app.utils.logger import logger


class ContentProcessor:
    """Handle content download and text extraction, supporting multiple content types."""
    
    @staticmethod
    def download_and_extract(url: str) -> str:
        """Download content from a URL and extract text based on its type."""
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/91.0.4472.124 Safari/537.36"
                )
            }
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            
            content_type = response.headers.get("content-type", "").lower()
            logger.info(f" Downloaded content with type: {content_type}")
            
            text = ""
            # Handle different content types
            if "application/pdf" in content_type:
                logger.info(" Detected PDF, processing with PyPDFLoader...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(response.content)
                    temp_file_path = temp_file.name
                
                try:
                    loader = PyPDFLoader(temp_file_path)
                    pages = loader.load()
                    text_parts = []
                    for i, page in enumerate(pages):
                        page_content = page.page_content.strip()
                        if page_content:
                            text_parts.append(f"\n=== Page {i + 1} ===\n{page_content}\n")
                    text = "".join(text_parts)
                except Exception as pdf_err:
                    logger.error(f"💥 Failed to parse PDF: {pdf_err}")
                    raise HTTPException(status_code=422, detail="Failed to parse PDF content.")
                finally:
                    os.unlink(temp_file_path)


            elif "text/plain" in content_type:
                logger.info(" Detected plain text, processing directly.")
                text = response.text

            else:
                logger.error(f"⚠ Unsupported content type: {content_type}")
                raise HTTPException(status_code=415, detail=f"Unsupported content type: {content_type}")

            if not text.strip():
                raise ValueError("No text could be extracted from the content.")

            # Clean NUL characters
            cleaned_text = text.replace("\x00", "")
            logger.info(f"Extracted and cleaned {len(cleaned_text)} characters.")
            return cleaned_text.strip()

        except HTTPException:
            # Re-raise FastAPI exceptions directly
            raise
        except Exception as e:
            logger.error(f" Failed to download and extract content: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process content from URL: {str(e)}")

