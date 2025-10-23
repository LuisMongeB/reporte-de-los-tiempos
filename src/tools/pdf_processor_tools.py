"""
PDFProcessorTools - Tools for processing PDF files

This module provides tools for extracting and analyzing content from PDF files:
- Text extraction using pypdf for text-based PDFs
- Vision LLM processing for scanned/image-based PDFs
- File download from Telegram
"""

import logging
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List
from io import BytesIO

from pypdf import PdfReader
from pdf2image import convert_from_path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from src.core.config import get_config

logger = logging.getLogger(__name__)
settings = get_config()


class PDFProcessorTools:
    """
    Tools for processing PDF files.

    Provides methods for:
    - Extracting text from PDFs using pypdf
    - Processing scanned PDFs using Vision LLM
    - Downloading PDF files from Telegram
    - Converting PDFs to images

    Example usage:
        tools = PDFProcessorTools()

        # Extract text
        result = await tools.extract_text_pypdf("/path/to/file.pdf")
        print(f"Pages: {result['page_count']}, Words: {result['word_count']}")

        # Process with Vision LLM
        result = await tools.extract_with_vision("/path/to/scanned.pdf", max_pages=20)
        print(f"Extracted content: {result['content']}")
    """

    def __init__(self):
        """Initialize PDF processor tools."""
        # Initialize Vision LLM for processing scanned PDFs
        self.vision_llm = ChatOpenAI(
            model=settings.pdf_vision_llm_model,
            api_key=settings.openai_api_key,
            max_tokens=settings.llm_max_tokens * 2,  # More tokens for vision processing
        )

        # Ensure temp directory exists
        settings.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info("PDFProcessorTools initialized successfully")

    async def extract_text_pypdf(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from a PDF file using pypdf library.

        This method extracts all text from all pages of the PDF. Best for
        text-based PDFs. For scanned PDFs, use extract_with_vision instead.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary containing:
                - success (bool): Whether extraction was successful
                - file_path (str): Path to the PDF file
                - page_count (int): Number of pages in PDF
                - text (str): Extracted text content
                - word_count (int): Approximate word count
                - error (str): Error message if extraction failed

        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF processing fails

        Example:
            result = await tools.extract_text_pypdf("document.pdf")
            if result['success']:
                print(f"Extracted {result['word_count']} words from {result['page_count']} pages")
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logger.info(f"Extracting text from PDF: {file_path}")

        try:
            # Read PDF
            reader = PdfReader(file_path)
            page_count = len(reader.pages)

            # Extract text from all pages
            text_parts = []
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"--- Page {page_num} ---\n{page_text}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue

            # Combine all text
            full_text = "\n\n".join(text_parts)

            if not full_text.strip():
                logger.warning(f"No text found in PDF: {file_path}")
                return {
                    'success': False,
                    'file_path': file_path,
                    'page_count': page_count,
                    'text': '',
                    'word_count': 0,
                    'error': 'No text found in PDF. This might be a scanned document - try using Vision LLM.'
                }

            # Calculate word count
            word_count = len(full_text.split())

            logger.info(f"Successfully extracted {word_count} words from {page_count} pages")

            return {
                'success': True,
                'file_path': file_path,
                'page_count': page_count,
                'text': full_text,
                'word_count': word_count,
            }

        except Exception as e:
            logger.error(f"Failed to extract text from PDF {file_path}: {str(e)}")
            return {
                'success': False,
                'file_path': file_path,
                'page_count': 0,
                'text': '',
                'word_count': 0,
                'error': str(e)
            }

    async def extract_with_vision(
        self,
        file_path: str,
        max_pages: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process PDF using Vision LLM (gpt-4o).

        This method converts PDF pages to images and uses Vision LLM to extract
        content. Best for scanned PDFs, forms, or documents with complex layouts.

        Args:
            file_path: Path to the PDF file
            max_pages: Maximum number of pages to process (default from config)

        Returns:
            Dictionary containing:
                - success (bool): Whether processing was successful
                - file_path (str): Path to the PDF file
                - pages_processed (int): Number of pages processed
                - content (str): Extracted content from Vision LLM
                - word_count (int): Approximate word count
                - error (str): Error message if processing failed

        Example:
            result = await tools.extract_with_vision("scanned.pdf", max_pages=20)
            if result['success']:
                print(result['content'])
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        max_pages = max_pages or settings.max_pdf_pages

        logger.info(f"Processing PDF with Vision LLM: {file_path} (max {max_pages} pages)")

        try:
            # Convert PDF pages to images
            logger.info("Converting PDF to images...")
            images = convert_from_path(
                file_path,
                first_page=1,
                last_page=max_pages,
                dpi=200  # Good balance between quality and file size
            )

            pages_processed = len(images)
            logger.info(f"Converted {pages_processed} pages to images")

            # Convert images to base64 for Vision LLM
            image_messages = []
            for i, image in enumerate(images, 1):
                # Convert PIL Image to base64
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                image_messages.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                })

            # Create prompt for Vision LLM
            prompt_text = f"""
Please extract and analyze all text content from these {pages_processed} PDF pages.

Provide:
1. Complete text transcription of all pages
2. Document structure (headings, sections)
3. Key information and main points

Format the extracted text clearly, preserving the document structure.
"""

            # Build message with text + images
            message_content = [{"type": "text", "text": prompt_text}] + image_messages

            # Call Vision LLM
            logger.info("Processing images with Vision LLM...")
            response = self.vision_llm.invoke([
                HumanMessage(content=message_content)
            ])

            content = response.content
            word_count = len(content.split())

            logger.info(f"Vision LLM processed {pages_processed} pages, extracted {word_count} words")

            return {
                'success': True,
                'file_path': file_path,
                'pages_processed': pages_processed,
                'content': content,
                'word_count': word_count,
            }

        except Exception as e:
            logger.error(f"Failed to process PDF with Vision LLM {file_path}: {str(e)}")
            return {
                'success': False,
                'file_path': file_path,
                'pages_processed': 0,
                'content': '',
                'word_count': 0,
                'error': str(e)
            }

    async def download_telegram_file(
        self,
        file_id: str,
        file_name: str,
        telegram_bot_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Download a PDF file from Telegram.

        This is a placeholder for Telegram file download functionality.
        In production, this will use the Telegram Bot API to download files.

        Args:
            file_id: Telegram file ID
            file_name: Name of the file
            telegram_bot_token: Optional bot token (uses settings if not provided)

        Returns:
            Dictionary containing:
                - success (bool): Whether download was successful
                - file_path (str): Path to downloaded file
                - file_size (int): File size in bytes
                - error (str): Error message if download failed

        Note:
            This is a placeholder implementation. Full Telegram integration
            will be implemented in the Telegram handler layer.
        """
        logger.info(f"Downloading Telegram file: {file_id} ({file_name})")

        # This will be implemented when integrating with Telegram bot
        # For now, return placeholder response
        return {
            'success': False,
            'file_path': '',
            'file_size': 0,
            'error': 'Telegram file download not yet implemented. Files should be downloaded by Telegram handler.'
        }

    def cleanup_file(self, file_path: str) -> bool:
        """
        Clean up a temporary PDF file.

        Args:
            file_path: Path to file to delete

        Returns:
            True if file was deleted successfully, False otherwise
        """
        try:
            file_path_obj = Path(file_path)
            if file_path_obj.exists():
                file_path_obj.unlink()
                logger.info(f"Cleaned up temporary file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to cleanup file {file_path}: {e}")
            return False
