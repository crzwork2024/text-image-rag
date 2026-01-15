"""
Document Processing Module - RAG Intelligent Q&A System
Author: RAG Project Team
Description: Processes Markdown documents, splitting them into retrievable text chunks.
"""

import re
import uuid
import json
import hashlib
import logging
from typing import List, Dict, Tuple
from pathlib import Path
from config import config
from utils.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)


def get_hash(text: str) -> str:
    """
    Calculate MD5 hash of text

    Args:
        text: Input text

    Returns:
        MD5 hash string
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


class DocumentProcessor:
    """Document Processor - Handles document chunking and preprocessing"""

    @staticmethod
    def process_markdown_to_chunks(
        md_text: str,
        export_debug: bool = True
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Split Markdown document into parent nodes (sections) and child nodes (paragraphs)

        Process Flow:
        1. Split document into sections by headers (Parent Nodes)
        2. Split each section into text chunks by paragraphs (Child Nodes)
        3. Generate hash ID for each parent node
        4. Child nodes store reference to parent node hash

        Args:
            md_text: Markdown document text
            export_debug: Whether to export debug file

        Returns:
            (Vector Items List, Parent Map Dict)
            - Vector Items: List of dicts containing id, text, metadata
            - Parent Map: Mapping from hash to full section text

        Raises:
            DocumentProcessingError: If processing fails
        """
        try:
            logger.info("Starting Markdown document processing")

            # Split document by headers (Parent Nodes)
            # Regex matches Markdown headers: # or ## or ### etc.
            sections = re.split(r'\n(?=#+ )', md_text)
            vector_items = []
            parent_map = {}

            logger.info(f"Document split into {len(sections)} sections")

            for idx, section in enumerate(sections):
                section = section.strip()
                if not section:
                    continue

                # Generate unique hash for section
                section_hash = get_hash(section)
                parent_map[section_hash] = section

                # Split paragraphs by double newlines (Child Nodes)
                paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]

                # Track valid paragraphs
                valid_paragraphs = 0

                for para in paragraphs:
                    # Skip header lines (already included in parent node)
                    if para.startswith('#'):
                        continue

                    # Skip paragraphs that are too short (likely noise)
                    if len(para) < 10:
                        continue

                    # Create vector item
                    vector_items.append({
                        "id": str(uuid.uuid4()),
                        "text": para,
                        "metadata": {
                            "parent_hash": section_hash,
                            "section_index": idx
                        }
                    })
                    valid_paragraphs += 1

                logger.debug(
                    f"Section {idx + 1}: "
                    f"Extracted {valid_paragraphs} valid paragraphs "
                    f"(Hash: {section_hash[:8]}...)"
                )

            logger.info(f"Processing complete: {len(vector_items)} chunks, {len(parent_map)} sections")

            # Export debug file
            if export_debug:
                DocumentProcessor._export_debug_chunks(vector_items)

            return vector_items, parent_map

        except Exception as e:
            error_msg = f"Document processing failed: {str(e)}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg, details=str(e))

    @staticmethod
    def _export_debug_chunks(vector_items: List[Dict]):
        """
        Export chunks to debug file

        Args:
            vector_items: List of vector items
        """
        try:
            debug_path = config.DEBUG_EXPORT_PATH
            logger.info(f"Exporting debug file to: {debug_path}")

            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(vector_items, f, ensure_ascii=False, indent=2)

            logger.debug(f"Debug file exported successfully, size: {Path(debug_path).stat().st_size} bytes")

        except Exception as e:
            logger.warning(f"Failed to export debug file: {e}")

    @staticmethod
    def validate_markdown(md_text: str) -> bool:
        """
        Validate Markdown document

        Args:
            md_text: Markdown text

        Returns:
            True if valid, False otherwise
        """
        if not md_text or len(md_text.strip()) == 0:
            logger.error("Markdown document is empty")
            return False

        # Check for headers
        if not re.search(r'^#+ ', md_text, re.MULTILINE):
            logger.warning("Markdown document contains no headers")
            return False

        return True


# Backwards compatibility interface
def process_markdown_to_chunks(md_text: str) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Process Markdown document into chunks (Legacy Interface)

    Args:
        md_text: Markdown document text

    Returns:
        (Vector Items List, Parent Map Dict)
    """
    return DocumentProcessor.process_markdown_to_chunks(md_text)
