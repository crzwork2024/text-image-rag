"""
Data Ingestion Module - RAG Intelligent Q&A System
Author: RAG Project Team
Description: Handles document ingestion, including chunking, embedding generation, and storage.
"""

import json
import logging
from pathlib import Path
from config import config
from core.processor import process_markdown_to_chunks
from core.embeddings import embedding_engine
from core.vector_store import vector_db
from utils.logger import setup_logger
from utils.exceptions import DocumentProcessingError

# Initialize Logger
logger = setup_logger(
    "Ingestion",
    log_level=config.LOG_LEVEL,
    log_format=config.LOG_FORMAT,
    log_dir=config.LOG_DIR
)


def run_ingestion(
    md_file_path: Path = None,
    force_reingest: bool = False
) -> bool:
    """
    Execute data ingestion workflow

    Workflow:
    1. Read Markdown document
    2. Split document into text chunks
    3. Generate text embeddings
    4. Store in vector database
    5. Save parent node mapping

    Args:
        md_file_path: Markdown file path, defaults to config
        force_reingest: Whether to force re-ingestion (clears existing data)

    Returns:
        True if successful, else False

    Raises:
        DocumentProcessingError: If processing fails
    """
    logger.info("=" * 60)
    logger.info("Starting Data Ingestion Workflow")
    logger.info("=" * 60)

    # Use default path
    if md_file_path is None:
        md_file_path = config.MD_FILE_PATH

    try:
        # Step 1: Verify file exists
        if not md_file_path.exists():
            error_msg = f"Markdown file not found: {md_file_path}"
            logger.error(error_msg)
            raise DocumentProcessingError(error_msg)

        logger.info(f"Reading Markdown file: {md_file_path}")
        file_size = md_file_path.stat().st_size / 1024  # KB
        logger.info(f"File size: {file_size:.2f} KB")

        # Step 2: Read file content
        with open(md_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        logger.info(f"File content length: {len(content)} chars")

        # Step 3: Process document chunks
        logger.info("-" * 60)
        logger.info("Step 1/4: Document Chunking")
        logger.info("-" * 60)

        vector_items, parent_store = process_markdown_to_chunks(content)

        logger.info(f"✓ Generated {len(vector_items)} text chunks")
        logger.info(f"✓ Generated {len(parent_store)} parent node mappings")

        # Step 4: Save parent node mapping
        logger.info("-" * 60)
        logger.info("Step 2/4: Saving Parent Node Mapping")
        logger.info("-" * 60)

        with open(config.PARENT_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(parent_store, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ Parent node mapping saved: {config.PARENT_STORE_PATH}")

        # Step 5: Generate Embeddings
        logger.info("-" * 60)
        logger.info("Step 3/4: Generating Text Embeddings")
        logger.info("-" * 60)

        documents = [item["text"] for item in vector_items]
        logger.info(f"Generating embeddings for {len(documents)} chunks...")

        embeddings = embedding_engine.encode(
            documents,
            batch_size=32,
            show_progress_bar=True
        )

        logger.info(f"✓ Embeddings generated, dimension: {len(embeddings[0]) if embeddings else 0}")

        # Step 6: Store in Vector Database
        logger.info("-" * 60)
        logger.info("Step 4/4: Storing to Vector Database")
        logger.info("-" * 60)

        # If force re-ingest, reset database first
        if force_reingest:
            logger.warning("Force re-ingest mode: Clearing existing data")
            vector_db.reset()

        vector_db.add_documents(
            ids=[item["id"] for item in vector_items],
            embeddings=embeddings,
            documents=documents,
            metadatas=[item["metadata"] for item in vector_items]
        )

        logger.info(f"✓ Data stored in ChromaDB")
        logger.info(f"✓ Current total documents: {vector_db.count()}")

        # Finish
        logger.info("=" * 60)
        logger.info("Data Ingestion Completed!")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"Data Ingestion Failed: {str(e)}")
        logger.error("=" * 60)
        raise


def main():
    """Main function - CLI Entry Point"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Document Ingestion Tool")
    parser.add_argument(
        "--file",
        type=str,
        help="Markdown file path (defaults to config)"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        help="PDF file path (will convert to Markdown first)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingest (clear existing data)"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration for PDF conversion (use CPU)"
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="GPU device ID for PDF conversion (defaults to config)"
    )

    args = parser.parse_args()

    # Handle PDF conversion if needed
    if args.pdf:
        logger.info("PDF file provided, converting to Markdown first...")
        try:
            from pdf_converter import PDFConverter
            
            # Create converter
            use_gpu = not args.no_gpu and config.PDF_USE_GPU
            converter = PDFConverter(use_gpu=use_gpu)
            
            # Convert and prepare
            gpu_id = args.gpu_id if args.gpu_id is not None else config.PDF_GPU_ID
            final_md, final_images = converter.convert_and_prepare(
                args.pdf,
                copy_to_project=True,
                gpu_id=gpu_id
            )
            
            logger.info(f"PDF conversion completed: {final_md}")
            md_file_path = final_md
            
        except ImportError:
            logger.error("PDF conversion requires 'magic-pdf' package. Install with: pip install magic-pdf[full]")
            exit(1)
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            exit(1)
    else:
        # Parse file path
        md_file_path = Path(args.file) if args.file else None

    try:
        success = run_ingestion(
            md_file_path=md_file_path,
            force_reingest=args.force
        )

        if success:
            logger.info("Ingestion successful, system ready")
            exit(0)
        else:
            logger.error("Ingestion failed")
            exit(1)

    except Exception as e:
        logger.error(f"Program exception: {e}")
        exit(1)


if __name__ == "__main__":
    main()
