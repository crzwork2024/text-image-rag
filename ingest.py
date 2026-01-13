import json
import logging
from config import config
from core.processor import process_markdown_to_chunks
from core.embeddings import embedding_engine
from core.vector_store import vector_db

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("Ingestion")

def run_ingestion():
    if not config.MD_FILE_PATH.exists():
        logger.error(f"Markdown file not found at {config.MD_FILE_PATH}")
        return

    logger.info("Reading Markdown file...")
    with open(config.MD_FILE_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    vector_items, parent_store = process_markdown_to_chunks(content)
    
    # Save parent store
    with open(config.PARENT_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(parent_store, f, ensure_ascii=False, indent=2)
    
    # Generate Embeddings
    documents = [item["text"] for item in vector_items]
    logger.info(f"Generating embeddings for {len(documents)} chunks...")
    embeddings = embedding_engine.encode(documents, show_progress_bar=True)
    
    # Store in Chroma
    logger.info("Storing in ChromaDB...")
    vector_db.add_documents(
        ids=[item["id"] for item in vector_items],
        embeddings=embeddings,
        documents=documents,
        metadatas=[item["metadata"] for item in vector_items]
    )
    logger.info("Ingestion process completed successfully.")

if __name__ == "__main__":
    run_ingestion()