import json
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from config import config
from core.embeddings import embedding_engine
from core.vector_store import vector_db
from core.llm_client import call_llm
from ingest import run_ingestion

# Setup Logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("API")

app = FastAPI(title="Professional RAG API")
parent_store = {}

class QueryRequest(BaseModel):
    prompt: str

@app.on_event("startup")
def startup_event():
    global parent_store
    
    if config.PARENT_STORE_PATH.exists():
        with open(config.PARENT_STORE_PATH, "r", encoding="utf-8") as f:
            parent_store = json.load(f)
        logger.info("Parent store loaded into memory.")

    if vector_db.count() == 0:
        logger.info("Vector database is empty. Triggering initial ingestion...")
        run_ingestion()
        with open(config.PARENT_STORE_PATH, "r", encoding="utf-8") as f:
            parent_store = json.load(f)

@app.post("/query")
async def query_rag(req: QueryRequest):
    logger.info(f"--- Processing Query: '{req.prompt[:50]}...' ---")
    
    # 1. Embed Query
    query_vec = embedding_engine.encode([req.prompt])
    
    # 2. Retrieve from Chroma
    results = vector_db.query(query_vec, n_results=config.RETRIEVAL_COUNT)
    
    # --- DEBUG LOG: Raw Query Results ---
    # Log the IDs and distances to see how well the vector search performed
    logger.info(f"ChromaDB found {len(results['ids'][0])} matches.")
    logger.debug(f"Raw Search Results: {results}")
    
    metadatas = results["metadatas"][0]
    
    # 3. Reconstruct Context
    unique_hashes = list(dict.fromkeys([m["parent_hash"] for m in metadatas]))
    
    # --- DEBUG LOG: Meta Hashcodes ---
    logger.info(f"Unique Parent Hashes retrieved: {unique_hashes}")
    
    retrieved_sections = []
    for h in unique_hashes:
        section = parent_store.get(h)
        if section:
            retrieved_sections.append(section)
        else:
            logger.warning(f"Hash {h} not found in parent_store!")

    context_text = "\n\n---\n\n".join(retrieved_sections)
    
    # --- DEBUG LOG: Final Context ---
    # We log the length and a snippet of the context
    logger.info(f"Final Context assembled. Total length: {len(context_text)} chars.")
    logger.debug(f"FULL CONTEXT SENT TO LLM:\n{context_text}")
    
    # 4. LLM Generation
    answer = call_llm(context_text, req.prompt)
    
    logger.info("--- Query Flow Complete ---")
    
    return {
        "answer": answer, 
        "sources_count": len(retrieved_sections)
    }

if config.STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(config.STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)