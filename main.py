import re
import uuid
import json
import hashlib
import logging
import requests
import sys
import os
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer

# Importing your specific config
from config import config

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# --- 1. Custom Embedding Wrapper (Strictly using your pattern) ---
class CustomEmbeddingWrapper:
    """A wrapper class for SentenceTransformer models to provide a consistent encode interface."""
    def __init__(self, model_instance: SentenceTransformer):
        self.model = model_instance

    def encode(self, sentences: List[str], **kwargs) -> List[List[float]]:
        """Encodes a list of sentences into embeddings."""
        return self.model.encode(sentences, **kwargs).tolist()

# --- 2. Chunking Logic with Debug Saving ---
def get_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def process_and_debug_chunks(md_text: str):
    sections = re.split(r'\n(?=#+ )', md_text)
    vector_items = []
    parent_map = {}

    for section in sections:
        section = section.strip()
        if not section: continue
        
        section_hash = get_hash(section)
        parent_map[section_hash] = section
        
        paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]
        for para in paragraphs:
            if para.startswith('#'): continue
            
            vector_items.append({
                "id": str(uuid.uuid4()),
                "text": para,
                "metadata": {"parent_hash": section_hash}
            })

    # --- Debug Export as requested ---
    logging.info("Saving debug chunk files: vector_ingest.json and parent_store.json")
    with open("vector_ingest.json", "w", encoding="utf-8") as f:
        json.dump(vector_items, f, ensure_ascii=False, indent=2)
    
    # Also save to the path defined in your config for the app to use
    with open(config.PARENT_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(parent_map, f, ensure_ascii=False, indent=2)
        
    return vector_items, parent_map

# --- 3. Initialization ---
app = FastAPI()

# Model Setup using your config.LOCAL_MODEL_PATH
try:
    if os.path.exists(config.LOCAL_MODEL_PATH):
        model_instance = SentenceTransformer(config.LOCAL_MODEL_PATH)
        embedding_wrapper = CustomEmbeddingWrapper(model_instance)
        logging.info(f"Local model loaded successfully from {config.LOCAL_MODEL_PATH}")
    else:
        logging.error(f"Path not found: {config.LOCAL_MODEL_PATH}")
        sys.exit(1)
except Exception as e:
    logging.error(f"Model init error: {e}")
    sys.exit(1)

# Chroma Setup (Manual Mode)
try:
    db_client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    # Note: No embedding_function passed here to avoid 'name' attribute errors
    collection = db_client.get_or_create_collection(name="book_rag_manual")
    logging.info(f"Chroma DB initialized at {config.CHROMA_PATH}")
except Exception as e:
    logging.error(f"Chroma init error: {e}")
    sys.exit(1)

parent_store: Dict[str, str] = {}

@app.on_event("startup")
def ingest_data():
    global parent_store
    
    # Load parent store from your config path
    if config.PARENT_STORE_PATH.exists():
        with open(config.PARENT_STORE_PATH, "r", encoding="utf-8") as f:
            parent_store = json.load(f)

    if collection.count() == 0 and config.MD_FILE_PATH.exists():
        logging.info(f"Starting ingestion from {config.MD_FILE_PATH}...")
        with open(config.MD_FILE_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        
        vector_items, parent_store = process_and_debug_chunks(content)
        
        documents = [item["text"] for item in vector_items]
        
        # --- Strictly Manual Embedding Call ---
        logging.info(f"Generating embeddings for {len(documents)} items...")
        embeddings_list = embedding_wrapper.encode(documents, show_progress_bar=True)
        
        collection.add(
            ids=[item["id"] for item in vector_items],
            embeddings=embeddings_list,
            documents=documents,
            metadatas=[item["metadata"] for item in vector_items]
        )
        logging.info("Ingestion complete and stored in Chroma.")

# --- 4. Query Logic ---
# Define the system prompt as a constant
SYSTEM_PROMPT = (
    "You are a helpful and precise assistant. Use the provided Context to answer the User Question. "
    "Follow these rules strictly:\n"
    "1. Grounding: Answer ONLY using the provided context. If the information is not present, "
    "state clearly that you do not know. Do not use outside knowledge.\n"
    "2. Table Handling: If the context contains relevant data structured as a table, or if the "
    "answer is best presented as a comparison, you MUST format your response using a Markdown table.\n"
    "3. Structure: Use bullet points or numbered lists for multi-step instructions or lists of items.\n"
    "4. Tone & Style: Maintain a professional tone. Provide direct answers and do not "
    "mention 'based on the provided context' or 'according to the text'.\n"
    "5. Formatting: Ensure all Markdown syntax (especially tables and bold text) is clean and valid."
)

def _call_llm(context: str, user_query: str) -> str:
    headers = {"Authorization": f"Bearer {config.SILICONFLOW_API_KEY}"}
    
    # Constructing the message list for a Chat completion
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
    ]
    
    payload = {
        "model": config.SILICONFLOW_MODEL_ID,
        "messages": messages,
        "temperature": 0.1  # Low temperature for factual consistency
    }
    
    try:
        resp = requests.post(config.SILICONFLOW_API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.error(f"LLM call failed: {e}")
        return "Error calling remote LLM."

class QueryRequest(BaseModel):
    prompt: str

@app.post("/query")
async def query_rag(req: QueryRequest):
    # Manual Query Embedding
    query_vec = embedding_wrapper.encode([req.prompt])
    
    results = collection.query(query_embeddings=query_vec, n_results=3)
    metadatas = results["metadatas"][0]
    unique_hashes = list(dict.fromkeys([m["parent_hash"] for m in metadatas]))
    
    retrieved_sections = [parent_store.get(h, "") for h in unique_hashes]
    context_text = "\n\n---\n\n".join(retrieved_sections)
    
    # Pass both arguments to the updated LLM function
    answer = _call_llm(context_text, req.prompt)
    
    return {"answer": answer, "sources_count": len(retrieved_sections)}

# Mount static files using your config path
if config.STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(config.STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # 注意：这里第一个参数必须是 "文件名:对象名" 的字符串格式
    # 如果你的文件名是 main.py，那么就是 "main:app"
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )