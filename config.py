import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # --- Project Paths ---
    BASE_DIR = Path(__file__).resolve().parent
    MD_FILE_PATH = BASE_DIR / "book.md"
    PARENT_STORE_PATH = BASE_DIR / "parent_store.json"
    CHROMA_PATH = BASE_DIR / "chroma_db"
    STATIC_DIR = BASE_DIR / "static"
    DEBUG_EXPORT_PATH = BASE_DIR / "vector_ingest.json"
    
    # --- Local Embedding Model ---
    LOCAL_MODEL_PATH = r"C:\Users\RONGZHEN CHEN\Desktop\Projects\silian\model\acge_text_embedding"
    
    # --- ChromaDB Settings ---
    CHROMA_COLLECTION_NAME = "book_rag_manual"
    RETRIEVAL_COUNT = 3

    # --- LLM Settings (SiliconFlow) ---
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
    SILICONFLOW_MODEL_ID = os.getenv("SILICONFLOW_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    SILICONFLOW_API_URL = os.getenv("SILICONFLOW_API_URL", "https://api.siliconflow.cn/v1/chat/completions")
    LLM_TEMPERATURE = 0
    
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

    # --- Logging Config ---
    LOG_FORMAT = '[%(asctime)s] - %(levelname)s - [%(name)s] - %(message)s'
    LOG_LEVEL = logging.DEBUG

config = Config()