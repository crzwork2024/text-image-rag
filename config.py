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
    IMAGE_DIR = BASE_DIR / "images"
    DEBUG_EXPORT_PATH = BASE_DIR / "vector_ingest.json"
    
    SIMILARITY_THRESHOLD = 0.55
    
    # --- Local Embedding Model ---
    LOCAL_MODEL_PATH = r"C:\Users\RONGZHEN CHEN\Desktop\Projects\silian\model\acge_text_embedding"
    
    # --- ChromaDB Settings ---
    CHROMA_COLLECTION_NAME = "book_rag_manual"
 

    # --- LLM Settings (SiliconFlow) ---
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
    SILICONFLOW_MODEL_ID = os.getenv("SILICONFLOW_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    SILICONFLOW_API_URL = os.getenv("SILICONFLOW_API_URL", "https://api.siliconflow.cn/v1/chat/completions")
    SILICONFLOW_RERANK_URL = os.getenv("SILICONFLOW_RERANK_URL", "https://api.siliconflow.cn/v1/rerank")
    SILICONFLOW_RERANK_MODEL = os.getenv("SILICONFLOW_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    LLM_TEMPERATURE = 0
    
    # config.py 增加以下项
    RETRIEVAL_COUNT = 10          # 向量检索初步召回数量
    RERANK_TOP_K = 3             # 重排后保留的数量
    RERANK_THRESHOLD = 0.01      # Rerank 分数阈值
    VECTOR_SEARCH_THRESHOLD = 0.20      # 向量搜索阈值
    
    SYSTEM_PROMPT = (
        "You are a strict Context-Only Assistant. Answer the User Question in Chinese using ONLY the provided Context.\n\n"
        "STRICT RULES:\n"
        "1. NO OUTSIDE KNOWLEDGE: If the answer isn't in the Context, say 'I do not have enough information.'\n"
        "2. IMAGES: If the context contains relevant images (e.g., ![](images/...) and Figure titles), "
        "you MUST include the exact image markdown and its title in your answer if it helps explain the topic.\n"
        "3. FORMATTING: Use Markdown tables for comparisons. Use bullet points for lists.\n"
        "4. DIRECTNESS: Do not say 'Based on the text'. Give the answer directly."
    )
    

    # --- Logging Config ---
    LOG_FORMAT = '[%(asctime)s] - %(levelname)s - [%(name)s] - %(message)s'
    LOG_LEVEL = logging.INFO

config = Config()