"""
Configuration Module - RAG Intelligent Q&A System
Author: RAG Project Team
Description: Central management of all system configuration parameters, including paths, API configs, model parameters, etc.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """System Config Class - Manages all configuration parameters"""

    # ==================== Project Path Config ====================
    BASE_DIR = Path(__file__).resolve().parent
    MD_FILE_PATH = BASE_DIR / "book.md"
    PARENT_STORE_PATH = BASE_DIR / "parent_store.json"
    CHROMA_PATH = BASE_DIR / "chroma_db"
    STATIC_DIR = BASE_DIR / "static"
    IMAGE_DIR = BASE_DIR / "images"
    DEBUG_EXPORT_PATH = BASE_DIR / "vector_ingest.json"
    LOG_DIR = BASE_DIR / "logs"
    
    # PDF Conversion Config
    PDF_OUTPUT_DIR = BASE_DIR / "pdf_output"  # MinerU output directory
    PDF_USE_GPU = os.getenv("PDF_USE_GPU", "True").lower() == "true"  # GPU acceleration
    PDF_GPU_ID = int(os.getenv("PDF_GPU_ID", "0"))  # GPU device ID

    # ==================== Local Embedding Model Config ====================
    LOCAL_MODEL_PATH = os.getenv(
        "LOCAL_EMBEDDING_MODEL_PATH",
        str(BASE_DIR / "models" / "acge_text_embedding")
    )

    # ==================== Vector Database Config ====================
    CHROMA_COLLECTION_NAME = "book_rag_manual"

    # ==================== LLM API Config (SiliconFlow) ====================
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")

    # Main LLM Model (for final answer generation)
    SILICONFLOW_MODEL_ID = os.getenv(
        "SILICONFLOW_MODEL_ID",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )

    # Query Enhancement Model (for keyword generation)
    # Uses lightweight non-reasoning model for speed
    QUERY_ENHANCEMENT_MODEL_ID = os.getenv(
        "QUERY_ENHANCEMENT_MODEL_ID",
        "Qwen/Qwen2.5-7B-Instruct"
    )

    SILICONFLOW_API_URL = os.getenv(
        "SILICONFLOW_API_URL",
        "https://api.siliconflow.cn/v1/chat/completions"
    )
    SILICONFLOW_RERANK_URL = os.getenv(
        "SILICONFLOW_RERANK_URL",
        "https://api.siliconflow.cn/v1/rerank"
    )
    SILICONFLOW_RERANK_MODEL = os.getenv(
        "SILICONFLOW_RERANK_MODEL",
        "BAAI/bge-reranker-v2-m3"
    )
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

    # ==================== Retrieval Parameters ====================
    RETRIEVAL_COUNT = int(os.getenv("RETRIEVAL_COUNT", "10"))        # Vector search recall count
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))              # Top K after rerank
    RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.30")) # Rerank score threshold (30%, filter low quality)

    # Vector Search Thresholds (Different modes use different thresholds)
    VECTOR_SEARCH_THRESHOLD_WITH_RERANK = float(os.getenv("VECTOR_SEARCH_THRESHOLD_WITH_RERANK", "0.20"))  # Precision Mode: Loose (Filtered by Rerank)
    VECTOR_SEARCH_THRESHOLD_WITHOUT_RERANK = float(os.getenv("VECTOR_SEARCH_THRESHOLD_WITHOUT_RERANK", "0.50"))  # Fast Mode: Strict (Directly determines quality)

    # Legacy Vector Search Threshold (Lowest priority)
    VECTOR_SEARCH_THRESHOLD = float(os.getenv("VECTOR_SEARCH_THRESHOLD", "0.20"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))  # Deprecated, kept for compatibility

    # ==================== Query Enhancement Config ====================
    QUERY_ENHANCEMENT_WEIGHT = float(os.getenv("QUERY_ENHANCEMENT_WEIGHT", "0.6"))  # Original question weight
    # Keyword weight = 1 - QUERY_ENHANCEMENT_WEIGHT

    # ==================== Redis Cache Config ====================
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))

    # Semantic Cache Thresholds
    CACHE_THRESHOLD_DIRECT = float(os.getenv("CACHE_THRESHOLD_DIRECT", "0.98"))   # Return cache directly if similarity >= this
    CACHE_THRESHOLD_CONFIRM = float(os.getenv("CACHE_THRESHOLD_CONFIRM", "0.95")) # Prompt user to confirm if similarity >= this

    # Cache Config
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))              # Cache TTL (seconds), default 1 hour
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))   # Max cache entries, LRU eviction

    # ==================== System Prompts ====================
    SYSTEM_PROMPT = (
        "You are an intelligent assistant strictly based on context. Please answer user questions using ONLY the provided context content.\n\n"
        "Strict Rules:\n"
        "1. No external knowledge: If the answer is not in the context, say 'I do not have enough information to answer this question.'\n"
        "2. Image citations: If the context contains relevant images (e.g. ![](images/...) and captions), "
        "you MUST include the exact image markdown and caption in your answer (if helpful for explaining the topic).\n"
        "3. Formatting: Use Markdown tables for comparisons, bullet points for lists.\n"
        "4. Direct answer: Do not say 'According to the text' or similar phrases, give the answer directly."
    )

    # ==================== App Service Config ====================
    APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("APP_PORT", "8000"))
    APP_RELOAD = os.getenv("APP_RELOAD", "True").lower() == "true"

    # ==================== Logging Config ====================
    LOG_FORMAT = '[%(asctime)s] - %(levelname)s - [%(name)s] - %(message)s'
    LOG_LEVEL = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
    
    # ==================== Admin Config ====================
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "")  # SHA256 Hash
    ADMIN_TOKEN_EXPIRE = int(os.getenv("ADMIN_TOKEN_EXPIRE", "3600"))  # 1 hour

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        errors = []

        # Check required API keys
        if not cls.SILICONFLOW_API_KEY:
            errors.append("Missing SILICONFLOW_API_KEY environment variable")

        # Check model path
        if not Path(cls.LOCAL_MODEL_PATH).exists():
            errors.append(f"Local embedding model path does not exist: {cls.LOCAL_MODEL_PATH}")

        # Print errors and return result
        if errors:
            for error in errors:
                logging.error(f"Configuration validation failed: {error}")
            return False

        return True

    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.CHROMA_PATH,
            cls.IMAGE_DIR,
            cls.LOG_DIR,
            cls.PDF_OUTPUT_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logging.info(f"Ensured directory exists: {directory}")


# Global Config Instance
config = Config()
