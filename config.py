import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys & URLs for LLM only
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
    SILICONFLOW_MODEL_ID = os.getenv("SILICONFLOW_MODEL_ID")
    SILICONFLOW_API_URL = os.getenv("SILICONFLOW_API_URL")
    
    # Local Paths
    BASE_DIR = Path(__file__).resolve().parent
    MD_FILE_PATH = BASE_DIR / "book.md"
    PARENT_STORE_PATH = BASE_DIR / "parent_store.json"
    CHROMA_PATH = BASE_DIR / "chroma_db"
    STATIC_DIR = BASE_DIR / "static"
    
    # Local Embedding Model Path
    LOCAL_MODEL_PATH = r"C:\Users\RONGZHEN CHEN\Desktop\Projects\silian\model\acge_text_embedding"

config = Config()