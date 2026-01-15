"""
配置模块 - RAG 智能问答系统
作者：RAG 项目团队
描述：集中管理所有系统配置参数，包括路径、API 配置、模型参数等
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    """系统配置类 - 管理所有配置参数"""

    # ==================== 项目路径配置 ====================
    BASE_DIR = Path(__file__).resolve().parent
    MD_FILE_PATH = BASE_DIR / "book.md"
    PARENT_STORE_PATH = BASE_DIR / "parent_store.json"
    CHROMA_PATH = BASE_DIR / "chroma_db"
    STATIC_DIR = BASE_DIR / "static"
    IMAGE_DIR = BASE_DIR / "images"
    DEBUG_EXPORT_PATH = BASE_DIR / "vector_ingest.json"
    LOG_DIR = BASE_DIR / "logs"

    # ==================== 本地嵌入模型配置 ====================
    LOCAL_MODEL_PATH = os.getenv(
        "LOCAL_EMBEDDING_MODEL_PATH",
        str(BASE_DIR / "models" / "acge_text_embedding")
    )

    # ==================== 向量数据库配置 ====================
    CHROMA_COLLECTION_NAME = "book_rag_manual"

    # ==================== LLM API 配置（SiliconFlow）====================
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")

    # 主LLM模型（用于最终答案生成）
    SILICONFLOW_MODEL_ID = os.getenv(
        "SILICONFLOW_MODEL_ID",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )

    # 查询增强专用模型（用于关键词生成）
    # 使用轻量级非推理模型，速度快且无思考过程
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

    # ==================== 检索参数配置 ====================
    RETRIEVAL_COUNT = int(os.getenv("RETRIEVAL_COUNT", "10"))        # 向量检索初步召回数量
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))              # 重排后保留的数量
    RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.30")) # Rerank 分数阈值（30%，过滤低质量结果）

    # 向量搜索阈值（不同模式使用不同阈值）
    VECTOR_SEARCH_THRESHOLD_WITH_RERANK = float(os.getenv("VECTOR_SEARCH_THRESHOLD_WITH_RERANK", "0.20"))  # 精排模式：宽松（有Rerank二次过滤）
    VECTOR_SEARCH_THRESHOLD_WITHOUT_RERANK = float(os.getenv("VECTOR_SEARCH_THRESHOLD_WITHOUT_RERANK", "0.50"))  # 直取模式：严格（直接决定质量）

    # 向量搜索阈值（向后兼容，优先级最低）
    VECTOR_SEARCH_THRESHOLD = float(os.getenv("VECTOR_SEARCH_THRESHOLD", "0.20"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))  # 已弃用，保留兼容性

    # ==================== 查询增强配置 ====================
    QUERY_ENHANCEMENT_WEIGHT = float(os.getenv("QUERY_ENHANCEMENT_WEIGHT", "0.6"))  # 原问题权重
    # 关键词权重 = 1 - QUERY_ENHANCEMENT_WEIGHT

    # ==================== Redis 缓存配置 ====================
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))

    # 语义缓存阈值
    CACHE_THRESHOLD_DIRECT = float(os.getenv("CACHE_THRESHOLD_DIRECT", "0.98"))   # 相似度 >= 此值时直接返回缓存
    CACHE_THRESHOLD_CONFIRM = float(os.getenv("CACHE_THRESHOLD_CONFIRM", "0.95")) # 相似度 >= 此值时提示用户确认

    # 缓存配置
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))              # 缓存过期时间（秒），默认1小时
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))   # 最大缓存条目数，超过后 LRU 淘汰

    # ==================== 系统提示词配置 ====================
    SYSTEM_PROMPT = (
        "你是一个严格基于上下文的智能助手。请仅使用提供的上下文内容回答用户问题。\n\n"
        "严格规则：\n"
        "1. 禁止使用外部知识：如果答案不在上下文中，请回答'我没有足够的信息来回答这个问题。'\n"
        "2. 图片引用：如果上下文包含相关图片（例如 ![](images/...) 和图片标题），"
        "你必须在回答中包含准确的图片 markdown 和标题（如果有助于解释主题）。\n"
        "3. 格式化：使用 Markdown 表格进行比较，使用项目符号列出列表。\n"
        "4. 直接回答：不要说'根据文本'之类的话，直接给出答案。"
    )

    # ==================== 应用服务配置 ====================
    APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT = int(os.getenv("APP_PORT", "8000"))
    APP_RELOAD = os.getenv("APP_RELOAD", "True").lower() == "true"

    # ==================== 日志配置 ====================
    LOG_FORMAT = '[%(asctime)s] - %(levelname)s - [%(name)s] - %(message)s'
    LOG_LEVEL = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
    
    # ==================== 管理员配置 ====================
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "")  # SHA256 哈希
    ADMIN_TOKEN_EXPIRE = int(os.getenv("ADMIN_TOKEN_EXPIRE", "3600"))  # 1小时

    @classmethod
    def validate(cls) -> bool:
        """验证配置的有效性"""
        errors = []

        # 检查必需的 API 密钥
        if not cls.SILICONFLOW_API_KEY:
            errors.append("缺少 SILICONFLOW_API_KEY 环境变量")

        # 检查模型路径
        if not Path(cls.LOCAL_MODEL_PATH).exists():
            errors.append(f"本地嵌入模型路径不存在: {cls.LOCAL_MODEL_PATH}")

        # 打印错误并返回结果
        if errors:
            for error in errors:
                logging.error(f"配置验证失败: {error}")
            return False

        return True

    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        directories = [
            cls.CHROMA_PATH,
            cls.IMAGE_DIR,
            cls.LOG_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logging.info(f"确保目录存在: {directory}")


# 全局配置实例
config = Config()
