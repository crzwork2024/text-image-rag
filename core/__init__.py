"""
核心模块 - RAG 智能问答系统
作者：RAG 项目团队
描述：提供核心功能组件，包括嵌入、向量存储、LLM、重排等
"""

from .embeddings import embedding_engine, EmbeddingEngine
from .vector_store import vector_db, VectorStoreManager
from .llm_client import llm_client, LLMClient, call_llm
from .reranker import rerank_engine, RerankEngine
from .processor import DocumentProcessor, process_markdown_to_chunks

__all__ = [
    # 嵌入
    'embedding_engine',
    'EmbeddingEngine',

    # 向量存储
    'vector_db',
    'VectorStoreManager',

    # LLM
    'llm_client',
    'LLMClient',
    'call_llm',

    # 重排
    'rerank_engine',
    'RerankEngine',

    # 文档处理
    'DocumentProcessor',
    'process_markdown_to_chunks',
]
