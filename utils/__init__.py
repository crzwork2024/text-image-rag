"""
工具模块 - RAG 智能问答系统
作者：RAG 项目团队
描述：提供通用工具函数和类
"""

from .logger import setup_logger, get_logger
from .exceptions import (
    RAGBaseException,
    ConfigurationError,
    ModelLoadError,
    EmbeddingError,
    VectorStoreError,
    RerankError,
    LLMAPIError,
    DocumentProcessingError,
    RetrievalError
)
from .responses import (
    StandardResponse,
    QueryResponse,
    ErrorResponse,
    success_response,
    error_response
)

__all__ = [
    # 日志
    'setup_logger',
    'get_logger',

    # 异常
    'RAGBaseException',
    'ConfigurationError',
    'ModelLoadError',
    'EmbeddingError',
    'VectorStoreError',
    'RerankError',
    'LLMAPIError',
    'DocumentProcessingError',
    'RetrievalError',

    # 响应
    'StandardResponse',
    'QueryResponse',
    'ErrorResponse',
    'success_response',
    'error_response',
]
