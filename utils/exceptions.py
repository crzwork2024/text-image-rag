"""
Exceptions Module - RAG Intelligent Q&A System
Author: RAG Project Team
Description: Defines custom exception classes used in the system.
"""


class RAGBaseException(Exception):
    """Base exception class for RAG system"""
    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class ConfigurationError(RAGBaseException):
    """Configuration Error"""
    pass


class ModelLoadError(RAGBaseException):
    """Model Load Error"""
    pass


class EmbeddingError(RAGBaseException):
    """Embedding Generation Error"""
    pass


class VectorStoreError(RAGBaseException):
    """Vector Store Error"""
    pass


class RerankError(RAGBaseException):
    """Rerank Error"""
    pass


class LLMAPIError(RAGBaseException):
    """LLM API Call Error"""
    pass


class DocumentProcessingError(RAGBaseException):
    """Document Processing Error"""
    pass


class RetrievalError(RAGBaseException):
    """Retrieval Error"""
    pass
