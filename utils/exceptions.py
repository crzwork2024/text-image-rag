"""
异常处理模块 - RAG 智能问答系统
作者：RAG 项目团队
描述：定义系统中使用的自定义异常类
"""


class RAGBaseException(Exception):
    """RAG 系统基础异常类"""
    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class ConfigurationError(RAGBaseException):
    """配置错误异常"""
    pass


class ModelLoadError(RAGBaseException):
    """模型加载错误异常"""
    pass


class EmbeddingError(RAGBaseException):
    """嵌入生成错误异常"""
    pass


class VectorStoreError(RAGBaseException):
    """向量数据库错误异常"""
    pass


class RerankError(RAGBaseException):
    """重排错误异常"""
    pass


class LLMAPIError(RAGBaseException):
    """LLM API 调用错误异常"""
    pass


class DocumentProcessingError(RAGBaseException):
    """文档处理错误异常"""
    pass


class RetrievalError(RAGBaseException):
    """检索错误异常"""
    pass
