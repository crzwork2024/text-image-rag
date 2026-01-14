"""
嵌入模型模块 - RAG 智能问答系统
作者：RAG 项目团队
描述：管理文本嵌入模型，提供文本向量化功能
"""

import logging
from typing import List
from sentence_transformers import SentenceTransformer
from config import config
from utils.exceptions import ModelLoadError, EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """嵌入引擎类 - 负责文本向量化"""

    def __init__(self):
        """初始化嵌入模型"""
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        加载本地嵌入模型

        异常:
            ModelLoadError: 模型加载失败时抛出
        """
        try:
            logger.info(f"正在加载本地嵌入模型: {config.LOCAL_MODEL_PATH}")
            self.model = SentenceTransformer(config.LOCAL_MODEL_PATH)
            logger.info("嵌入模型加载成功")
        except Exception as e:
            error_msg = f"嵌入模型加载失败: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg, details=str(e))

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs
    ) -> List[List[float]]:
        """
        将文本列表编码为向量

        参数:
            sentences: 待编码的文本列表
            batch_size: 批处理大小
            show_progress_bar: 是否显示进度条
            **kwargs: 其他参数传递给模型

        返回:
            文本向量列表（二维列表）

        异常:
            EmbeddingError: 编码失败时抛出
        """
        if not self.model:
            raise EmbeddingError("嵌入模型未初始化")

        try:
            # 返回列表格式，符合 ChromaDB 手动插入要求
            embeddings = self.model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                **kwargs
            )
            return embeddings.tolist()
        except Exception as e:
            error_msg = f"文本编码失败: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, details=str(e))

    def get_embedding_dimension(self) -> int:
        """
        获取嵌入向量的维度

        返回:
            向量维度
        """
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return 0


# 全局嵌入引擎实例（单例模式）
embedding_engine = EmbeddingEngine()
