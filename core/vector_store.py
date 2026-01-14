"""
向量数据库模块 - RAG 智能问答系统
作者：RAG 项目团队
描述：管理 ChromaDB 向量数据库，提供文档存储和检索功能
"""

import chromadb
import logging
from typing import List, Dict, Any
from config import config
from utils.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """向量存储管理器 - 负责向量数据库操作"""

    def __init__(self):
        """初始化向量数据库连接"""
        self.client = None
        self.collection = None
        self._initialize_db()

    def _initialize_db(self):
        """
        初始化 ChromaDB 客户端和集合

        异常:
            VectorStoreError: 初始化失败时抛出
        """
        try:
            logger.info(f"正在连接 ChromaDB，路径: {config.CHROMA_PATH}")

            # 创建持久化客户端
            self.client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))

            # 获取或创建集合，使用余弦相似度
            self.collection = self.client.get_or_create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )

            logger.info(f"ChromaDB 连接成功，集合名称: {config.CHROMA_COLLECTION_NAME}")
            logger.info(f"当前集合文档数量: {self.collection.count()}")

        except Exception as e:
            error_msg = f"ChromaDB 初始化失败: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, details=str(e))

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        """
        向向量数据库添加文档

        参数:
            ids: 文档 ID 列表
            embeddings: 文档向量列表
            documents: 文档文本列表
            metadatas: 文档元数据列表

        异常:
            VectorStoreError: 添加文档失败时抛出
        """
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"成功添加 {len(ids)} 个文档到向量数据库")
        except Exception as e:
            error_msg = f"添加文档到向量数据库失败: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, details=str(e))

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10
    ) -> Dict[str, Any]:
        """
        查询向量数据库

        参数:
            query_embeddings: 查询向量列表
            n_results: 返回结果数量

        返回:
            查询结果字典，包含 documents, metadatas, distances

        异常:
            VectorStoreError: 查询失败时抛出
        """
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results
            )
            logger.debug(f"向量查询成功，返回 {n_results} 个结果")
            return results
        except Exception as e:
            error_msg = f"向量数据库查询失败: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, details=str(e))

    def count(self) -> int:
        """
        获取集合中的文档数量

        返回:
            文档数量
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"获取文档数量失败: {e}")
            return 0

    def delete_collection(self):
        """删除当前集合（慎用）"""
        try:
            self.client.delete_collection(name=config.CHROMA_COLLECTION_NAME)
            logger.warning(f"已删除集合: {config.CHROMA_COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"删除集合失败: {e}")

    def reset(self):
        """重置向量数据库（删除并重新创建）"""
        try:
            self.delete_collection()
            self._initialize_db()
            logger.info("向量数据库已重置")
        except Exception as e:
            logger.error(f"重置向量数据库失败: {e}")


# 全局向量数据库实例（单例模式）
vector_db = VectorStoreManager()
