"""
重排模块 - RAG 智能问答系统
作者：RAG 项目团队
描述：使用 SiliconFlow Rerank API 对检索结果进行精排
"""

import requests
import logging
from typing import List, Dict, Any, Optional
from config import config
from utils.exceptions import RerankError

logger = logging.getLogger(__name__)


class RerankEngine:
    """重排引擎 - 负责对检索结果进行精确排序"""

    def __init__(self):
        """初始化重排引擎"""
        self.api_key = config.SILICONFLOW_API_KEY
        self.url = config.SILICONFLOW_RERANK_URL
        self.model = config.SILICONFLOW_RERANK_MODEL

        if not self.api_key:
            logger.warning("未配置 SILICONFLOW_API_KEY，重排功能将不可用")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        对文档列表进行重排

        参数:
            query: 查询文本
            documents: 待重排的文档列表
            top_n: 返回前 N 个结果，默认使用配置值

        返回:
            重排结果列表，每个元素包含 index 和 relevance_score
            如果重排失败，返回 None

        异常:
            RerankError: 重排失败时抛出（可选择是否捕获）
        """
        # 参数验证
        if not documents:
            logger.warning("重排器: 候选文档列表为空，跳过调用")
            return []

        if not self.api_key:
            logger.error("重排器: 缺少 API 密钥，无法执行重排")
            return None

        # 使用默认 top_n
        if top_n is None:
            top_n = config.RERANK_TOP_K

        logger.info(
            f"重排器: 正在对 {len(documents)} 条文档进行深度打分 "
            f"(模型: {self.model}, Top-N: {top_n})"
        )

        # 构建请求
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n
        }

        try:
            # 发送请求
            response = requests.post(
                self.url,
                json=payload,
                headers=headers,
                timeout=15
            )
            response.raise_for_status()

            # 解析结果
            results = response.json().get("results", [])
            logger.info(f"重排器: API 成功返回 {len(results)} 条重排结果")

            # 记录详细分数（调试用）
            for i, res in enumerate(results):
                score = res.get("relevance_score", 0)
                logger.debug(f"  排名 {i+1}: 原索引 {res.get('index')}, 分数 {score:.4f}")

            return results

        except requests.exceptions.Timeout:
            error_msg = "重排器: API 请求超时"
            logger.error(error_msg)
            return None

        except requests.exceptions.HTTPError as e:
            error_msg = f"重排器: HTTP 错误 {e.response.status_code}"
            logger.error(f"{error_msg}: {e.response.text}")
            return None

        except Exception as e:
            error_msg = f"重排器: API 调用失败: {str(e)}"
            logger.error(error_msg)
            return None

    def is_available(self) -> bool:
        """
        检查重排服务是否可用

        返回:
            True 如果配置完整，否则 False
        """
        return bool(self.api_key and self.url and self.model)


# 全局重排引擎实例（单例模式）
rerank_engine = RerankEngine()
