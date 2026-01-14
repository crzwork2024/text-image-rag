"""
查询增强模块 - RAG 智能问答系统
作者：RAG 项目团队
描述：使用HyDE技术生成假设关键词，提升检索效果
"""

import logging
from typing import Optional
from config import config
from core.llm_client import llm_client
from utils.exceptions import LLMAPIError

logger = logging.getLogger(__name__)


class QueryEnhancer:
    """查询增强器 - 生成假设关键词提升检索效果"""

    def __init__(self):
        """初始化查询增强器"""
        self.llm_client = llm_client

    def generate_hypothetical_keywords(self, query: str) -> Optional[str]:
        """
        基于用户问题生成假设的关键词

        这是HyDE (Hypothetical Document Embeddings) 技术的简化版本。
        生成可能在答案文档中出现的关键术语，用于辅助检索。

        参数:
            query: 用户的原始问题

        返回:
            生成的关键词字符串，如果失败返回 None

        异常:
            捕获所有异常，不影响主流程
        """
        try:
            logger.info("查询增强: 开始生成假设关键词")

            # 构建prompt - 简短高效
            prompt = f"""请基于以下问题，生成3-5个可能出现在答案文档中的专业术语或关键短语。
只输出关键词，用逗号分隔，不要解释，不要编号。

问题：{query}

关键词："""

            # 调用LLM生成
            response = self.llm_client.generate(
                context="",  # 不需要上下文
                user_query=prompt,
                temperature=0.3,  # 较低温度保证稳定性
                max_tokens=100    # 限制长度，降低成本
            )

            keywords = response.strip()
            logger.info(f"查询增强: 生成关键词成功 -> {keywords}")
            return keywords

        except LLMAPIError as e:
            logger.warning(f"查询增强: LLM调用失败，跳过增强 - {e.message}")
            return None

        except Exception as e:
            logger.warning(f"查询增强: 生成关键词失败，跳过增强 - {str(e)}")
            return None

    def is_available(self) -> bool:
        """
        检查查询增强服务是否可用

        返回:
            True 如果LLM可用，否则 False
        """
        return self.llm_client.is_available()


# 全局查询增强器实例（单例模式）
query_enhancer = QueryEnhancer()
