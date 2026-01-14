"""
查询增强模块 - RAG 智能问答系统
作者：RAG 项目团队
描述：使用HyDE技术生成假设关键词，提升检索效果
"""

import logging
import requests
from typing import Optional
from config import config
from utils.exceptions import LLMAPIError

logger = logging.getLogger(__name__)


class QueryEnhancer:
    """查询增强器 - 生成假设关键词提升检索效果"""

    def __init__(self):
        """初始化查询增强器"""
        self.api_key = config.SILICONFLOW_API_KEY
        self.api_url = config.SILICONFLOW_API_URL
        self.model_id = config.QUERY_ENHANCEMENT_MODEL_ID

        if not self.api_key:
            logger.warning("未配置 SILICONFLOW_API_KEY，查询增强功能将不可用")

    def generate_hypothetical_keywords(self, query: str) -> Optional[str]:
        """
        基于用户问题生成假设的关键词

        这是HyDE (Hypothetical Document Embeddings) 技术的简化版本。
        使用轻量级Qwen模型快速生成关键术语，避免推理模型的思考过程。

        参数:
            query: 用户的原始问题

        返回:
            生成的关键词字符串，如果失败返回 None

        异常:
            捕获所有异常，不影响主流程
        """
        try:
            logger.info(f"查询增强: 开始生成假设关键词 (模型: {self.model_id})")

            # 优化后的prompt - 针对非推理模型
            prompt = f"""直接输出关键词，不要解释。

任务：基于以下问题，生成3-5个可能出现在答案文档中的专业术语。
格式：用逗号分隔，直接开始。

问题：{query}
关键词："""

            # 构建请求
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,  # 极低温度，保证输出稳定
                "max_tokens": 100,   # 限制长度
                "top_p": 0.7         # 降低随机性
            }

            # 发送请求
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=10  # 关键词生成应该很快
            )
            response.raise_for_status()

            # 解析响应
            result = response.json()
            keywords = result["choices"][0]["message"]["content"].strip()

            # 后处理：清理可能的思考过程和多余内容
            keywords = self._clean_keywords(keywords)

            logger.info(f"查询增强: 生成关键词成功 -> {keywords}")
            return keywords

        except requests.exceptions.Timeout:
            logger.warning("查询增强: API 请求超时，跳过增强")
            return None

        except requests.exceptions.HTTPError as e:
            logger.warning(f"查询增强: HTTP 错误 {e.response.status_code}，跳过增强")
            return None

        except Exception as e:
            logger.warning(f"查询增强: 生成关键词失败，跳过增强 - {str(e)}")
            return None

    def _clean_keywords(self, keywords: str) -> str:
        """
        清理关键词输出，过滤思考过程和多余内容

        参数:
            keywords: 原始输出

        返回:
            清理后的关键词
        """
        # 过滤可能的思考标签（如果模型还是输出了）
        if "<think>" in keywords or "</think>" in keywords:
            logger.warning("检测到思考过程标签，正在过滤...")
            keywords = keywords.split("</think>")[-1].strip()

        # 过滤常见的前缀
        prefixes = ["关键词：", "关键词:", "答:", "答：", "A:", "A："]
        for prefix in prefixes:
            if keywords.startswith(prefix):
                keywords = keywords[len(prefix):].strip()

        # 清理多余的标点
        keywords = keywords.replace("。", ",").replace("、", ",")
        keywords = keywords.replace("；", ",").replace(";", ",")

        # 去掉首尾的逗号和空格
        keywords = keywords.strip(",").strip()

        # 如果太短，可能是无效输出
        if len(keywords) < 3:
            logger.warning(f"生成的关键词太短: '{keywords}'，可能质量不佳")

        return keywords

    def is_available(self) -> bool:
        """
        检查查询增强服务是否可用

        返回:
            True 如果配置完整，否则 False
        """
        return bool(self.api_key and self.api_url and self.model_id)


# 全局查询增强器实例（单例模式）
query_enhancer = QueryEnhancer()
