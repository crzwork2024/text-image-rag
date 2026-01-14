"""
LLM 客户端模块 - RAG 智能问答系统
作者：RAG 项目团队
描述：与 SiliconFlow LLM API 交互，生成回答
"""

import requests
import logging
from typing import Optional
from config import config
from utils.exceptions import LLMAPIError

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM 客户端 - 负责调用大语言模型生成回答"""

    def __init__(self):
        """初始化 LLM 客户端"""
        self.api_key = config.SILICONFLOW_API_KEY
        self.api_url = config.SILICONFLOW_API_URL
        self.model_id = config.SILICONFLOW_MODEL_ID
        self.temperature = config.LLM_TEMPERATURE

        if not self.api_key:
            logger.warning("未配置 SILICONFLOW_API_KEY，LLM 功能将不可用")

    def generate(
        self,
        context: str,
        user_query: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        生成回答

        参数:
            context: 上下文信息
            user_query: 用户问题
            system_prompt: 系统提示词，默认使用配置值
            temperature: 温度参数，默认使用配置值
            max_tokens: 最大生成token数，默认不限制

        返回:
            生成的回答文本

        异常:
            LLMAPIError: API 调用失败时抛出
        """
        if not self.api_key:
            error_msg = "LLM API 密钥未配置"
            logger.error(error_msg)
            raise LLMAPIError(error_msg)

        # 使用默认值
        if system_prompt is None:
            system_prompt = config.SYSTEM_PROMPT
        if temperature is None:
            temperature = self.temperature

        # 构建消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"上下文:\n{context}\n\n问题: {user_query}"}
        ]

        # 构建请求载荷
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature
        }

        # 如果指定了max_tokens，添加到payload
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            logger.info(f"正在调用 LLM API (模型: {self.model_id})")
            logger.debug(f"问题长度: {len(user_query)} 字符")
            logger.debug(f"上下文长度: {len(context)} 字符")

            # 发送请求
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            # 解析响应
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()

            logger.info(f"LLM 回答生成成功，长度: {len(answer)} 字符")
            return answer

        except requests.exceptions.Timeout:
            error_msg = "LLM API 请求超时"
            logger.error(error_msg)
            raise LLMAPIError(error_msg)

        except requests.exceptions.HTTPError as e:
            error_msg = f"LLM API HTTP 错误: {e.response.status_code}"
            logger.error(f"{error_msg}, 响应: {e.response.text}")
            raise LLMAPIError(error_msg, details=e.response.text)

        except KeyError as e:
            error_msg = f"LLM API 响应格式错误: 缺少字段 {str(e)}"
            logger.error(error_msg)
            raise LLMAPIError(error_msg)

        except Exception as e:
            error_msg = f"LLM API 调用失败: {str(e)}"
            logger.error(error_msg)
            raise LLMAPIError(error_msg, details=str(e))

    def is_available(self) -> bool:
        """
        检查 LLM 服务是否可用

        返回:
            True 如果配置完整，否则 False
        """
        return bool(self.api_key and self.api_url and self.model_id)


# 全局 LLM 客户端实例（单例模式）
llm_client = LLMClient()


# 向后兼容的函数接口
def call_llm(context: str, user_query: str) -> str:
    """
    调用 LLM 生成回答（向后兼容接口）

    参数:
        context: 上下文信息
        user_query: 用户问题

    返回:
        生成的回答文本
    """
    try:
        return llm_client.generate(context, user_query)
    except LLMAPIError as e:
        logger.error(f"LLM 调用失败: {e.message}")
        return "错误：无法生成回答，请稍后重试。"
