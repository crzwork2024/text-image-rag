"""
响应模块 - RAG 智能问答系统
作者：RAG 项目团队
描述：定义标准化的 API 响应格式
"""

from typing import Any, Optional
from pydantic import BaseModel


class StandardResponse(BaseModel):
    """标准响应模型"""
    success: bool
    message: str
    data: Optional[Any] = None


class QueryResponse(BaseModel):
    """查询响应模型"""
    answer: str
    best_score: str
    sources_count: int
    metadata: Optional[dict] = None


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str
    details: Optional[str] = None
    code: Optional[str] = None


def success_response(message: str = "操作成功", data: Any = None) -> dict:
    """
    创建成功响应

    参数:
        message: 响应消息
        data: 响应数据

    返回:
        标准化的成功响应字典
    """
    return {
        "success": True,
        "message": message,
        "data": data
    }


def error_response(
    error: str,
    details: Optional[str] = None,
    code: Optional[str] = None
) -> dict:
    """
    创建错误响应

    参数:
        error: 错误消息
        details: 错误详情
        code: 错误代码

    返回:
        标准化的错误响应字典
    """
    return {
        "success": False,
        "error": error,
        "details": details,
        "code": code
    }
