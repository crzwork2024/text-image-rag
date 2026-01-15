"""
Response Module - RAG Intelligent Q&A System
Author: RAG Project Team
Description: Defines standardized API response formats.
"""

from typing import Any, Optional
from pydantic import BaseModel


class StandardResponse(BaseModel):
    """Standard Response Model"""
    success: bool
    message: str
    data: Optional[Any] = None


class QueryResponse(BaseModel):
    """Query Response Model"""
    answer: str
    best_score: str
    sources_count: int
    source_hashes: Optional[list] = None  # List of parent_hashes for source documents
    metadata: Optional[dict] = None


class ErrorResponse(BaseModel):
    """Error Response Model"""
    error: str
    details: Optional[str] = None
    code: Optional[str] = None


def success_response(message: str = "Operation successful", data: Any = None) -> dict:
    """
    Create success response

    Args:
        message: Response message
        data: Response data

    Returns:
        Standardized success response dict
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
    Create error response

    Args:
        error: Error message
        details: Error details
        code: Error code

    Returns:
        Standardized error response dict
    """
    return {
        "success": False,
        "error": error,
        "details": details,
        "code": code
    }
