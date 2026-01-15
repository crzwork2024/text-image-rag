"""
LLM Client Module - RAG Intelligent Q&A System
Author: RAG Project Team
Description: Interacts with SiliconFlow LLM API to generate answers.
"""

import requests
import logging
from typing import Optional
from config import config
from utils.exceptions import LLMAPIError

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM Client - Responsible for calling Large Language Model to generate answers"""

    def __init__(self):
        """Initialize LLM Client"""
        self.api_key = config.SILICONFLOW_API_KEY
        self.api_url = config.SILICONFLOW_API_URL
        self.model_id = config.SILICONFLOW_MODEL_ID
        self.temperature = config.LLM_TEMPERATURE

        if not self.api_key:
            logger.warning("SILICONFLOW_API_KEY not configured, LLM feature unavailable")

    def generate(
        self,
        context: str,
        user_query: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate Answer

        Args:
            context: Context information
            user_query: User question
            system_prompt: System prompt, defaults to config
            temperature: Temperature parameter, defaults to config
            max_tokens: Max tokens to generate, defaults to None (unlimited)

        Returns:
            Generated answer text

        Raises:
            LLMAPIError: If API call fails
        """
        if not self.api_key:
            error_msg = "LLM API Key not configured"
            logger.error(error_msg)
            raise LLMAPIError(error_msg)

        # Use defaults
        if system_prompt is None:
            system_prompt = config.SYSTEM_PROMPT
        if temperature is None:
            temperature = self.temperature

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
        ]

        # Build request payload
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature
        }

        # If max_tokens is specified, add to payload
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            logger.info(f"Calling LLM API (Model: {self.model_id})")
            logger.debug(f"Question length: {len(user_query)} chars")
            logger.debug(f"Context length: {len(context)} chars")

            # Send request
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            # Parse response
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()

            logger.info(f"LLM Answer generated successfully, length: {len(answer)} chars")
            return answer

        except requests.exceptions.Timeout:
            error_msg = "LLM API Request Timeout"
            logger.error(error_msg)
            raise LLMAPIError(error_msg)

        except requests.exceptions.HTTPError as e:
            error_msg = f"LLM API HTTP Error: {e.response.status_code}"
            logger.error(f"{error_msg}, Response: {e.response.text}")
            raise LLMAPIError(error_msg, details=e.response.text)

        except KeyError as e:
            error_msg = f"LLM API Response Format Error: Missing field {str(e)}"
            logger.error(error_msg)
            raise LLMAPIError(error_msg)

        except Exception as e:
            error_msg = f"LLM API Call Failed: {str(e)}"
            logger.error(error_msg)
            raise LLMAPIError(error_msg, details=str(e))

    def is_available(self) -> bool:
        """
        Check if LLM service is available

        Returns:
            True if config is complete, else False
        """
        return bool(self.api_key and self.api_url and self.model_id)


# Global LLM Client Instance (Singleton)
llm_client = LLMClient()


# Backwards compatibility interface
def call_llm(context: str, user_query: str) -> str:
    """
    Call LLM to generate answer (Legacy Interface)

    Args:
        context: Context info
        user_query: User question

    Returns:
        Generated answer text
    """
    try:
        return llm_client.generate(context, user_query)
    except LLMAPIError as e:
        logger.error(f"LLM Call Failed: {e.message}")
        return "Error: Unable to generate answer, please try again later."
