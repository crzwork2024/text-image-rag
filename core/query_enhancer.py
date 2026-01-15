"""
Query Enhancer Module - RAG Intelligent Q&A System
Author: RAG Project Team
Description: Uses HyDE (Hypothetical Document Embeddings) to generate hypothetical keywords for better retrieval.
"""

import logging
import requests
from typing import Optional
from config import config
from utils.exceptions import LLMAPIError

logger = logging.getLogger(__name__)


class QueryEnhancer:
    """Query Enhancer - Generates hypothetical keywords to improve retrieval"""

    def __init__(self):
        """Initialize Query Enhancer"""
        self.api_key = config.SILICONFLOW_API_KEY
        self.api_url = config.SILICONFLOW_API_URL
        self.model_id = config.QUERY_ENHANCEMENT_MODEL_ID

        if not self.api_key:
            logger.warning("SILICONFLOW_API_KEY not configured, query enhancement unavailable")

    def generate_hypothetical_keywords(self, query: str) -> Optional[str]:
        """
        Generate hypothetical keywords based on user question

        This is a simplified version of HyDE (Hypothetical Document Embeddings).
        Uses a lightweight model to quickly generate key terms, avoiding reasoning model overhead.

        Args:
            query: User's original question

        Returns:
            Generated keywords string, or None if failed

        Raises:
            Captures all exceptions to prevent blocking main flow
        """
        try:
            logger.info(f"Query Enhancer: Generating keywords (Model: {self.model_id})")

            # Optimized prompt for non-reasoning models
            prompt = f"""Output keywords directly, do not explain.

Task: Based on the following question, generate 3-5 professional terms that might appear in the answer document.
Format: Comma separated, start directly.

Question: {query}
Keywords:"""

            # Build request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,  # Low temperature for stability
                "max_tokens": 100,   # Limit length
                "top_p": 0.7         # Reduce randomness
            }

            # Send request
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=10  # Keyword generation should be fast
            )
            response.raise_for_status()

            # Parse response
            result = response.json()
            keywords = result["choices"][0]["message"]["content"].strip()

            # Post-process: Clean up possible "thinking" artifacts and extras
            keywords = self._clean_keywords(keywords)

            logger.info(f"Query Enhancer: Keywords generated -> {keywords}")
            return keywords

        except requests.exceptions.Timeout:
            logger.warning("Query Enhancer: API timeout, skipping enhancement")
            return None

        except requests.exceptions.HTTPError as e:
            logger.warning(f"Query Enhancer: HTTP Error {e.response.status_code}, skipping enhancement")
            return None

        except Exception as e:
            logger.warning(f"Query Enhancer: Generation failed, skipping - {str(e)}")
            return None

    def _clean_keywords(self, keywords: str) -> str:
        """
        Clean keyword output, filtering thoughts and extras

        Args:
            keywords: Raw output

        Returns:
            Cleaned keywords
        """
        # Filter possible think tags
        if "<think>" in keywords or "</think>" in keywords:
            logger.warning("Detected think tags, filtering...")
            keywords = keywords.split("</think>")[-1].strip()

        # Filter common prefixes
        prefixes = ["Keywords:", "Answer:", "A:"]
        for prefix in prefixes:
            if keywords.startswith(prefix):
                keywords = keywords[len(prefix):].strip()

        # Clean punctuation (Chinese/English)
        keywords = keywords.replace("。", ",").replace("、", ",")
        keywords = keywords.replace("；", ",").replace(";", ",")

        # Trim commas and spaces
        keywords = keywords.strip(",").strip()

        # If too short, likely invalid
        if len(keywords) < 3:
            logger.warning(f"Generated keywords too short: '{keywords}', might be poor quality")

        return keywords

    def is_available(self) -> bool:
        """
        Check if query enhancement service is available

        Returns:
            True if configuration is complete, else False
        """
        return bool(self.api_key and self.api_url and self.model_id)


# Global Query Enhancer Instance (Singleton)
query_enhancer = QueryEnhancer()
