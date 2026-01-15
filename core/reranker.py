"""
Rerank Module - RAG Intelligent Q&A System
Author: RAG Project Team
Description: Uses SiliconFlow Rerank API to refine retrieval results.
"""

import requests
import logging
from typing import List, Dict, Any, Optional
from config import config
from utils.exceptions import RerankError

logger = logging.getLogger(__name__)


class RerankEngine:
    """Rerank Engine - Responsible for precise sorting of retrieval results"""

    def __init__(self):
        """Initialize Rerank Engine"""
        self.api_key = config.SILICONFLOW_API_KEY
        self.url = config.SILICONFLOW_RERANK_URL
        self.model = config.SILICONFLOW_RERANK_MODEL

        if not self.api_key:
            logger.warning("SILICONFLOW_API_KEY not configured, rerank feature unavailable")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Rerank a list of documents

        Args:
            query: Query text
            documents: List of documents to rerank
            top_n: Return top N results, defaults to config

        Returns:
            List of rerank results, each element containing index and relevance_score.
            Returns None if rerank fails.

        Raises:
            RerankError: Thrown if rerank fails (optional catch)
        """
        # Validate args
        if not documents:
            logger.warning("Reranker: Candidate document list is empty, skipping")
            return []

        if not self.api_key:
            logger.error("Reranker: Missing API Key, cannot execute rerank")
            return None

        # Use default top_n
        if top_n is None:
            top_n = config.RERANK_TOP_K

        logger.info(
            f"Reranker: Scoring {len(documents)} documents "
            f"(Model: {self.model}, Top-N: {top_n})"
        )

        # Build request
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
            # Send request
            response = requests.post(
                self.url,
                json=payload,
                headers=headers,
                timeout=15
            )
            response.raise_for_status()

            # Parse results
            results = response.json().get("results", [])
            logger.info(f"Reranker: API returned {len(results)} results")

            # Log detailed scores (for debugging)
            for i, res in enumerate(results):
                score = res.get("relevance_score", 0)
                logger.debug(f"  Rank {i+1}: Orig Index {res.get('index')}, Score {score:.4f}")

            return results

        except requests.exceptions.Timeout:
            error_msg = "Reranker: API Request Timeout"
            logger.error(error_msg)
            return None

        except requests.exceptions.HTTPError as e:
            error_msg = f"Reranker: HTTP Error {e.response.status_code}"
            logger.error(f"{error_msg}: {e.response.text}")
            return None

        except Exception as e:
            error_msg = f"Reranker: API Call Failed: {str(e)}"
            logger.error(error_msg)
            return None

    def is_available(self) -> bool:
        """
        Check if rerank service is available

        Returns:
            True if config is complete, else False
        """
        return bool(self.api_key and self.url and self.model)


# Global Rerank Engine Instance (Singleton)
rerank_engine = RerankEngine()
