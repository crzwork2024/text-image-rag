"""
Embedding Model Module - RAG Intelligent Q&A System
Author: RAG Project Team
Description: Manages text embedding models, providing text vectorization features.
"""

import logging
from typing import List
from sentence_transformers import SentenceTransformer
from config import config
from utils.exceptions import ModelLoadError, EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Embedding Engine Class - Handles text vectorization"""

    def __init__(self):
        """Initialize Embedding Model"""
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Load local embedding model

        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            logger.info(f"Loading local embedding model: {config.LOCAL_MODEL_PATH}")
            self.model = SentenceTransformer(config.LOCAL_MODEL_PATH)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load embedding model: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg, details=str(e))

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs
    ) -> List[List[float]]:
        """
        Encode text list into vectors

        Args:
            sentences: List of text to encode
            batch_size: Batch size
            show_progress_bar: Whether to show progress bar
            **kwargs: Other arguments passed to model

        Returns:
            List of text vectors (2D list)

        Raises:
            EmbeddingError: If encoding fails
        """
        if not self.model:
            raise EmbeddingError("Embedding model not initialized")

        try:
            # Return list format, compatible with ChromaDB manual insertion
            embeddings = self.model.encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                **kwargs
            )
            return embeddings.tolist()
        except Exception as e:
            error_msg = f"Text encoding failed: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, details=str(e))

    def get_embedding_dimension(self) -> int:
        """
        Get embedding vector dimension

        Returns:
            Vector dimension
        """
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return 0


# Global Embedding Engine Instance (Singleton)
embedding_engine = EmbeddingEngine()
