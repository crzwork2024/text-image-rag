import logging
from typing import List
from sentence_transformers import SentenceTransformer
from config import config

logger = logging.getLogger(__name__)

class CustomEmbeddingWrapper:
    def __init__(self):
        try:
            logger.info(f"Loading local embedding model from: {config.LOCAL_MODEL_PATH}")
            self.model = SentenceTransformer(config.LOCAL_MODEL_PATH)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def encode(self, sentences: List[str], **kwargs) -> List[List[float]]:
        # Returns list of lists as required by Chroma manual insertion
        return self.model.encode(sentences, **kwargs).tolist()

# Singleton instance
embedding_engine = CustomEmbeddingWrapper()