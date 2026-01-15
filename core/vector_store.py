"""
Vector Store Module - RAG Intelligent Q&A System
Author: RAG Project Team
Description: Manages ChromaDB vector database, providing document storage and retrieval.
"""

import chromadb
import logging
from typing import List, Dict, Any
from config import config
from utils.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Vector Store Manager - Handles vector database operations"""

    def __init__(self):
        """Initialize Vector Database Connection"""
        self.client = None
        self.collection = None
        self._initialize_db()

    def _initialize_db(self):
        """
        Initialize ChromaDB client and collection

        Raises:
            VectorStoreError: If initialization fails
        """
        try:
            logger.info(f"Connecting to ChromaDB, path: {config.CHROMA_PATH}")

            # Create Persistent Client
            self.client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))

            # Get or create collection, using cosine similarity
            self.collection = self.client.get_or_create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )

            logger.info(f"ChromaDB connected successfully, Collection: {config.CHROMA_COLLECTION_NAME}")
            logger.info(f"Current document count: {self.collection.count()}")

        except Exception as e:
            error_msg = f"ChromaDB initialization failed: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, details=str(e))

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        """
        Add documents to vector database

        Args:
            ids: List of document IDs
            embeddings: List of document vectors
            documents: List of document texts
            metadatas: List of document metadata

        Raises:
            VectorStoreError: If addition fails
        """
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Successfully added {len(ids)} documents to vector database")
        except Exception as e:
            error_msg = f"Failed to add documents to vector database: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, details=str(e))

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10
    ) -> Dict[str, Any]:
        """
        Query the vector database

        Args:
            query_embeddings: List of query vectors
            n_results: Number of results to return

        Returns:
            Query results dict containing documents, metadatas, distances

        Raises:
            VectorStoreError: If query fails
        """
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results
            )
            logger.debug(f"Vector query successful, returned {n_results} results")
            return results
        except Exception as e:
            error_msg = f"Vector database query failed: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg, details=str(e))

    def count(self) -> int:
        """
        Get total document count in collection

        Returns:
            Document count
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0

    def delete_collection(self):
        """Delete current collection (Use with caution)"""
        try:
            self.client.delete_collection(name=config.CHROMA_COLLECTION_NAME)
            logger.warning(f"Deleted collection: {config.CHROMA_COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")

    def reset(self):
        """Reset vector database (Delete and Recreate)"""
        try:
            self.delete_collection()
            self._initialize_db()
            logger.info("Vector database reset")
        except Exception as e:
            logger.error(f"Failed to reset vector database: {e}")


# Global Vector Database Instance (Singleton)
vector_db = VectorStoreManager()
