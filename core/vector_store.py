import chromadb
import logging
from config import config

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
            self.collection = self.client.get_or_create_collection(
                name=config.CHROMA_COLLECTION_NAME
            )
            logger.info(f"Connected to ChromaDB at {config.CHROMA_PATH}")
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            raise

    def add_documents(self, ids, embeddings, documents, metadatas):
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def query(self, query_embeddings: list, n_results: int):
        return self.collection.query(
            query_embeddings=query_embeddings, 
            n_results=n_results
        )

    def count(self):
        return self.collection.count()

vector_db = VectorStoreManager()