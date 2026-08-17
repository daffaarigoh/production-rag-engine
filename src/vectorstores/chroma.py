"""
ChromaDB Vector Store Manager.
Provides persistent storage, similarity search, and collection stats.
"""
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from src.core.config import get_settings
from src.core.logger import get_logger
from src.embeddings.fastembed import LocalEmbeddingsManager

logger = get_logger("chroma_store")


class ChromaVectorStore:
    """
    Production-ready ChromaDB manager with local persistence.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embeddings_manager: Optional[LocalEmbeddingsManager] = None,
    ):
        settings = get_settings()
        self.persist_directory = Path(persist_directory or settings.chroma_persist_dir).resolve()
        self.collection_name = collection_name or settings.chroma_collection_name
        self.embeddings_manager = embeddings_manager or LocalEmbeddingsManager()
        self.embedding_function = self.embeddings_manager.get_embeddings()

        # Ensure persistence directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self._vector_store = None
        self._init_vector_store()

    def _init_vector_store(self) -> None:
        """Initialize Chroma vector store instance."""
        try:
            from langchain_chroma import Chroma
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=str(self.persist_directory),
            )
            logger.info(
                f"Connected to ChromaDB at '{self.persist_directory}' (Collection: '{self.collection_name}')"
            )
        except ImportError:
            from langchain_community.vectorstores import Chroma
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=str(self.persist_directory),
            )
            logger.info(
                f"Connected to ChromaDB (community) at '{self.persist_directory}' (Collection: '{self.collection_name}')"
            )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add chunked documents to ChromaDB collection.
        Returns the list of assigned IDs.
        """
        if not documents:
            logger.warning("No documents provided to add to ChromaDB.")
            return []

        ids = [doc.metadata.get("chunk_id") for doc in documents]
        # Fallback if chunk_id missing
        ids = [doc_id if doc_id else f"chunk_{i}" for i, doc_id in enumerate(ids)]

        logger.info(f"Indexing [bold cyan]{len(documents)}[/bold cyan] chunks into ChromaDB...")
        assigned_ids = self._vector_store.add_documents(documents=documents, ids=ids)
        logger.info(f"Successfully indexed {len(assigned_ids)} chunks into collection '{self.collection_name}'.")
        return assigned_ids

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Execute vector similarity search for a query.
        """
        return self._vector_store.similarity_search(query=query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Execute vector similarity search returning document and distance score.
        """
        return self._vector_store.similarity_search_with_score(query=query, k=k)

    def get_collection_stats(self) -> dict:
        """
        Return total document count and collection details.
        """
        try:
            count = self._vector_store._collection.count()
            return {
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory),
                "total_chunks": count,
            }
        except Exception as e:
            logger.warning(f"Could not retrieve collection count: {e}")
            return {
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory),
                "total_chunks": "unknown",
            }

    def reset_collection(self) -> None:
        """
        Delete all records in the collection and re-initialize.
        """
        logger.warning(f"Resetting ChromaDB collection: '{self.collection_name}'")
        self._vector_store.delete_collection()
        self._init_vector_store()
