"""
BM25 Sparse Keyword Indexer for Hybrid Search.
Uses Rank-BM25 with disk persistence.
"""
import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from src.core.logger import get_logger

logger = get_logger("bm25_indexer")


class BM25Indexer:
    """
    BM25 Sparse Retriever for exact keyword matching and term frequency scoring.
    """

    def __init__(self, persistence_path: str = "./data/bm25_index.pkl"):
        self.persistence_path = Path(persistence_path).resolve()
        self.documents: List[Document] = []
        self.bm25: Optional[BM25Okapi] = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple, fast regex-based alphanumeric tokenizer."""
        return re.findall(r"\w+", text.lower())

    def index_documents(self, documents: List[Document]) -> None:
        """
        Build the BM25 index from a list of chunked Document objects.
        """
        if not documents:
            logger.warning("No documents provided to build BM25 index.")
            return

        self.documents = documents
        corpus_tokens = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(corpus_tokens)
        logger.info(f"Built BM25 index with [bold cyan]{len(documents)}[/bold cyan] chunks.")
        self.save()

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        Search for top-k matching documents using BM25 scoring.
        Returns a list of (Document, score) tuples.
        """
        if not self.bm25 or not self.documents:
            if not self.load():
                logger.warning("BM25 index is empty. Please index documents first.")
                return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Pair documents with scores and filter out 0 or negative scores if desired
        doc_score_pairs = list(zip(self.documents, scores))
        # Sort descending by score
        sorted_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        return sorted_pairs[:top_k]

    def save(self, custom_path: Optional[str] = None) -> None:
        """Persist BM25 index and documents to disk."""
        target_path = Path(custom_path).resolve() if custom_path else self.persistence_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with open(target_path, "wb") as f:
            pickle.dump({"documents": self.documents, "bm25": self.bm25}, f)
        logger.info(f"BM25 index saved to: '{target_path}'")

    def load(self, custom_path: Optional[str] = None) -> bool:
        """Load BM25 index and documents from disk."""
        target_path = Path(custom_path).resolve() if custom_path else self.persistence_path
        if not target_path.exists():
            return False

        try:
            with open(target_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data.get("documents", [])
                self.bm25 = data.get("bm25")
            logger.info(f"BM25 index loaded from: '{target_path}' ({len(self.documents)} chunks)")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index from {target_path}: {e}")
            return False
