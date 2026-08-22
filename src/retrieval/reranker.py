"""
Cross-Encoder Reranker using FlashRank.
Boosts accuracy of the retrieved chunks by calculating precise relevance scores.
"""
from typing import List
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest
from src.core.logger import get_logger
from src.core.config import get_settings

logger = get_logger("reranker")

class FlashRankReranker:
    def __init__(self, model_name: str = None):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.reranker_model
        logger.info(f"Loading FlashRank model: {self.model_name}")
        self.ranker = Ranker(model_name=self.model_name, cache_dir="./data/flashrank_cache")

    def rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Document]:
        if not documents:
            return []
            
        if top_k is None:
            top_k = self.settings.top_k_rerank
            
        logger.info(f"Reranking {len(documents)} documents for query: '{query}'")
        
        # FlashRank expects a list of dictionaries with 'id' and 'text'
        passages = []
        for i, doc in enumerate(documents):
            passages.append({
                "id": doc.metadata.get("chunk_id", str(i)),
                "text": doc.page_content,
                "meta": doc.metadata
            })
            
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)
        
        # results is a list of dicts with 'id', 'text', 'score' sorted by score desc
        reranked_docs = []
        for res in results[:top_k]:
            meta = res.get("meta", {})
            meta["rerank_score"] = res.get("score")
            doc = Document(
                page_content=res.get("text"),
                metadata=meta
            )
            reranked_docs.append(doc)
            
        logger.info(f"Reranking complete. Selected top {len(reranked_docs)} documents.")
        return reranked_docs
