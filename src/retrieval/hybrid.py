"""
Hybrid Search Retriever combining Vector Search (Chroma) and Sparse Search (BM25)
using Reciprocal Rank Fusion (RRF).
"""
from typing import List
from langchain_core.documents import Document
from src.core.logger import get_logger
from src.vectorstores.chroma import ChromaVectorStore
from src.retrieval.bm25 import BM25Indexer
from src.core.config import get_settings

logger = get_logger("hybrid_retriever")

class HybridRetriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        sparse_indexer: BM25Indexer,
        k: int = 60
    ):
        self.vector_store = vector_store
        self.sparse_indexer = sparse_indexer
        self.rrf_k = k
        self.settings = get_settings()

    def search(self, query: str, top_k: int = None) -> List[Document]:
        if top_k is None:
            top_k = self.settings.top_k_retrieval
            
        logger.info(f"Executing Hybrid Search for query: '{query}'")
        
        # 1. Vector Search
        # langchain-chroma similarity_search_with_score returns L2 distance (lower is better)
        vector_results = self.vector_store.similarity_search_with_score(query, k=top_k)
        
        # 2. Sparse Search
        # BM25 returns score (higher is better)
        sparse_results = self.sparse_indexer.search(query, top_k=top_k)
        
        # Perform Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        doc_map = {}
        
        def process_results(results):
            for rank, (doc, score) in enumerate(results, start=1):
                chunk_id = doc.metadata.get("chunk_id")
                # Fallback to page_content hash if chunk_id is missing
                if not chunk_id:
                    chunk_id = hash(doc.page_content)
                    
                if chunk_id not in doc_map:
                    doc_map[chunk_id] = doc
                    rrf_scores[chunk_id] = 0.0
                    
                # RRF Score formula
                rrf_scores[chunk_id] += 1.0 / (self.rrf_k + rank)
                
        process_results(vector_results)
        process_results(sparse_results)
        
        # Sort by RRF score descending
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for chunk_id, score in sorted_chunks[:top_k]:
            doc = doc_map[chunk_id]
            doc.metadata["rrf_score"] = score
            final_results.append(doc)
            
        logger.info(f"Hybrid search returned {len(final_results)} fused documents.")
        return final_results
