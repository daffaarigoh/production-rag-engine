"""
End-to-End RAG Pipeline combining Retrieval, Reranking, and Generation.
"""
from typing import Dict, Any
import time
from src.core.logger import get_logger
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import FlashRankReranker
from src.generation.llm import LLMManager
from src.generation.prompts import qa_prompt_template, format_docs_for_context

logger = get_logger("rag_pipeline")

class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: FlashRankReranker
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = LLMManager.get_llm()
        self.chain = qa_prompt_template | self.llm
        
    def ask(self, query: str) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline for a given query.
        Returns the answer, source chunks, and latency metadata.
        """
        start_time = time.time()
        logger.info(f"Processing query: '{query}'")
        
        # 1. Hybrid Retrieval
        retrieval_start = time.time()
        initial_docs = self.retriever.search(query)
        retrieval_time = time.time() - retrieval_start
        
        # 2. Reranking
        rerank_start = time.time()
        reranked_docs = self.reranker.rerank(query, initial_docs)
        rerank_time = time.time() - rerank_start
        
        # 3. Generation
        gen_start = time.time()
        context_str = format_docs_for_context(reranked_docs)
        
        response = self.chain.invoke({
            "context": context_str,
            "question": query
        })
        gen_time = time.time() - gen_start
        
        total_time = time.time() - start_time
        
        logger.info(f"Pipeline completed in {total_time:.2f}s (Retrieval: {retrieval_time:.2f}s, Rerank: {rerank_time:.2f}s, Gen: {gen_time:.2f}s)")
        
        return {
            "answer": response.content,
            "source_documents": reranked_docs,
            "metrics": {
                "total_time": round(total_time, 3),
                "retrieval_time": round(retrieval_time, 3),
                "rerank_time": round(rerank_time, 3),
                "generation_time": round(gen_time, 3),
            }
        }
