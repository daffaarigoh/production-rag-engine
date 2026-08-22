import os
import sys
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.logger import get_logger
from src.vectorstores.chroma import ChromaVectorStore
from src.retrieval.bm25 import BM25Indexer
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import FlashRankReranker
from src.generation.rag_chain import RAGPipeline

logger = get_logger("test_day3")

def main():
    load_dotenv()
    
    logger.info("Initializing vector stores...")
    vector_store = ChromaVectorStore()
    
    sparse_indexer = BM25Indexer()
    # Load pre-built index if it exists from Day 2 ingest
    if not sparse_indexer.load():
        logger.warning("BM25 index not found. Did you run the ingest script?")
    
    logger.info("Initializing retrievers and pipeline...")
    hybrid_retriever = HybridRetriever(vector_store, sparse_indexer)
    reranker = FlashRankReranker()
    
    pipeline = RAGPipeline(hybrid_retriever, reranker)
    
    query = "What is the hybrid search strategy and how does it prevent hallucinations?"
    logger.info(f"Test Query: {query}")
    
    try:
        result = pipeline.ask(query)
        print("\n" + "="*50)
        print("🤖 RAG Response:")
        print(result["answer"])
        print("\n" + "="*50)
        print("⏱️ Metrics:")
        for k, v in result["metrics"].items():
            print(f"  - {k}: {v}s")
            
        print("\n📚 Source Documents Used:")
        for doc in result["source_documents"]:
            print(f"  - {doc.metadata.get('chunk_id')} (Rerank Score: {doc.metadata.get('rerank_score', 'N/A')})")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")

if __name__ == "__main__":
    main()
