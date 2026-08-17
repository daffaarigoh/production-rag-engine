# Enterprise Retrieval-Augmented Generation (RAG) Architecture Guide

## 1. Executive Summary
Retrieval-Augmented Generation (RAG) is an architectural pattern that enhances Large Language Models (LLMs) by dynamically retrieving relevant contextual documents from an authoritative external knowledge base before generating responses. This mitigates hallucination, enables real-time domain adaptation, and provides clear source citations.

## 2. Advanced Ingestion & Chunking Strategies
In production RAG systems, naive fixed-size chunking often splits sentences awkwardly, leading to loss of context.
A robust production engine implements **Semantic Recursive Chunking**:
- **Chunk Size**: Typically 500-1000 characters or 200-400 tokens.
- **Overlap**: 10-20% overlap (e.g., 100-150 characters) ensures transitional continuity between sequential chunks.
- **Metadata Tagging**: Every chunk must retain document provenance, including source filename, page number, section header, and a deterministic chunk index (`doc#page#chunk`).

## 3. Hybrid Search & Reranking (RAG 2.0)
Vector search (Dense Retrieval) excels at capturing semantic intent and synonyms but can miss exact keyword codes, product IDs, and specific acronyms.
Therefore, an enterprise RAG system utilizes **Dual-Indexing**:
1. **Dense Retrieval (ChromaDB + BGE Embeddings)**: Captures conceptual meaning.
2. **Sparse Retrieval (BM25)**: Delivers high-precision keyword matching.
3. **Reciprocal Rank Fusion (RRF)**: Merges ranked results from dense and sparse retrievers using the formula:
   $$RRF\_Score(d) = \sum_{r \in R} \frac{1}{k + rank(r, d)}$$ (where $k=60$).
4. **Cross-Encoder Reranking (FlashRank)**: Re-scores top-k candidate chunks against the user query for maximal relevance before passing to the LLM context window.

## 4. Latency and Cost Optimization
Running embeddings locally using lightweight models such as `BAAI/bge-small-en-v1.5` on CPU delivers sub-50ms inference times without consuming expensive remote API credits.
For LLM generation, utilizing high-throughput providers like Groq with Llama 3.3 70B achieves generation speeds exceeding 250 tokens/second.
