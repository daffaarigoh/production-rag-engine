"""
Batch Ingestion CLI Script for Production RAG Engine.
Loads raw documents, applies recursive chunking, and builds dual indexes (ChromaDB + BM25).

Usage:
    python -m scripts.ingest [--docs-dir data/sample_docs] [--reset]
"""
import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import get_settings
from src.core.logger import get_logger, setup_logging
from src.ingestion.chunker import DocumentChunker
from src.ingestion.loaders import DocumentLoader
from src.retrieval.bm25 import BM25Indexer
from src.vectorstores.chroma import ChromaVectorStore

try:
    from rich.console import Console
    from rich.table import Table
    RICH_TABLE = True
except ImportError:
    RICH_TABLE = False


def run_ingestion(docs_dir: str, reset: bool = False) -> None:
    """
    Execute full ingestion and dual-indexing pipeline.
    """
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = get_logger("ingestion_pipeline")

    start_time = time.time()
    logger.info("[bold green]Starting Document Ingestion Pipeline...[/bold green]")

    target_dir = Path(docs_dir).resolve()
    if not target_dir.exists():
        logger.error(f"Target directory does not exist: {target_dir}")
        sys.exit(1)

    # 1. Load Documents
    logger.info(f"Scanning directory: '{target_dir}'")
    raw_docs = DocumentLoader.load_directory(target_dir)

    if not raw_docs:
        logger.warning(f"No valid documents found in '{target_dir}'. Exiting.")
        return

    # 2. Chunk Documents
    chunker = DocumentChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    chunks = chunker.split_documents(raw_docs)

    if not chunks:
        logger.warning("No chunks generated from documents. Exiting.")
        return

    # 3. Vector Indexing (ChromaDB)
    vector_store = ChromaVectorStore()
    if reset:
        logger.warning("Flag --reset detected. Clearing existing ChromaDB collection...")
        vector_store.reset_collection()

    vector_store.add_documents(chunks)

    # 4. Sparse Indexing (BM25)
    logger.info("Building BM25 sparse keyword index...")
    bm25_indexer = BM25Indexer()
    bm25_indexer.index_documents(chunks)

    elapsed_time = time.time() - start_time
    stats = vector_store.get_collection_stats()

    # 5. Display Ingestion Summary
    if RICH_TABLE:
        console = Console(highlight=False)
        table = Table(title="[bold]Ingestion Pipeline Summary[/bold]", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=28)
        table.add_column("Value", style="green")

        table.add_row("Source Directory", str(target_dir))
        table.add_row("Raw Documents Loaded", str(len(raw_docs)))
        table.add_row("Total Chunks Created", str(len(chunks)))
        table.add_row("ChromaDB Total Items", str(stats.get("total_chunks")))
        table.add_row("BM25 Index Status", "Persisted (./data/bm25_index.pkl)")
        table.add_row("Embedding Model", settings.embedding_model)
        table.add_row("Total Ingestion Time", f"{elapsed_time:.2f} seconds")

        console.print()
        console.print(table)
        console.print()
    else:
        logger.info(
            f"Ingestion completed in {elapsed_time:.2f}s. Total Chunks: {len(chunks)} | ChromaDB: {stats.get('total_chunks')}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Production RAG Engine")
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="./data/sample_docs",
        help="Directory containing documents to ingest (default: ./data/sample_docs)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing vector collection before indexing",
    )

    args = parser.parse_args()
    run_ingestion(docs_dir=args.docs_dir, reset=args.reset)
