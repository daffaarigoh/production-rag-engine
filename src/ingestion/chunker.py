"""
Smart Token-Aware Chunking Strategy for Production RAG Engine.
Splits documents using recursive character separators while preserving rich metadata.
"""
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.config import get_settings
from src.core.logger import get_logger

logger = get_logger("document_chunker")


class DocumentChunker:
    """
    Chunking manager that splits documents into semantically coherent segments
    and attaches unique chunk IDs and parent metadata.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            keep_separator=True,
            length_function=len,
        )
        logger.info(
            f"Initialized DocumentChunker (chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})"
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of raw documents into chunks with indexed metadata.
        """
        if not documents:
            logger.warning("Empty document list provided for chunking.")
            return []

        raw_chunks = self.splitter.split_documents(documents)
        enriched_chunks: List[Document] = []

        file_chunk_counter: dict[str, int] = {}

        for chunk in raw_chunks:
            file_name = chunk.metadata.get("file_name", "unknown_doc")
            page_num = chunk.metadata.get("page_number", 1)

            # Generate progressive chunk index per file
            current_idx = file_chunk_counter.get(file_name, 0)
            file_chunk_counter[file_name] = current_idx + 1

            chunk_id = f"{file_name}_p{page_num}_c{current_idx}"

            # Enrich metadata
            metadata = dict(chunk.metadata)
            metadata["chunk_id"] = chunk_id
            metadata["chunk_index"] = current_idx
            metadata["chunk_char_count"] = len(chunk.page_content)

            enriched_chunks.append(
                Document(
                    page_content=chunk.page_content.strip(),
                    metadata=metadata,
                )
            )

        logger.info(
            f"Split [bold blue]{len(documents)}[/bold blue] raw document(s) into [bold green]{len(enriched_chunks)}[/bold green] chunks"
        )
        return enriched_chunks
