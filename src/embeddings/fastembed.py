"""
Local Embeddings module using FastEmbed / HuggingFace models.
"""
import os
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from src.core.config import get_settings
from src.core.logger import get_logger

# Suppress Windows symlink warning on HuggingFace cache
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logger = get_logger("embeddings")


class LocalEmbeddingsManager:
    """
    Manager for 100% local, CPU-friendly FastEmbed embeddings.
    Default model: BAAI/bge-small-en-v1.5 (Fast, accurate, and lightweight).
    """

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self._embeddings: Optional[Embeddings] = None

    def get_embeddings(self) -> Embeddings:
        """
        Get or initialize the LangChain-compatible embedding instance.
        """
        if self._embeddings is not None:
            return self._embeddings

        logger.info(f"Loading local embedding model: [bold green]{self.model_name}[/bold green]")
        try:
            from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

            self._embeddings = FastEmbedEmbeddings(
                model_name=self.model_name,
                max_length=512,
            )
            logger.info("FastEmbed embedding model initialized successfully.")
            return self._embeddings
        except Exception as e:
            logger.warning(f"FastEmbed failed to initialize ({e}), falling back to HuggingFaceEmbeddings...")
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings

                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                logger.info("HuggingFaceEmbeddings fallback initialized successfully.")
                return self._embeddings
            except Exception as hf_err:
                logger.error(f"Failed to initialize embedding model: {hf_err}")
                raise hf_err
