"""
Application Configuration using Pydantic Settings.
Loads and validates environment variables from .env file.
"""
from functools import lru_cache
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Type-safe configuration schema for Production RAG Engine.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application Settings
    app_name: str = Field(default="Production RAG Engine", description="Application name")
    app_env: str = Field(default="development", description="Environment: development, staging, production")
    log_level: str = Field(default="INFO", description="Logging verbosity level")

    # LLM Settings
    llm_provider: Literal["groq", "gemini"] = Field(default="groq", description="Active LLM provider")
    llm_model: str = Field(default="llama-3.3-70b-versatile", description="Model identifier")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Sampling temperature")
    llm_max_output_tokens: int = Field(default=2048, ge=128, le=8192, description="Max generation tokens")

    # API Keys
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key")
    google_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")

    # Embeddings & Vector Database
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", description="HuggingFace / FastEmbed model")
    embedding_device: str = Field(default="cpu", description="Compute device: cpu, cuda")
    chroma_persist_dir: str = Field(default="./data/chroma_db", description="ChromaDB persistent directory")
    chroma_collection_name: str = Field(default="rag_documents", description="ChromaDB collection name")

    # Document Chunking
    chunk_size: int = Field(default=800, ge=100, le=4000, description="Chunk size in characters/tokens")
    chunk_overlap: int = Field(default=150, ge=0, le=1000, description="Chunk overlap")

    # Retrieval & Reranker
    top_k_retrieval: int = Field(default=10, ge=1, le=50, description="Initial candidates from hybrid search")
    top_k_rerank: int = Field(default=3, ge=1, le=20, description="Final documents after reranking")
    reranker_model: str = Field(default="ms-marco-TinyBERT-L-2-v2", description="FlashRank cross-encoder model")


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached singleton instance of the application settings.
    """
    return Settings()
