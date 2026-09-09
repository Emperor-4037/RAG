"""
Centralized application configuration via Pydantic Settings.

All values are configurable through environment variables or a .env file.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- API Server ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    log_level: str = "info"

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_http_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_collection_name: str = "documents"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Celery ---
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # --- Embedding Model (ONNX, CPU) ---
    embedding_model_name: str = "BAAI/bge-base-en-v1.5"
    embedding_dimension: int = 768
    embedding_batch_size: int = 32

    # --- Reranker Model (ONNX, CPU) ---
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- LLM Server (llama.cpp) ---
    llm_base_url: str = "http://localhost:8080"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.1
    llm_context_window: int = 4096

    # --- Retrieval ---
    retrieval_top_k: int = 20
    reranker_top_n: int = 5
    semantic_cache_threshold: float = 0.92

    # --- Rate Limiting ---
    rate_limit_query: str = "30/minute"
    rate_limit_ingest: str = "10/minute"

    # --- Chunking ---
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # --- Observability ---
    otel_service_name: str = "project-rag"

    @property
    def qdrant_url(self) -> str:
        """Construct Qdrant HTTP URL from host and port."""
        return f"http://{self.qdrant_host}:{self.qdrant_http_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor for application settings."""
    return Settings()
