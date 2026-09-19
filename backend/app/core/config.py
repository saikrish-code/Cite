"""Core configuration and system settings for CiteRAG."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and environment variable validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # General Application
    PROJECT_NAME: str = "CiteRAG - Research Paper Assistant"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"

    # Server Binding
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        """Convert comma-separated CORS_ORIGINS string into a list of origins.

        Returns:
            list[str]: Cleaned list of allowed origin URLs.
        """
        return [
            origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()
        ]

    # File Upload Limits
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_UPLOAD_EXTENSIONS: str = ".pdf,.docx,.txt"

    @property
    def allowed_extensions_set(self) -> set[str]:
        """Convert comma-separated ALLOWED_UPLOAD_EXTENSIONS into a set of lowercase extensions.

        Returns:
            set[str]: Set of allowed file extensions (e.g. {'.pdf', '.docx', '.txt'}).
        """
        return {
            ext.strip().lower()
            for ext in self.ALLOWED_UPLOAD_EXTENSIONS.split(",")
            if ext.strip()
        }

    @property
    def max_upload_bytes(self) -> int:
        """Convert MAX_UPLOAD_SIZE_MB to bytes.

        Returns:
            int: Maximum upload size in bytes.
        """
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    # Security
    JWT_SECRET_KEY: str = Field(
        default="default-insecure-dev-key-change-in-production-1234567890",
        description="Secret key used for signing JWT tokens",
    )
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # PostgreSQL Database
    DATABASE_URL: str = (
        "postgresql+asyncpg://postgres:postgrespassword@localhost:5432/citerag"
    )

    # ChromaDB
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_data"
    CHROMA_COLLECTION_NAME: str = "citerag_chunks"
    CHROMA_SERVER_HOST: str = "localhost"
    CHROMA_SERVER_PORT: int = 8001

    # Models
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Hybrid Retrieval & Reranker Settings
    RETRIEVAL_MODE: str = "hybrid_rerank"  # "vector_only", "hybrid", or "hybrid_rerank"
    RETRIEVAL_VECTOR_TOP_K: int = 20
    RETRIEVAL_BM25_TOP_K: int = 20
    RRF_K: int = 60
    RERANKER_TOP_K: int = 4
    BM25_K1: float = 1.5
    BM25_B: float = 0.75

    # LLM Settings & API Keys
    LLM_PROVIDER: str = (
        "openai"  # "openai", "anthropic" / "claude", "ollama", or "mock"
    )
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_BASE_URL: str | None = None

    ANTHROPIC_API_KEY: str | None = None
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20241022"
    ANTHROPIC_BASE_URL: str | None = None

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    GEMINI_API_KEY: str | None = None

    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 1024


@lru_cache
def get_settings() -> Settings:
    """Retrieve cached application settings instance.

    Returns:
        Settings: Singleton application settings instance.
    """
    return Settings()


settings = get_settings()
