"""Dense text embedding service using sentence-transformers.

Provides an abstract BaseEmbeddingService interface and a concrete
SentenceTransformerEmbeddingService with configurable model selection,
thread-offloaded async methods, batching, and embedding normalization.
"""

from __future__ import annotations

import asyncio
import sys
from abc import ABC, abstractmethod
from typing import Any

# Prevent broken torchcodec from attempting to load non-existent ffmpeg dlls on Windows/Python 3.14
if "torchcodec" not in sys.modules:
    sys.modules["torchcodec"] = None

from app.core.config import settings


class BaseEmbeddingService(ABC):
    """Abstract interface defining the contract for dense text embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality produced by the embedding model.

        Returns:
            int: Number of dimensions in the output embedding vector.
        """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate dense vector embeddings for a list of document chunk texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            list[list[float]]: List of embedding vectors.
        """

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Generate a dense vector embedding for a single search query.

        Args:
            text: Query string to embed.

        Returns:
            list[float]: Embedding vector for the query.
        """

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously generate dense vector embeddings for documents.

        Offloads CPU/GPU-bound tensor computation to a worker thread pool to
        prevent blocking the async event loop.

        Args:
            texts: List of text strings to embed.

        Returns:
            list[list[float]]: List of embedding vectors.
        """
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronously generate a dense vector embedding for a query.

        Offloads CPU/GPU-bound tensor computation to a worker thread pool.

        Args:
            text: Query string to embed.

        Returns:
            list[float]: Embedding vector for the query.
        """
        return await asyncio.to_thread(self.embed_query, text)


class SentenceTransformerEmbeddingService(BaseEmbeddingService):
    """Concrete embedding service leveraging local sentence-transformers models.

    Defaults to BAAI/bge-small-en-v1.5 (configured via settings.EMBEDDING_MODEL_NAME),
    providing high retrieval precision on MTEB benchmarks.

    Args:
        model_name: HuggingFace model identifier or local directory path.
            Defaults to settings.EMBEDDING_MODEL_NAME.
        device: Device to place the model on ('cpu', 'cuda', etc.). Defaults to auto-select.
        batch_size: Number of texts per forward pass during batch embedding. Defaults to 32.
        normalize_embeddings: Whether to L2-normalize vectors for cosine similarity. Defaults to True.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model: Any = None

    @property
    def model(self) -> Any:
        """Lazily initialize and load the SentenceTransformer model on first access.

        Returns:
            SentenceTransformer: Cached instance of the loaded model.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                model_name_or_path=self.model_name,
                device=self.device,
            )
        return self._model

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality of the loaded model.

        Returns:
            int: Dimension size (e.g., 384 for bge-small-en-v1.5).
        """
        if hasattr(self.model, "get_embedding_dimension"):
            dim = self.model.get_embedding_dimension()
        else:
            dim = self.model.get_sentence_embedding_dimension()
        return int(dim)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embeddings for a batch of document texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            list[list[float]]: List of float embedding vectors.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vec.tolist() for vec in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Generate a dense embedding vector for a single query text.

        Args:
            text: Query string.

        Returns:
            list[float]: Embedding vector.

        Raises:
            ValueError: If the query text is empty or blank.
        """
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Query text cannot be empty or whitespace only.")

        embedding = self.model.encode(
            cleaned,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embedding.tolist()
