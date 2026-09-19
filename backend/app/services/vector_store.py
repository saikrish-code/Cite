"""Vector store abstraction and ChromaDB implementation for CiteRAG.

Provides an abstract BaseVectorStore interface and a concrete ChromaVectorStore
wrapper with support for chunk indexing, metadata persistence (user_id, document_id,
filename, page, chunk_index), multi-attribute filtered similarity search,
and document-level deletion.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from app.core.config import settings
from app.schemas.document import TextChunk, VectorChunkMetadata, VectorSearchResult
from app.services.embeddings import (
    BaseEmbeddingService,
    SentenceTransformerEmbeddingService,
)

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Abstract interface defining operations for vector databases."""

    @abstractmethod
    def add_chunks(
        self,
        chunks: list[TextChunk],
        document_id: str,
        user_id: str,
        embeddings: list[list[float]] | None = None,
    ) -> list[str]:
        """Index text chunks with metadata into the vector store.

        Args:
            chunks: List of TextChunk objects to index.
            document_id: Unique identifier of the parent document.
            user_id: ID of the user owning the document.
            embeddings: Optional precomputed embeddings. If None, generated
                using the configured embedding service.

        Returns:
            list[str]: List of stored chunk identifiers.
        """

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Perform semantic similarity search against indexed chunks.

        Args:
            query: Natural language query string.
            k: Maximum number of top matching chunks to retrieve. Defaults to 4.
            filters: Optional metadata filters (e.g. {'user_id': 'u1', 'document_id': 'd1'}).

        Returns:
            list[VectorSearchResult]: Ranked list of results ordered by similarity score.
        """

    @abstractmethod
    def delete_by_document(self, document_id: str) -> int:
        """Delete all indexed chunks belonging to a specific document.

        Args:
            document_id: Unique identifier of the document to delete.

        Returns:
            int: Number of chunks removed.
        """


class ChromaVectorStore(BaseVectorStore):
    """Concrete vector store wrapper around ChromaDB.

    Supports persistent storage, in-memory ephemeral testing, collection
    partitioning with cosine similarity space, multi-attribute metadata
    filtering, and document-level deletion.

    Args:
        embedding_service: Embedding service instance. Defaults to
            SentenceTransformerEmbeddingService.
        persist_directory: Filesystem path to persist vector data. If None,
            falls back to settings.CHROMA_PERSIST_DIRECTORY.
        collection_name: Chroma collection name. Defaults to
            settings.CHROMA_COLLECTION_NAME.
        client: Optional pre-configured ChromaDB Client (e.g., EphemeralClient).
    """

    def __init__(
        self,
        embedding_service: BaseEmbeddingService | None = None,
        persist_directory: str | None = None,
        collection_name: str | None = None,
        client: Any = None,
    ) -> None:
        import chromadb

        self.embedding_service = (
            embedding_service or SentenceTransformerEmbeddingService()
        )
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME

        if client is not None:
            self._client = client
        elif persist_directory == ":memory:":
            self._client = chromadb.EphemeralClient()
        else:
            storage_path = persist_directory or settings.CHROMA_PERSIST_DIRECTORY
            self._client = chromadb.PersistentClient(path=storage_path)

        self.collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _build_where_clause(
        self, filters: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Transform a flat metadata filter dictionary into a Chroma-valid where clause.

        Chroma requires multiple filter conditions to be combined using the `$and`
        logical operator.

        Args:
            filters: Dictionary of metadata filters (e.g. {'user_id': 'u1', 'page': 2}).

        Returns:
            dict[str, Any] | None: ChromaDB-compatible query clause, or None if no filters.
        """
        if not filters:
            return None

        # If already formatted with logical operators ($and, $or), pass through
        if "$and" in filters or "$or" in filters:
            return filters

        conditions: list[dict[str, Any]] = [
            {key: value} for key, value in filters.items() if value is not None
        ]

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def add_chunks(
        self,
        chunks: list[TextChunk],
        document_id: str,
        user_id: str,
        embeddings: list[list[float]] | None = None,
    ) -> list[str]:
        """Index chunks with provenance metadata in ChromaDB.

        Args:
            chunks: List of TextChunk objects to store.
            document_id: Parent document identifier.
            user_id: Document owner user identifier.
            embeddings: Optional precalculated embeddings.

        Returns:
            list[str]: Stored chunk identifiers.

        Raises:
            ValueError: If chunks list is empty or arguments are invalid.
        """
        if not chunks:
            return []

        if not document_id or not user_id:
            raise ValueError("document_id and user_id must be non-empty strings.")

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, str | int | float | bool]] = []
        seen_ids: set[str] = set()

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_{chunk.metadata.chunk_index}"
            if chunk_id in seen_ids:
                chunk_id = f"{document_id}_{chunk.metadata.page_number}_{chunk.metadata.chunk_index}_{idx}"
            seen_ids.add(chunk_id)

            ids.append(chunk_id)
            documents.append(chunk.text)
            meta = VectorChunkMetadata.from_chunk(
                chunk=chunk,
                document_id=document_id,
                user_id=user_id,
            )
            metadatas.append(meta.to_chroma_dict())

        if embeddings is None:
            embeddings = self.embedding_service.embed_documents(documents)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        logger.info(
            "Successfully indexed %d chunks for document_id=%s (user_id=%s)",
            len(ids),
            document_id,
            user_id,
        )
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Query ChromaDB for semantically similar chunks.

        Args:
            query: Search query text.
            k: Number of results to return.
            filters: Metadata filter dictionary.

        Returns:
            list[VectorSearchResult]: Ordered search results.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        query_embedding = self.embedding_service.embed_query(cleaned_query)
        where_clause = self._build_where_clause(filters)

        total_count = self.collection.count()
        if total_count == 0:
            return []

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(k, total_count),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_clause:
            query_kwargs["where"] = where_clause

        response = self.collection.query(**query_kwargs)

        results: list[VectorSearchResult] = []
        ids_list = response.get("ids", [[]])[0]
        docs_list = response.get("documents", [[]])[0]
        meta_list = response.get("metadatas", [[]])[0]
        dist_list = response.get("distances", [[]])[0]

        for i, chunk_id in enumerate(ids_list):
            text = docs_list[i] if i < len(docs_list) else ""
            meta_dict = meta_list[i] if i < len(meta_list) else {}
            distance = dist_list[i] if i < len(dist_list) else 0.0

            # Cosine distance in Chroma is in [0, 2]; similarity = 1 - distance
            score = max(0.0, 1.0 - float(distance))

            chunk_meta = VectorChunkMetadata(
                user_id=str(meta_dict.get("user_id", "")),
                document_id=str(meta_dict.get("document_id", "")),
                filename=str(meta_dict.get("filename", "")),
                page=int(meta_dict.get("page", 1)),
                chunk_index=int(meta_dict.get("chunk_index", 0)),
                char_start=meta_dict.get("char_start"),
                char_end=meta_dict.get("char_end"),
            )

            results.append(
                VectorSearchResult(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=chunk_meta,
                    score=round(score, 4),
                )
            )

        return results

    def delete_by_document(self, document_id: str) -> int:
        """Purge all chunks associated with a document ID from ChromaDB.

        Args:
            document_id: ID of document whose chunks should be deleted.

        Returns:
            int: Count of deleted chunks.
        """
        if not document_id:
            return 0

        existing = self.collection.get(where={"document_id": document_id})
        existing_ids = existing.get("ids", [])
        count = len(existing_ids)

        if count > 0:
            self.collection.delete(where={"document_id": document_id})
            logger.info(
                "Deleted %d chunks for document_id=%s from collection %s",
                count,
                document_id,
                self.collection_name,
            )

        return count
