"""Unit and integration tests for ChromaVectorStore using temporary directories.

Tests cover:
- add_chunks with metadata (user_id, document_id, filename, page, chunk_index)
- similarity_search with k, scoring, and metadata filtering
- multi-attribute filters ($and translation)
- delete_by_document verification
- multi-tenant isolation across users
- temporary filesystem persistence with tmp_path
"""

from __future__ import annotations

import pathlib

import pytest

from app.schemas.document import ChunkMetadata, TextChunk
from app.services.embeddings import (
    BaseEmbeddingService,
    SentenceTransformerEmbeddingService,
)
from app.services.vector_store import BaseVectorStore, ChromaVectorStore


class MockEmbeddingService(BaseEmbeddingService):
    """Deterministic fast embedding service for vector store unit tests.

    Produces predictable 4-dimensional vectors based on keyword matches.
    """

    @property
    def dimension(self) -> int:
        return 4

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        lower = text.lower()
        # Create directional vectors based on keywords
        if "transformer" in lower or "attention" in lower:
            return [1.0, 0.0, 0.0, 0.0]
        elif "convolution" in lower or "cnn" in lower:
            return [0.0, 1.0, 0.0, 0.0]
        elif "database" in lower or "sql" in lower:
            return [0.0, 0.0, 1.0, 0.0]
        return [0.25, 0.25, 0.25, 0.25]


def create_sample_chunk(
    text: str,
    filename: str = "paper.pdf",
    page: int = 1,
    chunk_index: int = 0,
    char_start: int = 0,
    char_end: int = 100,
) -> TextChunk:
    """Helper to create a TextChunk with valid metadata."""
    return TextChunk(
        text=text,
        metadata=ChunkMetadata(
            source_filename=filename,
            page_number=page,
            chunk_index=chunk_index,
            char_start=char_start,
            char_end=char_end,
        ),
    )


@pytest.fixture
def temp_chroma_dir(tmp_path: pathlib.Path) -> str:
    """Provide an isolated temporary directory for ChromaDB persistence."""
    chroma_path = tmp_path / "chroma_test_store"
    chroma_path.mkdir(parents=True, exist_ok=True)
    return str(chroma_path)


@pytest.fixture
def mock_embedder() -> BaseEmbeddingService:
    """Deterministic mock embedder."""
    return MockEmbeddingService()


@pytest.fixture
def vector_store(
    temp_chroma_dir: str, mock_embedder: BaseEmbeddingService
) -> ChromaVectorStore:
    """Vector store initialized in temporary directory with deterministic embeddings."""
    return ChromaVectorStore(
        embedding_service=mock_embedder,
        persist_directory=temp_chroma_dir,
        collection_name="test_collection",
    )


class TestChromaVectorStore:
    """Test suite for ChromaVectorStore functionality."""

    def test_implements_base_interface(self, vector_store: ChromaVectorStore) -> None:
        """ChromaVectorStore must be an instance of abstract BaseVectorStore."""
        assert isinstance(vector_store, BaseVectorStore)

    def test_add_chunks_stores_required_metadata(
        self, vector_store: ChromaVectorStore
    ) -> None:
        """Verify all 5 required metadata fields are stored accurately."""
        chunks = [
            create_sample_chunk(
                text="The Transformer model is based solely on attention mechanisms.",
                filename="attention.pdf",
                page=1,
                chunk_index=0,
            ),
            create_sample_chunk(
                text="Multi-head attention allows the model to jointly attend to information.",
                filename="attention.pdf",
                page=2,
                chunk_index=1,
            ),
        ]

        doc_id = "doc_test_123"
        user_id = "user_alice"

        ids = vector_store.add_chunks(
            chunks=chunks,
            document_id=doc_id,
            user_id=user_id,
        )

        assert len(ids) == 2
        assert ids[0] == f"{doc_id}_0"
        assert ids[1] == f"{doc_id}_1"

        # Direct retrieval verification from Chroma collection
        stored = vector_store.collection.get(
            ids=ids, include=["metadatas", "documents"]
        )
        assert len(stored["ids"]) == 2

        meta_0 = stored["metadatas"][0]
        assert meta_0["user_id"] == "user_alice"
        assert meta_0["document_id"] == "doc_test_123"
        assert meta_0["filename"] == "attention.pdf"
        assert meta_0["page"] == 1
        assert meta_0["chunk_index"] == 0

        meta_1 = stored["metadatas"][1]
        assert meta_1["user_id"] == "user_alice"
        assert meta_1["document_id"] == "doc_test_123"
        assert meta_1["filename"] == "attention.pdf"
        assert meta_1["page"] == 2
        assert meta_1["chunk_index"] == 1

    def test_add_chunks_validation(self, vector_store: ChromaVectorStore) -> None:
        """add_chunks should handle empty chunks and reject blank user_id/document_id."""
        assert vector_store.add_chunks([], document_id="doc1", user_id="user1") == []

        chunk = create_sample_chunk("text")
        with pytest.raises(
            ValueError, match="document_id and user_id must be non-empty"
        ):
            vector_store.add_chunks([chunk], document_id="", user_id="user1")

        with pytest.raises(
            ValueError, match="document_id and user_id must be non-empty"
        ):
            vector_store.add_chunks([chunk], document_id="doc1", user_id="")

    def test_similarity_search_ranking(self, vector_store: ChromaVectorStore) -> None:
        """Semantic search should rank matching chunks highest."""
        chunks = [
            create_sample_chunk(
                text="Transformers replace recurrent neural networks with attention.",
                chunk_index=0,
            ),
            create_sample_chunk(
                text="Convolutional neural networks excel at computer vision tasks.",
                chunk_index=1,
            ),
            create_sample_chunk(
                text="PostgreSQL is a relational database management system.",
                chunk_index=2,
            ),
        ]

        vector_store.add_chunks(chunks=chunks, document_id="doc_topics", user_id="u1")

        # Query about attention / transformers
        results = vector_store.similarity_search(
            "Tell me about Transformer attention", k=2
        )
        assert len(results) == 2

        # The first result should be the transformer chunk
        assert "Transformers" in results[0].text
        assert results[0].metadata.chunk_index == 0
        assert results[0].score >= results[1].score

    def test_similarity_search_metadata_filter(
        self, vector_store: ChromaVectorStore
    ) -> None:
        """Filters must narrow search results to specific documents, users, or pages."""
        chunks = [
            create_sample_chunk(
                text="Attention mechanism page 1", page=1, chunk_index=0
            ),
            create_sample_chunk(
                text="Attention mechanism page 2", page=2, chunk_index=1
            ),
            create_sample_chunk(
                text="Attention mechanism page 3", page=3, chunk_index=2
            ),
        ]
        vector_store.add_chunks(chunks=chunks, document_id="doc_pages", user_id="u1")

        # Filter for page 2 only
        filtered_results = vector_store.similarity_search(
            query="Attention mechanism",
            k=5,
            filters={"page": 2},
        )
        assert len(filtered_results) == 1
        assert filtered_results[0].metadata.page == 2
        assert filtered_results[0].metadata.chunk_index == 1

    def test_multi_attribute_filtering(self, vector_store: ChromaVectorStore) -> None:
        """Testing $and combination of user_id and document_id filters."""
        c1 = create_sample_chunk("Attention paper content doc1", chunk_index=0)
        c2 = create_sample_chunk("Attention paper content doc2", chunk_index=0)

        vector_store.add_chunks([c1], document_id="doc1", user_id="alice")
        vector_store.add_chunks([c2], document_id="doc2", user_id="alice")

        # Both match alice, but filter by document_id="doc2" as well
        results = vector_store.similarity_search(
            query="Attention paper",
            filters={"user_id": "alice", "document_id": "doc2"},
        )
        assert len(results) == 1
        assert results[0].metadata.document_id == "doc2"

    def test_multi_tenant_isolation(self, vector_store: ChromaVectorStore) -> None:
        """User A must never retrieve User B's documents when filtered by user_id."""
        c_alice = create_sample_chunk("Secret financial report of Alice", chunk_index=0)
        c_bob = create_sample_chunk("Secret financial report of Bob", chunk_index=0)

        vector_store.add_chunks([c_alice], document_id="doc_a", user_id="alice")
        vector_store.add_chunks([c_bob], document_id="doc_b", user_id="bob")

        alice_results = vector_store.similarity_search(
            query="Secret financial report",
            filters={"user_id": "alice"},
        )
        assert all(r.metadata.user_id == "alice" for r in alice_results)
        assert not any(r.metadata.user_id == "bob" for r in alice_results)

    def test_delete_by_document(self, vector_store: ChromaVectorStore) -> None:
        """delete_by_document removes all chunks of target document while preserving others."""
        doc1_chunks = [
            create_sample_chunk("Doc 1 chunk 0", chunk_index=0),
            create_sample_chunk("Doc 1 chunk 1", chunk_index=1),
        ]
        doc2_chunks = [
            create_sample_chunk("Doc 2 chunk 0", chunk_index=0),
        ]

        vector_store.add_chunks(doc1_chunks, document_id="doc1", user_id="u1")
        vector_store.add_chunks(doc2_chunks, document_id="doc2", user_id="u1")

        assert vector_store.collection.count() == 3

        deleted_count = vector_store.delete_by_document("doc1")
        assert deleted_count == 2
        assert vector_store.collection.count() == 1

        # Search should only find doc2 now
        remaining = vector_store.collection.get()["ids"]
        assert remaining == ["doc2_0"]

        # Deleting non-existent document returns 0
        assert vector_store.delete_by_document("doc_non_existent") == 0

    def test_empty_query_and_empty_store(self, vector_store: ChromaVectorStore) -> None:
        """Edge cases: empty query string and querying an empty store."""
        # Querying an empty store
        assert vector_store.similarity_search("test") == []

        # Empty query on populated store
        vector_store.add_chunks(
            [create_sample_chunk("Sample text")],
            document_id="d1",
            user_id="u1",
        )
        assert vector_store.similarity_search("") == []
        assert vector_store.similarity_search("   ") == []

    def test_ephemeral_client_initialization(self) -> None:
        """Vector store should support ':memory:' ephemeral in-memory client."""
        store = ChromaVectorStore(
            persist_directory=":memory:",
            collection_name="ephemeral_test",
        )
        assert store.collection.count() == 0


class TestChromaVectorStoreWithRealSentenceTransformer:
    """Integration test combining ChromaVectorStore with real SentenceTransformer model."""

    def test_real_embedding_integration(self, temp_chroma_dir: str) -> None:
        """End-to-end vector store test using the actual SentenceTransformer model."""
        real_embedder = SentenceTransformerEmbeddingService()
        store = ChromaVectorStore(
            embedding_service=real_embedder,
            persist_directory=temp_chroma_dir,
            collection_name="real_embed_test",
        )

        chunks = [
            create_sample_chunk(
                text="The deep neural network utilizes backpropagation with stochastic gradient descent.",
                filename="ml_intro.pdf",
                page=1,
                chunk_index=0,
            ),
            create_sample_chunk(
                text="The French Revolution began in 1789 with the storming of the Bastille.",
                filename="history.pdf",
                page=1,
                chunk_index=1,
            ),
        ]

        ids = store.add_chunks(chunks, document_id="doc_real", user_id="researcher_1")
        assert len(ids) == 2

        results = store.similarity_search(
            "gradient descent optimization in neural networks", k=1
        )
        assert len(results) == 1
        assert "neural network" in results[0].text
        assert results[0].metadata.document_id == "doc_real"
        assert results[0].metadata.user_id == "researcher_1"
        assert results[0].metadata.filename == "ml_intro.pdf"
        assert results[0].score > 0.4
