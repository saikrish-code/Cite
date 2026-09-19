"""Unit and integration tests for HybridRetriever, stage toggling, latency profiling, and tenant isolation."""

from unittest.mock import MagicMock

import pytest

from app.schemas.document import VectorChunkMetadata, VectorSearchResult
from app.services.bm25 import BM25SearchService
from app.services.hybrid_retriever import HybridRetriever
from app.services.reranker import MockReranker
from app.services.vector_store import BaseVectorStore


def _make_result(chunk_id: str, text: str, user_id: str = "u1", score: float = 0.5) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        text=text,
        metadata=VectorChunkMetadata(
            document_id="d1",
            user_id=user_id,
            filename="paper.txt",
            page=1,
            chunk_index=0,
        ),
        score=score,
    )


class MockVectorStore(BaseVectorStore):
    """In-memory vector store mock for fast hybrid retriever testing."""

    def __init__(self, corpus: list[VectorSearchResult]) -> None:
        self.corpus = corpus

    def add_chunks(self, chunks, document_id, user_id, embeddings=None):
        return []

    def similarity_search(self, query, k=4, filters=None):
        scoped = self.get_chunks(filters)
        return scoped[:k]

    def delete_by_document(self, document_id, user_id=None):
        return 0

    def get_chunks(self, filters=None):
        if not filters:
            return self.corpus
        user_id = filters.get("user_id")
        if user_id:
            return [c for c in self.corpus if c.metadata.user_id == user_id]
        return self.corpus


class TestHybridRetriever:
    """Tests for HybridRetriever coordinator."""

    @pytest.fixture
    def sample_corpus(self) -> list[VectorSearchResult]:
        return [
            _make_result("v1", "Dense semantic passage on transformers attention.", user_id="alice", score=0.9),
            _make_result("v2", "FlashAttention memory IO complexity on A100 GPU.", user_id="alice", score=0.8),
            _make_result("v3", "Standard convolutional neural network architecture.", user_id="alice", score=0.6),
            _make_result("b1", "Bob private notes on quantum circuits.", user_id="bob", score=0.95),
        ]

    @pytest.fixture
    def retriever(self, sample_corpus: list[VectorSearchResult]) -> HybridRetriever:
        vstore = MockVectorStore(sample_corpus)
        bm25 = BM25SearchService()
        reranker = MockReranker()
        return HybridRetriever(
            vector_store=vstore,
            bm25_service=bm25,
            reranker=reranker,
            default_mode="hybrid_rerank",
            vector_top_k=5,
            bm25_top_k=5,
            reranker_top_k=2,
        )

    def test_mode_vector_only(self, retriever: HybridRetriever) -> None:
        results, metrics = retriever.retrieve(
            query="FlashAttention GPU",
            k=2,
            filters={"user_id": "alice"},
            mode="vector_only",
        )

        assert metrics.mode == "vector_only"
        assert metrics.vector_ms >= 0.0
        assert metrics.bm25_ms == 0.0
        assert metrics.rrf_ms == 0.0
        assert metrics.rerank_ms == 0.0
        assert metrics.total_ms >= 0.0
        assert len(results) <= 2
        for r in results:
            assert r.metadata.user_id == "alice"

    def test_mode_hybrid(self, retriever: HybridRetriever) -> None:
        results, metrics = retriever.retrieve(
            query="FlashAttention GPU",
            k=2,
            filters={"user_id": "alice"},
            mode="hybrid",
        )

        assert metrics.mode == "hybrid"
        assert metrics.vector_ms >= 0.0
        assert metrics.bm25_ms >= 0.0
        assert metrics.rrf_ms >= 0.0
        assert metrics.rerank_ms == 0.0
        assert metrics.total_ms >= 0.0
        assert len(results) <= 2
        # v2 matches both dense and BM25 keywords, so it should rank first after RRF
        assert results[0].chunk_id == "v2"

    def test_mode_hybrid_rerank(self, retriever: HybridRetriever) -> None:
        results, metrics = retriever.retrieve(
            query="FlashAttention GPU",
            k=2,
            filters={"user_id": "alice"},
            mode="hybrid_rerank",
        )

        assert metrics.mode == "hybrid_rerank"
        assert metrics.vector_ms >= 0.0
        assert metrics.bm25_ms >= 0.0
        assert metrics.rrf_ms >= 0.0
        assert metrics.rerank_ms >= 0.0
        assert metrics.total_ms >= 0.0
        assert len(results) <= 2
        assert results[0].chunk_id == "v2"

    def test_strict_tenant_isolation_across_all_stages(self, retriever: HybridRetriever) -> None:
        # User Bob's query for Alice's topics should NEVER retrieve Alice's chunks
        results_bob, _ = retriever.retrieve(
            query="FlashAttention GPU memory",
            k=4,
            filters={"user_id": "bob"},
            mode="hybrid_rerank",
        )

        for res in results_bob:
            assert res.metadata.user_id == "bob"
            assert res.chunk_id != "v2"

        # User Alice's query should NEVER retrieve Bob's chunks
        results_alice, _ = retriever.retrieve(
            query="quantum circuits",
            k=4,
            filters={"user_id": "alice"},
            mode="hybrid_rerank",
        )

        for res in results_alice:
            assert res.metadata.user_id == "alice"
            assert res.chunk_id != "b1"
