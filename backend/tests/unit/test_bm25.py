"""Unit tests for sparse BM25 keyword search service."""

import pytest

from app.schemas.document import VectorChunkMetadata, VectorSearchResult
from app.services.bm25 import BM25SearchService, tokenize_text


def _make_result(chunk_id: str, text: str, user_id: str = "u1", doc_id: str = "d1") -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        text=text,
        metadata=VectorChunkMetadata(
            document_id=doc_id,
            user_id=user_id,
            filename=f"{doc_id}.txt",
            page=1,
            chunk_index=0,
        ),
        score=1.0,
    )


class TestBM25Tokenizer:
    """Tests for word tokenization logic."""

    def test_tokenize_clean_words(self) -> None:
        tokens = tokenize_text("FlashAttention: Fast and Memory-Efficient Exact Attention")
        assert "flashattention" in tokens
        assert "fast" in tokens
        assert "memory" in tokens
        assert "efficient" in tokens

    def test_tokenize_empty_and_whitespace(self) -> None:
        assert tokenize_text("") == []
        assert tokenize_text("   \n\t  ") == []

    def test_tokenize_case_insensitivity(self) -> None:
        assert tokenize_text("Attention Attention ATTENTION") == ["attention", "attention", "attention"]


class TestBM25SearchService:
    """Tests for BM25 ranking and corpus retrieval."""

    @pytest.fixture
    def service(self) -> BM25SearchService:
        return BM25SearchService(k1=1.5, b=0.75)

    @pytest.fixture
    def sample_chunks(self) -> list[VectorSearchResult]:
        return [
            _make_result("c1", "FlashAttention uses tiling to reduce memory reads and writes between GPU HBM and SRAM."),
            _make_result("c2", "Convolutional neural networks are widely used in computer vision for feature extraction."),
            _make_result("c3", "Recurrent neural networks process sequential data with hidden state updates."),
            _make_result("c4", "Tiling and kernel fusion optimize GPU SRAM memory bandwidth in deep learning models."),
        ]

    def test_search_exact_keyword_ranks_highest(
        self, service: BM25SearchService, sample_chunks: list[VectorSearchResult]
    ) -> None:
        results = service.search(query="FlashAttention GPU SRAM", chunks=sample_chunks, top_k=2)
        assert len(results) > 0
        assert results[0].chunk_id == "c1"
        assert results[0].score > 0.0

    def test_search_respects_top_k(
        self, service: BM25SearchService, sample_chunks: list[VectorSearchResult]
    ) -> None:
        results = service.search(query="GPU neural", chunks=sample_chunks, top_k=1)
        assert len(results) == 1

    def test_search_empty_query_returns_empty(
        self, service: BM25SearchService, sample_chunks: list[VectorSearchResult]
    ) -> None:
        assert service.search(query="", chunks=sample_chunks) == []
        assert service.search(query="   ", chunks=sample_chunks) == []

    def test_search_empty_corpus_returns_empty(self, service: BM25SearchService) -> None:
        assert service.search(query="transformer", chunks=[]) == []

    def test_search_no_matching_terms_returns_empty(
        self, service: BM25SearchService, sample_chunks: list[VectorSearchResult]
    ) -> None:
        results = service.search(query="quantum cryptography photon", chunks=sample_chunks)
        assert len(results) == 0

    def test_user_scoped_chunk_isolation(self, service: BM25SearchService) -> None:
        user_a_chunks = [
            _make_result("a1", "Confidential financial quarterly report for User A.", user_id="user_a"),
        ]
        user_b_chunks = [
            _make_result("b1", "User B research manuscript on quantum computing.", user_id="user_b"),
        ]

        # Searching user B's pool with user A's keywords returns zero results
        results = service.search(query="financial report", chunks=user_b_chunks)
        assert len(results) == 0

        # Searching user A's pool returns user A's result
        results_a = service.search(query="financial report", chunks=user_a_chunks)
        assert len(results_a) == 1
        assert results_a[0].metadata.user_id == "user_a"
