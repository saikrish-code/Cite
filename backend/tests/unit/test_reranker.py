"""Unit tests for Cross-Encoder passage reranker."""

from unittest.mock import MagicMock, patch

import pytest

from app.schemas.document import VectorChunkMetadata, VectorSearchResult
from app.services.reranker import CrossEncoderReranker, MockReranker


def _make_result(chunk_id: str, text: str, score: float = 0.5) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        text=text,
        metadata=VectorChunkMetadata(
            document_id="doc1",
            user_id="user1",
            filename="doc1.txt",
            page=1,
            chunk_index=0,
        ),
        score=score,
    )


class TestMockReranker:
    """Tests for deterministic MockReranker."""

    def test_rerank_orders_by_query_overlap(self) -> None:
        reranker = MockReranker()
        candidates = [
            _make_result("c1", "Unrelated content about history and geography.", score=0.9),
            _make_result("c2", "FlashAttention exact attention mechanism on GPU.", score=0.1),
            _make_result("c3", "Attention mechanisms in transformers.", score=0.5),
        ]

        reranked = reranker.rerank(query="FlashAttention GPU", candidates=candidates, top_k=2)

        assert len(reranked) == 2
        # c2 has 2 word matches ("FlashAttention", "GPU")
        assert reranked[0].chunk_id == "c2"

    def test_rerank_empty_candidates_returns_empty(self) -> None:
        reranker = MockReranker()
        assert reranker.rerank(query="test", candidates=[]) == []


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker with mocked neural inference."""

    def test_rerank_empty_and_single_candidate(self) -> None:
        reranker = CrossEncoderReranker(model_name="test-model")
        assert reranker.rerank(query="test", candidates=[]) == []

        single = [_make_result("s1", "Single passage.")]
        assert reranker.rerank(query="test", candidates=single) == single

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    def test_rerank_with_mocked_model(
        self,
        mock_model_cls: MagicMock,
        mock_tok_cls: MagicMock,
    ) -> None:
        import torch

        # Mock tokenizer and model outputs
        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.tensor([[1, 2], [3, 4]])}
        mock_tok_cls.return_value = mock_tok

        mock_model = MagicMock()
        # Logits for 2 candidates: candidate 0 gets 0.2, candidate 1 gets 0.9
        mock_model.return_value.logits = torch.tensor([[0.2], [0.9]])
        mock_model_cls.return_value = mock_model

        reranker = CrossEncoderReranker(model_name="test-model")
        candidates = [
            _make_result("c1", "Lower relevance text.", score=0.8),
            _make_result("c2", "High relevance target text.", score=0.3),
        ]

        reranked = reranker.rerank(query="query", candidates=candidates, top_k=2)

        assert len(reranked) == 2
        # c2 had logit 0.9, so it should be ranked first
        assert reranked[0].chunk_id == "c2"
        assert reranked[0].score == 0.9
        assert reranked[1].chunk_id == "c1"
        assert reranked[1].score == 0.2
