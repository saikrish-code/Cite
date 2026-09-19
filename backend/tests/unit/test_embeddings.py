"""Unit tests for dense embedding services."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from app.services.embeddings import (
    BaseEmbeddingService,
    SentenceTransformerEmbeddingService,
)


class DummyEmbeddingService(BaseEmbeddingService):
    """Minimal concrete implementation of BaseEmbeddingService for testing."""

    @property
    def dimension(self) -> int:
        return 4

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.0, 0.5] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0, 0.5]


class TestBaseEmbeddingService:
    """Tests verifying the abstract BaseEmbeddingService contract."""

    def test_subclass_implements_interface(self) -> None:
        service = DummyEmbeddingService()
        assert service.dimension == 4
        assert service.embed_query("test") == [4.0, 1.0, 0.0, 0.5]
        docs = service.embed_documents(["a", "abc"])
        assert len(docs) == 2
        assert docs[0] == [1.0, 1.0, 0.0, 0.5]
        assert docs[1] == [3.0, 1.0, 0.0, 0.5]

    @pytest.mark.asyncio
    async def test_async_embed_methods(self) -> None:
        service = DummyEmbeddingService()
        query_vec = await service.aembed_query("async query")
        assert len(query_vec) == 4

        doc_vecs = await service.aembed_documents(["doc1", "doc2"])
        assert len(doc_vecs) == 2
        assert len(doc_vecs[0]) == 4


class TestSentenceTransformerEmbeddingService:
    """Unit tests for the SentenceTransformerEmbeddingService."""

    def test_lazy_initialization(self) -> None:
        """The heavy model should not be instantiated until first property access."""
        service = SentenceTransformerEmbeddingService(model_name="test-model")
        assert service._model is None
        assert service.model_name == "test-model"

    def test_embed_documents_empty_list(self) -> None:
        """Embedding an empty list should immediately return an empty list without calling model."""
        service = SentenceTransformerEmbeddingService()
        result = service.embed_documents([])
        assert result == []
        assert service._model is None

    def test_embed_query_empty_raises_error(self) -> None:
        """Blank or whitespace-only queries should raise ValueError."""
        service = SentenceTransformerEmbeddingService()
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            service.embed_query("")
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            service.embed_query("   ")

    def test_mocked_model_inference(self) -> None:
        """Verify batch encoding, dimension extraction, and normalization flow."""
        service = SentenceTransformerEmbeddingService(model_name="mock-model")

        mock_st = MagicMock()
        mock_st.get_embedding_dimension.return_value = 384
        mock_st.get_sentence_embedding_dimension.return_value = 384

        import numpy as np

        # Create 2 normalized 384-d vectors
        v1 = np.ones(384) / math.sqrt(384)
        v2 = np.ones(384) / math.sqrt(384)
        mock_st.encode.side_effect = lambda sentences, **kwargs: (
            v1 if isinstance(sentences, str) else np.array([v1, v2])
        )

        service._model = mock_st

        assert service.dimension == 384

        # Test query embedding
        q_vec = service.embed_query("Sample search query")
        assert len(q_vec) == 384
        assert pytest.approx(sum(x * x for x in q_vec), abs=1e-4) == 1.0

        # Test document batch embedding
        doc_vecs = service.embed_documents(["First passage", "Second passage"])
        assert len(doc_vecs) == 2
        assert len(doc_vecs[0]) == 384
        assert len(doc_vecs[1]) == 384

    @pytest.mark.asyncio
    async def test_async_delegation(self) -> None:
        """Verify aembed_documents and aembed_query offload to thread correctly."""
        service = SentenceTransformerEmbeddingService(model_name="mock-model")
        mock_st = MagicMock()
        mock_st.get_sentence_embedding_dimension.return_value = 128

        import numpy as np

        v = np.zeros(128)
        mock_st.encode.side_effect = lambda sentences, **kwargs: (
            v if isinstance(sentences, str) else np.array([v])
        )
        service._model = mock_st

        q_res = await service.aembed_query("Async query")
        assert len(q_res) == 128

        d_res = await service.aembed_documents(["Async doc"])
        assert len(d_res) == 1
        assert len(d_res[0]) == 128
