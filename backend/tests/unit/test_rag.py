"""Unit tests for the provider-agnostic LLM clients and RAG answer service.

Tests cover:
- LLM client factory (OpenAI, Claude, Ollama, Mock, and invalid provider validation)
- MockLLMClient prompt history and response sequencing
- Missing API key guards for OpenAI and Claude clients
- Strict prompt generation with [1], [2] citation markers
- Grounded answer generation and Citation schema extraction
- Multi-citation parsing e.g. [1][2]
- 'I don't know' handling and fallback on insufficient or empty context
- Filtering by document_id and user_id in retrieval
- Async aanswer execution
"""

from __future__ import annotations

import pytest

from app.schemas.chat import RAGResponse
from app.schemas.document import ChunkMetadata, TextChunk
from app.services.embeddings import BaseEmbeddingService
from app.services.llm import (
    ClaudeLLMClient,
    MockLLMClient,
    OllamaLLMClient,
    OpenAILLMClient,
    get_llm_client,
)
from app.services.rag import RAGService
from app.services.vector_store import ChromaVectorStore


class MockFastEmbedder(BaseEmbeddingService):
    """Deterministic fast embedder for RAG tests."""

    @property
    def dimension(self) -> int:
        return 4

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        lower = text.lower()
        if "transformer" in lower or "attention" in lower:
            return [1.0, 0.0, 0.0, 0.0]
        elif "optimizer" in lower or "adam" in lower:
            return [0.0, 1.0, 0.0, 0.0]
        return [0.25, 0.25, 0.25, 0.25]


def create_chunk(
    text: str,
    filename: str = "attention.pdf",
    page: int = 1,
    chunk_index: int = 0,
) -> TextChunk:
    """Helper to construct TextChunk instances."""
    return TextChunk(
        text=text,
        metadata=ChunkMetadata(
            source_filename=filename,
            page_number=page,
            chunk_index=chunk_index,
            char_start=0,
            char_end=len(text),
        ),
    )


@pytest.fixture
def populated_vector_store() -> ChromaVectorStore:
    """In-memory Chroma vector store populated with sample chunks."""
    store = ChromaVectorStore(
        embedding_service=MockFastEmbedder(),
        persist_directory=":memory:",
        collection_name="rag_test_collection",
    )
    chunks = [
        create_chunk(
            text="The Transformer is the first transduction model relying entirely on self-attention.",
            filename="attention.pdf",
            page=1,
            chunk_index=0,
        ),
        create_chunk(
            text="Multi-head attention allows the model to jointly attend to information from different representation subspaces.",
            filename="attention.pdf",
            page=2,
            chunk_index=1,
        ),
        create_chunk(
            text="We use the Adam optimizer with beta_1 = 0.9, beta_2 = 0.98 and epsilon = 10^-9.",
            filename="attention.pdf",
            page=7,
            chunk_index=2,
        ),
    ]
    store.add_chunks(chunks, document_id="doc_attn_123", user_id="user_alice")
    return store


class TestLLMClients:
    """Tests for LLM client factory and provider implementations."""

    def test_factory_returns_correct_providers(self) -> None:
        """get_llm_client should instantiate the appropriate provider class."""
        assert isinstance(get_llm_client("mock"), MockLLMClient)
        assert isinstance(get_llm_client("openai", api_key="test-key"), OpenAILLMClient)
        assert isinstance(get_llm_client("claude", api_key="test-key"), ClaudeLLMClient)
        assert isinstance(
            get_llm_client("anthropic", api_key="test-key"), ClaudeLLMClient
        )
        assert isinstance(get_llm_client("ollama"), OllamaLLMClient)

    def test_factory_rejects_unknown_provider(self) -> None:
        """Unknown provider strings should raise ValueError with helpful hint."""
        with pytest.raises(
            ValueError, match="Unsupported LLM provider 'invalid_provider'"
        ):
            get_llm_client("invalid_provider")

    def test_mock_llm_client_recording_and_sequencing(self) -> None:
        """MockLLMClient should record prompt calls and return sequential responses."""
        mock = MockLLMClient(responses=["Response 1 [1]", "Response 2 [2]"])
        assert mock.model_name == "mock-model"

        r1 = mock.generate("Prompt 1", system_prompt="System 1")
        assert r1 == "Response 1 [1]"
        assert len(mock.call_history) == 1
        assert mock.call_history[0]["prompt"] == "Prompt 1"
        assert mock.call_history[0]["system_prompt"] == "System 1"

        r2 = mock.generate("Prompt 2")
        assert r2 == "Response 2 [2]"

        # Falls back to default response once sequence is exhausted
        r3 = mock.generate("Prompt 3")
        assert "[1]" in r3

    @pytest.mark.asyncio
    async def test_mock_llm_client_async(self) -> None:
        """MockLLMClient should support async agenerate."""
        mock = MockLLMClient(default_response="Async mock response [1]")
        result = await mock.agenerate("Async prompt", system_prompt="Sys")
        assert result == "Async mock response [1]"
        assert len(mock.call_history) == 1

    def test_openai_missing_key_raises_error(self) -> None:
        """OpenAILLMClient should raise ValueError on call if no API key is set."""
        client = OpenAILLMClient(api_key=None)
        # Clear out any env var for test
        client.api_key = None
        with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
            client.generate("Test prompt")

    def test_claude_missing_key_raises_error(self) -> None:
        """ClaudeLLMClient should raise ValueError on call if no API key is set."""
        client = ClaudeLLMClient(api_key=None)
        client.api_key = None
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
            client.generate("Test prompt")


class TestRAGService:
    """Tests for the end-to-end RAG answer service."""

    def test_answer_with_single_citation(
        self, populated_vector_store: ChromaVectorStore
    ) -> None:
        """LLM answers with citation [1], mapped to source metadata."""
        mock_llm = MockLLMClient(
            default_response="The Transformer relies solely on self-attention mechanisms [1]."
        )
        rag_service = RAGService(
            vector_store=populated_vector_store,
            llm_client=mock_llm,
        )

        resp = rag_service.answer(query="What is the Transformer based on?", k=2)

        assert isinstance(resp, RAGResponse)
        assert resp.context_found is True
        assert "[1]" in resp.answer
        assert len(resp.citations) == 1

        citation = resp.citations[0]
        assert citation.source_id == 1
        assert citation.filename == "attention.pdf"
        assert citation.page == 1
        assert "self-attention" in citation.snippet
        assert citation.score > 0.0

    def test_answer_with_multiple_citations(
        self, populated_vector_store: ChromaVectorStore
    ) -> None:
        """LLM cites both [1] and [2] in response."""
        mock_llm = MockLLMClient(
            default_response=(
                "The Transformer uses self-attention [1]. "
                "Multi-head attention attends to different subspaces jointly [2]."
            )
        )
        rag_service = RAGService(
            vector_store=populated_vector_store,
            llm_client=mock_llm,
        )

        resp = rag_service.answer(
            query="Explain self-attention and multi-head attention", k=3
        )

        assert resp.context_found is True
        assert len(resp.citations) == 2
        assert resp.citations[0].source_id == 1
        assert resp.citations[0].page == 1
        assert resp.citations[1].source_id == 2
        assert resp.citations[1].page == 2

    def test_answer_with_combined_brackets(
        self, populated_vector_store: ChromaVectorStore
    ) -> None:
        """Multi-citation tags like [1][2] should be extracted cleanly."""
        mock_llm = MockLLMClient(
            default_response="Attention mechanisms are central to modern sequence models [1][2]."
        )
        rag_service = RAGService(
            vector_store=populated_vector_store,
            llm_client=mock_llm,
        )

        resp = rag_service.answer(query="Attention mechanisms", k=2)
        assert len(resp.citations) == 2
        assert resp.citations[0].source_id == 1
        assert resp.citations[1].source_id == 2

    def test_fallback_when_context_is_insufficient(
        self, populated_vector_store: ChromaVectorStore
    ) -> None:
        """When LLM returns 'I don't know', context_found should be False and citations empty."""
        mock_llm = MockLLMClient(
            default_response="I don't know based on the provided context."
        )
        rag_service = RAGService(
            vector_store=populated_vector_store,
            llm_client=mock_llm,
        )

        resp = rag_service.answer(query="What is the capital of France?", k=2)

        assert resp.context_found is False
        assert "I don't know" in resp.answer
        assert resp.citations == []

    def test_empty_query_returns_unknown(
        self, populated_vector_store: ChromaVectorStore
    ) -> None:
        """Empty or blank query should return 'I don't know' without calling LLM."""
        mock_llm = MockLLMClient()
        rag_service = RAGService(
            vector_store=populated_vector_store,
            llm_client=mock_llm,
        )

        resp = rag_service.answer(query="   ")
        assert resp.context_found is False
        assert resp.citations == []
        assert len(mock_llm.call_history) == 0

    def test_empty_vector_store_returns_unknown(self) -> None:
        """If vector store has no documents, return 'I don't know' immediately."""
        empty_store = ChromaVectorStore(
            embedding_service=MockFastEmbedder(),
            persist_directory=":memory:",
            collection_name="empty_rag_store",
        )
        mock_llm = MockLLMClient()
        rag_service = RAGService(
            vector_store=empty_store,
            llm_client=mock_llm,
        )

        resp = rag_service.answer(query="Any question")
        assert resp.context_found is False
        assert resp.citations == []
        assert len(mock_llm.call_history) == 0

    def test_filter_isolation_by_user_id(
        self, populated_vector_store: ChromaVectorStore
    ) -> None:
        """Filtering by an unknown user_id should find no chunks and return 'I don't know'."""
        mock_llm = MockLLMClient()
        rag_service = RAGService(
            vector_store=populated_vector_store,
            llm_client=mock_llm,
        )

        resp = rag_service.answer(
            query="Transformer attention",
            user_id="user_bob_who_has_no_docs",
        )
        assert resp.context_found is False
        assert resp.citations == []
        assert len(mock_llm.call_history) == 0

    def test_filter_matching_document_id(
        self, populated_vector_store: ChromaVectorStore
    ) -> None:
        """Specifying valid document_id and user_id routes to correct documents."""
        mock_llm = MockLLMClient(
            default_response="Adam optimizer is configured with beta_1 = 0.9 [1]."
        )
        rag_service = RAGService(
            vector_store=populated_vector_store,
            llm_client=mock_llm,
        )

        resp = rag_service.answer(
            query="What optimizer is used?",
            document_id="doc_attn_123",
            user_id="user_alice",
        )
        assert resp.context_found is True
        assert len(resp.citations) == 1
        assert resp.citations[0].filename == "attention.pdf"

    def test_prompt_format_contains_strict_rules(
        self, populated_vector_store: ChromaVectorStore
    ) -> None:
        """Check that the generated prompt passed to LLM includes strict rules and context labels."""
        mock_llm = MockLLMClient()
        rag_service = RAGService(
            vector_store=populated_vector_store,
            llm_client=mock_llm,
        )

        rag_service.answer(query="What is self-attention?", k=2)

        assert len(mock_llm.call_history) == 1
        recorded = mock_llm.call_history[0]
        prompt = recorded["prompt"]
        system_prompt = recorded["system_prompt"]

        # Check system prompt constraints
        assert "ONLY" in system_prompt
        assert "[1]" in system_prompt
        assert "I don't know" in system_prompt

        # Check user prompt context headers
        assert "[1] (Source: attention.pdf, Page: 1)" in prompt
        assert "[2] (Source: attention.pdf, Page: 2)" in prompt

    @pytest.mark.asyncio
    async def test_async_aanswer(
        self, populated_vector_store: ChromaVectorStore
    ) -> None:
        """Async aanswer method should return valid RAGResponse."""
        mock_llm = MockLLMClient(
            default_response="Self-attention allows joint attending across subspaces [1]."
        )
        rag_service = RAGService(
            vector_store=populated_vector_store,
            llm_client=mock_llm,
        )

        resp = await rag_service.aanswer(query="Tell me about self-attention", k=1)
        assert isinstance(resp, RAGResponse)
        assert resp.context_found is True
        assert len(resp.citations) == 1
        assert resp.citations[0].source_id == 1
