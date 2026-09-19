"""Integration tests for chat SSE streaming and history API endpoints."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_current_user
from app.core.database import AsyncSessionLocal, Base, engine
from app.main import app
from app.models.chat import ChatMessage
from app.models.document import Document
from app.models.user import User
from app.schemas.document import VectorChunkMetadata, VectorSearchResult
from app.services.document_store import chat_store
from sqlalchemy import select


@pytest.fixture(autouse=True)
async def _setup_db_and_auth():
    """Ensure DB schema and mock authenticated user for chat endpoints."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    test_user = User(
        id="test-user-id",
        email="test_chat_user@university.edu",
        hashed_password="hashed_pwd",
        full_name="Chat Tester",
        is_active=True,
    )
    test_doc = Document(
        id="doc-abc-123",
        user_id=test_user.id,
        filename="test_paper.pdf",
        status="ready",
    )
    async with AsyncSessionLocal() as session:
        session.add(test_user)
        session.add(test_doc)
        await session.commit()

    app.dependency_overrides[get_current_user] = lambda: test_user
    chat_store.clear()
    yield
    chat_store.clear()
    app.dependency_overrides.pop(get_current_user, None)


async def _add_test_message(
    entry_id: str, query: str, answer: str, user_id: str = "test-user-id"
) -> None:
    async with AsyncSessionLocal() as session:
        msg = ChatMessage(
            id=entry_id,
            user_id=user_id,
            query=query,
            answer=answer,
            citations=[],
            context_found=True,
        )
        session.add(msg)
        await session.commit()
    chat_store.add_entry(query=query, answer=answer, entry_id=entry_id)


@pytest.fixture
async def client() -> AsyncClient:
    """Provide an async HTTP client bound to the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


def _make_search_result(
    text: str = "Sample chunk text about machine learning.",
    filename: str = "paper.pdf",
    page: int = 1,
    chunk_index: int = 0,
    score: float = 0.92,
) -> VectorSearchResult:
    """Create a VectorSearchResult fixture for testing."""
    return VectorSearchResult(
        chunk_id=f"test_{chunk_index}",
        text=text,
        metadata=VectorChunkMetadata(
            user_id="anonymous",
            document_id="test-doc",
            filename=filename,
            page=page,
            chunk_index=chunk_index,
        ),
        score=score,
    )


class TestChatStream:
    """Tests for POST /api/v1/chat (SSE streaming)."""

    async def test_chat_stream_returns_sse_events(self, client: AsyncClient) -> None:
        """A valid chat request returns SSE events with token, citations, and done."""
        mock_search_results = [
            _make_search_result("Machine learning is a subset of AI [1].", page=1, chunk_index=0),
            _make_search_result("Deep learning uses neural networks [2].", page=2, chunk_index=1),
        ]

        # Mock the vector store and LLM client
        with (
            patch("app.api.v1.endpoints.chat.ChromaVectorStore") as MockVS,
            patch("app.api.v1.endpoints.chat.get_llm_client") as MockLLMFactory,
        ):
            mock_vs_instance = MagicMock()
            mock_vs_instance.similarity_search.return_value = mock_search_results
            MockVS.return_value = mock_vs_instance

            # Create a mock LLM that streams word by word
            mock_llm = MagicMock()
            mock_llm.model_name = "mock-model"

            async def mock_stream(*args, **kwargs):
                for word in ["Machine", " learning", " is", " AI", " [1]."]:
                    yield word

            mock_llm.astream_generate = mock_stream
            MockLLMFactory.return_value = mock_llm

            response = await client.post(
                "/api/v1/chat",
                json={"query": "What is machine learning?"},
            )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Parse SSE events
        events = _parse_sse_events(response.text)
        event_types = [e["event"] for e in events]

        # Should have token events, then citations, then done
        assert "token" in event_types
        assert "citations" in event_types
        assert "done" in event_types

        # Verify token events contain text
        token_events = [e for e in events if e["event"] == "token"]
        assert len(token_events) > 0
        full_text = "".join(e["data"]["token"] for e in token_events)
        assert "Machine" in full_text

        # Verify done event has the full answer
        done_event = next(e for e in events if e["event"] == "done")
        assert "answer" in done_event["data"]
        assert "query" in done_event["data"]

    async def test_chat_stream_no_context(self, client: AsyncClient) -> None:
        """When no documents exist, returns 'I don't know' via SSE."""
        with (
            patch("app.api.v1.endpoints.chat.ChromaVectorStore") as MockVS,
            patch("app.api.v1.endpoints.chat.get_llm_client") as MockLLMFactory,
        ):
            mock_vs_instance = MagicMock()
            mock_vs_instance.similarity_search.return_value = []
            MockVS.return_value = mock_vs_instance

            mock_llm = MagicMock()
            MockLLMFactory.return_value = mock_llm

            response = await client.post(
                "/api/v1/chat",
                json={"query": "What is quantum computing?"},
            )

        assert response.status_code == 200
        events = _parse_sse_events(response.text)

        # Should have token with "I don't know" message
        token_events = [e for e in events if e["event"] == "token"]
        full_text = "".join(e["data"]["token"] for e in token_events)
        assert "don't know" in full_text.lower()

        # Citations should be empty with context_found=False
        citations_event = next(e for e in events if e["event"] == "citations")
        assert citations_event["data"]["citations"] == []
        assert citations_event["data"]["context_found"] is False

    async def test_chat_stream_empty_query_returns_422(self, client: AsyncClient) -> None:
        """Sending an empty query returns 422 validation error."""
        response = await client.post(
            "/api/v1/chat",
            json={"query": ""},
        )

        assert response.status_code == 422

    async def test_chat_stream_missing_query_returns_422(self, client: AsyncClient) -> None:
        """Sending request without query field returns 422."""
        response = await client.post(
            "/api/v1/chat",
            json={},
        )

        assert response.status_code == 422

    async def test_chat_stream_with_document_filter(self, client: AsyncClient) -> None:
        """Chat request with document_id passes filter to vector store."""
        with (
            patch("app.api.v1.endpoints.chat.ChromaVectorStore") as MockVS,
            patch("app.api.v1.endpoints.chat.get_llm_client") as MockLLMFactory,
        ):
            mock_vs_instance = MagicMock()
            mock_vs_instance.similarity_search.return_value = []
            MockVS.return_value = mock_vs_instance

            mock_llm = MagicMock()
            MockLLMFactory.return_value = mock_llm

            response = await client.post(
                "/api/v1/chat",
                json={
                    "query": "Summarize the paper",
                    "document_id": "doc-abc-123",
                    "k": 6,
                },
            )

        assert response.status_code == 200

    async def test_chat_stream_citations_event_structure(self, client: AsyncClient) -> None:
        """Citations event contains properly structured citation objects."""
        mock_results = [
            _make_search_result("Neural nets learn features [1].", page=3, chunk_index=0, score=0.95),
        ]

        with (
            patch("app.api.v1.endpoints.chat.ChromaVectorStore") as MockVS,
            patch("app.api.v1.endpoints.chat.get_llm_client") as MockLLMFactory,
        ):
            mock_vs_instance = MagicMock()
            mock_vs_instance.similarity_search.return_value = mock_results
            MockVS.return_value = mock_vs_instance

            mock_llm = MagicMock()

            async def mock_stream(*args, **kwargs):
                yield "Neural networks learn features [1]."

            mock_llm.astream_generate = mock_stream
            MockLLMFactory.return_value = mock_llm

            response = await client.post(
                "/api/v1/chat",
                json={"query": "How do neural networks work?"},
            )

        events = _parse_sse_events(response.text)
        citations_event = next(e for e in events if e["event"] == "citations")
        citations = citations_event["data"]["citations"]

        assert len(citations) >= 1
        first_citation = citations[0]
        assert "source_id" in first_citation
        assert "filename" in first_citation
        assert "page" in first_citation
        assert "snippet" in first_citation
        assert "score" in first_citation


class TestChatHistory:
    """Tests for GET /api/v1/chat/history."""

    async def test_history_empty(self, client: AsyncClient) -> None:
        """Empty history returns empty list."""
        response = await client.get("/api/v1/chat/history")

        assert response.status_code == 200
        assert response.json() == []

    async def test_history_returns_entries(self, client: AsyncClient) -> None:
        """History returns previously added entries."""
        await _add_test_message(
            entry_id="entry-1",
            query="What is ML?",
            answer="Machine learning is a field of AI [1].",
        )
        await _add_test_message(
            entry_id="entry-2",
            query="What is DL?",
            answer="Deep learning is a subset of ML [1].",
        )

        response = await client.get("/api/v1/chat/history")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        # Both entries returned
        entry_ids = [d["id"] for d in data]
        assert "entry-1" in entry_ids
        assert "entry-2" in entry_ids

    async def test_history_pagination(self, client: AsyncClient) -> None:
        """History respects limit and offset parameters."""
        for i in range(5):
            await _add_test_message(
                entry_id=f"entry-{i}",
                query=f"Question {i}",
                answer=f"Answer {i}",
            )

        # Get only 2, skip 1
        response = await client.get(
            "/api/v1/chat/history",
            params={"limit": 2, "offset": 1},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    async def test_history_entry_structure(self, client: AsyncClient) -> None:
        """History entries have the expected fields."""
        await _add_test_message(
            entry_id="struct-test",
            query="Test query",
            answer="Test answer",
        )

        response = await client.get("/api/v1/chat/history")

        assert response.status_code == 200
        entry = response.json()[0]
        assert entry["id"] == "struct-test"
        assert entry["query"] == "Test query"
        assert entry["answer"] == "Test answer"
        assert entry["context_found"] is True
        assert "created_at" in entry
        assert "citations" in entry


def _parse_sse_events(raw_text: str) -> list[dict]:
    """Parse raw SSE text into a list of event dictionaries.

    Args:
        raw_text: Raw SSE response body.

    Returns:
        list[dict]: Parsed events with 'event' and 'data' keys.
    """
    events = []
    current_event: dict = {}

    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line:
            if current_event:
                events.append(current_event)
                current_event = {}
            continue

        if line.startswith("event: "):
            current_event["event"] = line[7:]
        elif line.startswith("data: "):
            try:
                current_event["data"] = json.loads(line[6:])
            except json.JSONDecodeError:
                current_event["data"] = line[6:]

    # Don't forget the last event if no trailing newline
    if current_event:
        events.append(current_event)

    return events
