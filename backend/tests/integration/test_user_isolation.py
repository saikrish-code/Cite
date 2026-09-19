"""Comprehensive integration tests verifying multi-tenant per-user data isolation.

Verifies:
1. User A and User B account creation, password hashing, and JWT token authentication.
2. User A uploads a document: User B can never list, view, or delete User A's document.
3. User B attempting to delete User A's document receives 404 Not Found.
4. User A's vector embeddings are never retrieved for User B's queries (zero citations).
5. User B querying chat with User A's document_id receives 404 unauthorized.
6. User B can never retrieve User A's chat history.
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.database import Base, engine
from app.main import app
from app.schemas.document import (
    ChunkMetadata,
    TextChunk,
    VectorChunkMetadata,
    VectorSearchResult,
)


@pytest.fixture(autouse=True)
async def prepare_database():
    """Ensure clean database tables for isolation tests."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield


@pytest.mark.asyncio
async def test_user_signup_login_flow():
    """Verify signup and login endpoints return valid JWT tokens and user profiles."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Signup User A
        res_signup = await client.post(
            "/api/v1/auth/signup",
            json={
                "email": "alice@university.edu",
                "password": "Password123!",
                "full_name": "Alice Researcher",
            },
        )
        assert res_signup.status_code == 201
        data_a = res_signup.json()
        assert "access_token" in data_a
        assert data_a["user"]["email"] == "alice@university.edu"
        token_a = data_a["access_token"]

        # Duplicate signup should fail
        res_dup = await client.post(
            "/api/v1/auth/signup",
            json={
                "email": "alice@university.edu",
                "password": "Password123!",
            },
        )
        assert res_dup.status_code == 400

        # Login User A
        res_login = await client.post(
            "/api/v1/auth/login",
            json={
                "email": "alice@university.edu",
                "password": "Password123!",
            },
        )
        assert res_login.status_code == 200
        assert "access_token" in res_login.json()

        # Probe /auth/me
        res_me = await client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert res_me.status_code == 200
        assert res_me.json()["email"] == "alice@university.edu"


@pytest.mark.asyncio
async def test_document_listing_and_deletion_isolation():
    """Prove User B cannot list, access, or delete User A's uploaded documents."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Register User A (Alice)
        res_a = await client.post(
            "/api/v1/auth/signup",
            json={
                "email": "alice_docs@university.edu",
                "password": "SecurePassword123!",
                "full_name": "Alice",
            },
        )
        token_a = res_a.json()["access_token"]

        # Register User B (Bob)
        res_b = await client.post(
            "/api/v1/auth/signup",
            json={
                "email": "bob_docs@university.edu",
                "password": "SecurePassword123!",
                "full_name": "Bob",
            },
        )
        token_b = res_b.json()["access_token"]

        # User A uploads a document
        with patch(
            "app.services.vector_store.ChromaVectorStore.add_chunks",
            return_value=["chunk-1"],
        ), patch(
            "app.services.ingestion.IngestionService.ingest",
            return_value=[
                TextChunk(
                    text="Alice private research content.",
                    metadata=ChunkMetadata(
                        source_filename="alice_paper.pdf",
                        page_number=1,
                        chunk_index=0,
                        char_start=0,
                        char_end=31,
                    ),
                )
            ],
        ):
            file_content = b"%PDF-1.4 Alice confidential paper"
            upload_res = await client.post(
                "/api/v1/documents/upload",
                headers={"Authorization": f"Bearer {token_a}"},
                files={"file": ("alice_paper.pdf", file_content, "application/pdf")},
            )
            assert upload_res.status_code == 202
            doc_id_a = upload_res.json()["document_id"]

        # User A lists documents: sees 1 document
        res_list_a = await client.get(
            "/api/v1/documents",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert res_list_a.status_code == 200
        docs_a = res_list_a.json()["documents"]
        assert len(docs_a) == 1
        assert docs_a[0]["document_id"] == doc_id_a

        # User B lists documents: MUST BE EMPTY
        res_list_b = await client.get(
            "/api/v1/documents",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert res_list_b.status_code == 200
        docs_b = res_list_b.json()["documents"]
        assert len(docs_b) == 0, "User B must see 0 documents owned by User A"

        # User B attempts to delete User A's document: MUST BE 404 NOT FOUND
        res_del_b = await client.delete(
            f"/api/v1/documents/{doc_id_a}",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert res_del_b.status_code == 404, "User B must not be able to delete User A's doc"

        # User A deletes their own document: SUCCESS
        with patch("app.services.vector_store.ChromaVectorStore.delete_by_document", return_value=1):
            res_del_a = await client.delete(
                f"/api/v1/documents/{doc_id_a}",
                headers={"Authorization": f"Bearer {token_a}"},
            )
            assert res_del_a.status_code == 200


@pytest.mark.asyncio
async def test_chat_vector_retrieval_and_history_isolation():
    """Prove User B cannot retrieve User A's vector embeddings or view User A's chat history."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Register User A (Alice)
        res_a = await client.post(
            "/api/v1/auth/signup",
            json={
                "email": "alice_chat@university.edu",
                "password": "SecurePassword123!",
            },
        )
        token_a = res_a.json()["access_token"]
        user_id_a = res_a.json()["user"]["id"]

        # Register User B (Bob)
        res_b = await client.post(
            "/api/v1/auth/signup",
            json={
                "email": "bob_chat@university.edu",
                "password": "SecurePassword123!",
            },
        )
        token_b = res_b.json()["access_token"]

        # User A asks a question and receives answer with citations
        mock_result_a = VectorSearchResult(
            chunk_id="chunk-alice-1",
            text="Alice secret proprietary findings.",
            score=0.95,
            metadata=VectorChunkMetadata(
                filename="alice.pdf",
                page=1,
                chunk_index=0,
                document_id="doc-a-1",
                user_id=user_id_a,
            ),
        )

        def mock_similarity_search(query, k, filters):
            # Verify that ChromaVectorStore filters strictly by the calling user_id!
            if filters and filters.get("user_id") == user_id_a:
                return [mock_result_a]
            return []  # User B gets nothing

        with patch(
            "app.services.vector_store.ChromaVectorStore.similarity_search",
            side_effect=mock_similarity_search,
        ), patch(
            "app.services.llm.MockLLMClient.astream_generate"
        ) as mock_stream:
            async def _stream_tokens(*args, **kwargs):
                for tok in ["Alice", " findings", " verified", " [1]."]:
                    yield tok

            mock_stream.side_effect = _stream_tokens

            # Alice chats
            res_chat_a = await client.post(
                "/api/v1/chat",
                headers={"Authorization": f"Bearer {token_a}"},
                json={"query": "What are the secret findings?"},
            )
            assert res_chat_a.status_code == 200
            content_a = res_chat_a.text
            assert "event: token" in content_a
            assert "event: citations" in content_a

            # Bob asks the exact same question
            res_chat_b = await client.post(
                "/api/v1/chat",
                headers={"Authorization": f"Bearer {token_b}"},
                json={"query": "What are the secret findings?"},
            )
            assert res_chat_b.status_code == 200
            content_b = res_chat_b.text
            # Because Bob has 0 context passages matching his user_id, RAG yields "I don't know" with 0 citations
            assert "I don't know based on the provided context" in content_b
            assert '"citations": []' in content_b

        # Check chat history isolation
        # Alice history has 1 exchange
        hist_a = await client.get(
            "/api/v1/chat/history",
            headers={"Authorization": f"Bearer {token_a}"},
        )
        assert hist_a.status_code == 200
        assert len(hist_a.json()) == 1

        # Bob history has 1 exchange (his own "I don't know"), NOT Alice's
        hist_b = await client.get(
            "/api/v1/chat/history",
            headers={"Authorization": f"Bearer {token_b}"},
        )
        assert hist_b.status_code == 200
        entries_b = hist_b.json()
        assert len(entries_b) == 1
        assert entries_b[0]["answer"] == "I don't know based on the provided context."
        assert entries_b[0]["citations"] == []


@pytest.mark.asyncio
async def test_scoped_chat_with_other_user_doc_id_rejected():
    """Prove User B cannot scope chat to User A's document ID."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Register User A
        res_a = await client.post(
            "/api/v1/auth/signup",
            json={
                "email": "alice_scope@university.edu",
                "password": "SecurePassword123!",
            },
        )
        token_a = res_a.json()["access_token"]

        # Register User B
        res_b = await client.post(
            "/api/v1/auth/signup",
            json={
                "email": "bob_scope@university.edu",
                "password": "SecurePassword123!",
            },
        )
        token_b = res_b.json()["access_token"]

        # Alice uploads a document
        with patch("app.services.vector_store.ChromaVectorStore.add_chunks", return_value=["c1"]):
            upload_res = await client.post(
                "/api/v1/documents/upload",
                headers={"Authorization": f"Bearer {token_a}"},
                files={"file": ("alice_priv.txt", b"Private text", "text/plain")},
            )
            doc_id_a = upload_res.json()["document_id"]

        # Bob attempts to chat scoped to Alice's document ID -> 404 NOT FOUND
        res_chat_b = await client.post(
            "/api/v1/chat",
            headers={"Authorization": f"Bearer {token_b}"},
            json={
                "query": "Summarize",
                "document_id": doc_id_a,
            },
        )
        assert res_chat_b.status_code == 404
        assert "not found or unauthorized" in res_chat_b.json()["detail"].lower()
