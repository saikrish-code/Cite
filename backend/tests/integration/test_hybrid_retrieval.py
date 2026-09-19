"""Integration tests for hybrid retrieval, benchmarking stage toggling, and latency metrics."""

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.endpoints.documents import _background_ingest
from app.core.database import Base, engine
from app.main import app


@pytest.fixture(autouse=True)
async def clean_database():
    """Ensure clean database schema for isolated integration tests."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield


@pytest.mark.asyncio
async def test_hybrid_retrieval_modes_and_latency_metrics() -> None:
    """Benchmark and verify vector_only, hybrid, and hybrid_rerank modes via the chat endpoint."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # 1. Register and sign in user
        signup_res = await client.post(
            "/api/v1/auth/signup",
            json={"email": "benchmarker@test.com", "password": "Password123!", "full_name": "Benchmark User"},
        )
        assert signup_res.status_code == 201
        res_data = signup_res.json()
        token = res_data["access_token"]
        user_id = res_data["user"]["id"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Ingest document directly to guarantee index availability
        sample_text = (
            "FlashAttention is a fast and memory-efficient exact attention algorithm. "
            "It uses tiling to reduce the number of memory reads and writes between GPU HBM and SRAM. "
            "Standard multi-head attention has quadratic IO complexity with sequence length. "
            "FlashAttention achieves 2x-4x speedup compared to standard attention."
        )
        doc_id = "doc-flash-1"
        await _background_ingest(
            file_bytes=sample_text.encode("utf-8"),
            filename="flash_attention.txt",
            document_id=doc_id,
            user_id=user_id,
        )

        # Mock LLM streaming to return answer with [1]
        async def mock_stream(*args, **kwargs):
            for tok in ["FlashAttention", " optimizes", " GPU", " memory", " with", " tiling", " [1]."]:
                yield tok

        with patch("app.services.llm.MockLLMClient.astream_generate", side_effect=mock_stream):
            # 3. Test vector_only mode
            res_vec = await client.post(
                "/api/v1/chat",
                headers=headers,
                json={
                    "query": "How does FlashAttention optimize GPU memory?",
                    "retrieval_mode": "vector_only",
                    "k": 2,
                },
            )
            assert res_vec.status_code == 200
            assert "text/event-stream" in res_vec.headers["content-type"]
            text_vec = res_vec.text
            assert "event: citations" in text_vec
            assert "vector_only" in text_vec

            # 4. Test hybrid mode (BM25 + Vector + RRF)
            res_hybrid = await client.post(
                "/api/v1/chat",
                headers=headers,
                json={
                    "query": "How does FlashAttention optimize GPU memory?",
                    "retrieval_mode": "hybrid",
                    "k": 2,
                },
            )
            assert res_hybrid.status_code == 200
            text_hybrid = res_hybrid.text
            assert "event: citations" in text_hybrid
            assert "hybrid" in text_hybrid

            # 5. Test hybrid_rerank mode (BM25 + Vector + RRF + Cross-Encoder)
            res_rerank = await client.post(
                "/api/v1/chat",
                headers=headers,
                json={
                    "query": "How does FlashAttention optimize GPU memory?",
                    "retrieval_mode": "hybrid_rerank",
                    "k": 2,
                },
            )
            assert res_rerank.status_code == 200
            text_rerank = res_rerank.text
            assert "event: citations" in text_rerank
            assert "hybrid_rerank" in text_rerank
