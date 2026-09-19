"""Pydantic schemas for RAG chat, citations, SSE events, and chat history."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """Source citation linking generated claims back to retrieved document chunks."""

    model_config = ConfigDict(frozen=True)

    source_id: int = Field(
        ...,
        ge=1,
        description="1-indexed citation label referenced in answer text e.g. [1]",
    )
    filename: str = Field(..., description="Original filename of the cited document")
    page: int = Field(
        ..., ge=1, description="1-indexed page number of the cited passage"
    )
    snippet: str = Field(
        ..., description="Excerpt from the source chunk supporting the assertion"
    )
    score: float = Field(
        ..., description="Relevance similarity score from vector retrieval"
    )


class RetrievalMetrics(BaseModel):
    """Latency metrics measured across the retrieval pipeline stages."""

    model_config = ConfigDict(frozen=False)

    vector_ms: float = Field(default=0.0, description="Dense vector retrieval latency in ms")
    bm25_ms: float = Field(default=0.0, description="Sparse BM25 retrieval latency in ms")
    rrf_ms: float = Field(default=0.0, description="Reciprocal Rank Fusion latency in ms")
    rerank_ms: float = Field(default=0.0, description="Cross-Encoder reranking latency in ms")
    total_ms: float = Field(default=0.0, description="Total retrieval pipeline latency in ms")
    mode: str = Field(default="hybrid_rerank", description="Active retrieval mode")
    retrieval_mode: str = Field(default="hybrid_rerank", description="Alias for active retrieval mode")

    def model_post_init(self, __context: object) -> None:
        if self.mode:
            object.__setattr__(self, "retrieval_mode", self.mode)


class RAGRequest(BaseModel):
    """Request payload for RAG document question answering."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(
        ..., min_length=1, description="Natural language question to answer"
    )
    k: int = Field(
        default=4, ge=1, le=20, description="Number of context passages to retrieve"
    )
    document_id: str | None = Field(
        default=None, description="Optional document ID to scope retrieval"
    )
    user_id: str | None = Field(
        default=None, description="Optional user ID for multi-tenant isolation"
    )
    retrieval_mode: str | None = Field(
        default=None,
        description="Optional retrieval mode override: 'vector_only', 'hybrid', 'hybrid_rerank'",
    )


class RAGResponse(BaseModel):
    """Structured response containing grounded answer and verified citations."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., description="Original question submitted")
    answer: str = Field(
        ...,
        description="Synthesized answer text with embedded [1], [2] citation markers",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Structured citations corresponding to cited sources in the answer",
    )
    context_found: bool = Field(
        default=True,
        description="False if the model indicated insufficient context or answered 'I don't know'",
    )
    retrieval_metrics: RetrievalMetrics | None = Field(
        default=None,
        description="Detailed latency metrics per retrieval pipeline stage",
    )


# ---------------------------------------------------------------------------
# SSE (Server-Sent Events) schemas for streaming chat
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Request payload for the streaming SSE chat endpoint."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(
        ..., min_length=1, description="Natural language question to answer"
    )
    k: int = Field(
        default=4, ge=1, le=20, description="Number of context passages to retrieve"
    )
    document_id: str | None = Field(
        default=None, description="Optional document ID to scope retrieval"
    )
    user_id: str | None = Field(
        default=None,
        description="Optional user ID for multi-tenant isolation (defaults to 'anonymous')",
    )
    retrieval_mode: str | None = Field(
        default=None,
        description="Optional retrieval mode override: 'vector_only', 'hybrid', 'hybrid_rerank'",
    )


class ChatSSETokenEvent(BaseModel):
    """SSE data payload for a single streamed token."""

    model_config = ConfigDict(frozen=True)

    token: str = Field(..., description="Single token or text fragment from the LLM")


class ChatSSECitationsEvent(BaseModel):
    """SSE data payload containing resolved citations after streaming completes."""

    model_config = ConfigDict(frozen=True)

    citations: list[Citation] = Field(
        default_factory=list,
        description="Resolved source citations extracted from the streamed answer",
    )
    context_found: bool = Field(
        default=True,
        description="False if the model indicated insufficient context",
    )
    retrieval_metrics: RetrievalMetrics | None = Field(
        default=None,
        description="Detailed latency metrics per retrieval pipeline stage",
    )


class ChatSSEDoneEvent(BaseModel):
    """SSE data payload signalling the end of the streaming response."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Full accumulated answer text")


class ChatSSEErrorEvent(BaseModel):
    """SSE data payload for error conditions during streaming."""

    model_config = ConfigDict(frozen=True)

    detail: str = Field(..., description="Human-readable error description")


class ChatHistoryEntry(BaseModel):
    """Persisted record of a completed chat Q&A exchange."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Unique identifier for this chat entry")
    query: str = Field(..., description="User question")
    answer: str = Field(..., description="Model-generated answer")
    citations: list[Citation] = Field(
        default_factory=list, description="Resolved citations"
    )
    context_found: bool = Field(default=True, description="Whether context was found")
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of the exchange",
    )
