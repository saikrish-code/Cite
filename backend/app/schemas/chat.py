"""Pydantic schemas for RAG chat and citation payloads."""

from __future__ import annotations

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
