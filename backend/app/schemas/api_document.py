"""Pydantic schemas for document management REST API endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentUploadResponse(BaseModel):
    """Response payload returned immediately after a document upload is accepted."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(..., description="Unique identifier assigned to the uploaded document")
    filename: str = Field(..., description="Original filename of the uploaded document")
    status: str = Field(
        default="processing",
        description="Current ingestion status: 'processing', 'ready', or 'failed'",
    )
    message: str = Field(
        default="Document accepted for processing",
        description="Human-readable status message",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the upload was accepted",
    )


class DocumentListItem(BaseModel):
    """Summary representation of a single document for listing endpoints."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Ingestion status: 'processing', 'ready', or 'failed'")
    chunk_count: int = Field(default=0, ge=0, description="Number of indexed chunks")
    user_id: str = Field(default="anonymous", description="Owner user ID")
    created_at: datetime = Field(..., description="UTC timestamp of upload")


class DocumentListResponse(BaseModel):
    """Paginated response wrapping a list of documents."""

    model_config = ConfigDict(frozen=True)

    documents: list[DocumentListItem] = Field(
        default_factory=list, description="List of document summaries"
    )
    total: int = Field(default=0, ge=0, description="Total number of documents")


class DocumentDeleteResponse(BaseModel):
    """Confirmation response after deleting a document and its chunks."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(..., description="ID of the deleted document")
    filename: str = Field(..., description="Original filename of the deleted document")
    chunks_deleted: int = Field(
        default=0, ge=0, description="Number of vector store chunks removed"
    )
    message: str = Field(
        default="Document deleted successfully",
        description="Human-readable confirmation message",
    )
