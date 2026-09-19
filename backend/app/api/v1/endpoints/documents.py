"""Document management REST API endpoints.

Provides:
- POST /documents/upload — Upload and ingest a document (background processing)
- GET /documents — List all tracked documents
- DELETE /documents/{document_id} — Delete a document and its vector chunks
"""

from __future__ import annotations

import logging
import os
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, status

from app.core.config import settings
from app.schemas.api_document import (
    DocumentDeleteResponse,
    DocumentListItem,
    DocumentListResponse,
    DocumentUploadResponse,
)
from app.services.document_store import document_store
from app.services.ingestion import IngestionService
from app.services.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

router = APIRouter()


def _validate_file(file: UploadFile) -> None:
    """Validate uploaded file type and size.

    Args:
        file: FastAPI UploadFile instance.

    Raises:
        HTTPException: If the file type is unsupported or file exceeds size limit.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )

    _, ext = os.path.splitext(file.filename)
    ext = ext.lower()

    if ext not in settings.allowed_extensions_set:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed types: {', '.join(sorted(settings.allowed_extensions_set))}"
            ),
        )


async def _read_and_validate_size(file: UploadFile) -> bytes:
    """Read file content and validate size.

    Args:
        file: FastAPI UploadFile instance.

    Returns:
        bytes: Raw file content.

    Raises:
        HTTPException: If the file exceeds MAX_UPLOAD_SIZE_MB or is empty.
    """
    content = await file.read()

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size ({len(content) / (1024 * 1024):.1f} MB) exceeds "
                f"maximum allowed size ({settings.MAX_UPLOAD_SIZE_MB} MB)."
            ),
        )

    return content


def _background_ingest(
    file_bytes: bytes,
    filename: str,
    document_id: str,
    user_id: str,
) -> None:
    """Background task: parse, chunk, embed, and index a document.

    Updates the document store status to 'ready' on success or 'failed' on error.

    Args:
        file_bytes: Raw file content.
        filename: Original filename.
        document_id: Unique document identifier.
        user_id: Owner user ID.
    """
    try:
        ingestion_service = IngestionService()
        vector_store = ChromaVectorStore()

        # Parse and chunk
        chunks = ingestion_service.ingest(file_bytes, filename)

        if not chunks:
            document_store.update_status(
                document_id=document_id,
                status="failed",
                error_message="No text content could be extracted from the document.",
            )
            return

        # Embed and index
        chunk_ids = vector_store.add_chunks(
            chunks=chunks,
            document_id=document_id,
            user_id=user_id,
        )

        document_store.update_status(
            document_id=document_id,
            status="ready",
            chunk_count=len(chunk_ids),
        )
        logger.info(
            "Successfully ingested document %s (%s): %d chunks indexed",
            document_id,
            filename,
            len(chunk_ids),
        )

    except Exception as exc:
        logger.exception("Failed to ingest document %s (%s)", document_id, filename)
        document_store.update_status(
            document_id=document_id,
            status="failed",
            error_message=str(exc),
        )


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload Document",
    description=(
        "Upload a PDF, DOCX, or TXT file for ingestion. "
        "The document is processed in the background. Returns immediately with a "
        "document ID and 'processing' status."
    ),
)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    user_id: str = "anonymous",
) -> DocumentUploadResponse:
    """Accept a document upload and launch background ingestion.

    Args:
        file: Uploaded file (multipart form data).
        background_tasks: FastAPI background task manager.
        user_id: Optional owner user ID (query param, defaults to 'anonymous').

    Returns:
        DocumentUploadResponse: Confirmation with document_id and status.
    """
    _validate_file(file)
    file_bytes = await _read_and_validate_size(file)

    document_id = str(uuid.uuid4())
    filename = file.filename or "unknown"

    # Register in document store
    document_store.add_document(
        filename=filename,
        user_id=user_id,
        document_id=document_id,
    )

    # Launch background ingestion
    background_tasks.add_task(
        _background_ingest,
        file_bytes=file_bytes,
        filename=filename,
        document_id=document_id,
        user_id=user_id,
    )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        status="processing",
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Documents",
    description="Retrieve a list of all tracked documents with their ingestion status.",
)
async def list_documents(
    user_id: str | None = None,
) -> DocumentListResponse:
    """List all documents, optionally filtered by user.

    Args:
        user_id: Optional user ID filter (query param).

    Returns:
        DocumentListResponse: Paginated document list.
    """
    records = document_store.list_documents(user_id=user_id)
    items = [
        DocumentListItem(
            document_id=r.document_id,
            filename=r.filename,
            status=r.status,
            chunk_count=r.chunk_count,
            user_id=r.user_id,
            created_at=r.created_at,
        )
        for r in records
    ]
    return DocumentListResponse(documents=items, total=len(items))


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete Document",
    description="Delete a document and all its indexed vector chunks.",
)
async def delete_document(document_id: str) -> DocumentDeleteResponse:
    """Delete a document and its vector store chunks.

    Args:
        document_id: Unique document identifier (path param).

    Returns:
        DocumentDeleteResponse: Confirmation with chunk deletion count.

    Raises:
        HTTPException: 404 if document not found.
    """
    record = document_store.get_document(document_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    # Delete from vector store
    try:
        vector_store = ChromaVectorStore()
        chunks_deleted = vector_store.delete_by_document(document_id)
    except Exception as exc:
        logger.exception("Failed to delete vector chunks for document %s", document_id)
        chunks_deleted = 0

    # Remove from document store
    document_store.delete_document(document_id)

    return DocumentDeleteResponse(
        document_id=document_id,
        filename=record.filename,
        chunks_deleted=chunks_deleted,
    )
