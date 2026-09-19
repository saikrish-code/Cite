"""Document management REST API endpoints with strict per-user multi-tenant isolation.

Provides:
- POST /documents/upload — Upload and ingest a document (background processing, user-scoped)
- GET /documents — List all tracked documents belonging to the authenticated user
- DELETE /documents/{document_id} — Delete an authenticated user's document and vector chunks
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.core.config import settings
from app.core.database import AsyncSessionLocal, get_db
from app.models.document import Document
from app.models.user import User
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
        HTTPException: If the file type is unsupported or file name is missing.
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
    """Read file content and validate size limits.

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


async def _update_document_status_db(
    document_id: str,
    status_str: str,
    chunk_count: int = 0,
    error_message: str | None = None,
) -> None:
    """Update document status directly in database from background task."""
    try:
        async with AsyncSessionLocal() as session:
            stmt = select(Document).where(Document.id == document_id)
            res = await session.execute(stmt)
            doc = res.scalar_one_or_none()
            if doc:
                doc.status = status_str
                doc.chunk_count = chunk_count
                doc.error_message = error_message
                await session.commit()
    except Exception as exc:
        logger.exception("Failed to update database status for doc %s: %s", document_id, exc)


async def _background_ingest(
    file_bytes: bytes,
    filename: str,
    document_id: str,
    user_id: str,
) -> None:
    """Background task: parse, chunk, embed, and index a document with user_id scoping.

    Updates the database and document store status to 'ready' on success or 'failed' on error.

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
            err = "No text content could be extracted from the document."
            await _update_document_status_db(document_id, "failed", error_message=err)
            document_store.update_status(
                document_id=document_id,
                status="failed",
                error_message=err,
            )
            return

        # Embed and index with strict user_id tag
        chunk_ids = vector_store.add_chunks(
            chunks=chunks,
            document_id=document_id,
            user_id=user_id,
        )

        await _update_document_status_db(document_id, "ready", chunk_count=len(chunk_ids))
        document_store.update_status(
            document_id=document_id,
            status="ready",
            chunk_count=len(chunk_ids),
        )
        logger.info(
            "Successfully ingested document %s (%s) for user %s: %d chunks indexed",
            document_id,
            filename,
            user_id,
            len(chunk_ids),
        )

    except Exception as exc:
        logger.exception("Failed to ingest document %s (%s)", document_id, filename)
        err = str(exc)
        await _update_document_status_db(document_id, "failed", error_message=err)
        document_store.update_status(
            document_id=document_id,
            status="failed",
            error_message=err,
        )


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload Document",
    description=(
        "Upload a PDF, DOCX, or TXT file for background ingestion. "
        "Strictly associated with the authenticated caller's account."
    ),
)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentUploadResponse:
    """Accept document upload and launch background ingestion for current user.

    Args:
        file: Uploaded file (multipart form data).
        background_tasks: FastAPI background task manager.
        current_user: Authenticated user.
        db: Scoped database session.

    Returns:
        DocumentUploadResponse: Confirmation with document_id and 'processing' status.
    """
    _validate_file(file)
    file_bytes = await _read_and_validate_size(file)

    document_id = str(uuid.uuid4())
    filename = file.filename or "unknown"
    _, ext = os.path.splitext(filename)

    # Persist in Database
    doc_record = Document(
        id=document_id,
        user_id=current_user.id,
        filename=filename,
        file_type=ext.lstrip(".").lower(),
        file_size_bytes=len(file_bytes),
        status="processing",
        chunk_count=0,
    )
    db.add(doc_record)
    await db.commit()

    # Register in document store
    document_store.add_document(
        filename=filename,
        user_id=current_user.id,
        document_id=document_id,
    )

    # Launch background ingestion with user isolation
    background_tasks.add_task(
        _background_ingest,
        file_bytes=file_bytes,
        filename=filename,
        document_id=document_id,
        user_id=current_user.id,
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
    description="Retrieve list of documents owned by the authenticated user.",
)
async def list_documents(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentListResponse:
    """List documents belonging exclusively to the authenticated caller."""
    stmt = (
        select(Document)
        .where(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
    )
    result = await db.execute(stmt)
    records = result.scalars().all()

    items = [
        DocumentListItem(
            document_id=r.id,
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
    description="Delete an authenticated user's document and its indexed vector chunks.",
)
async def delete_document(
    document_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DocumentDeleteResponse:
    """Delete a document and purge vector chunks with strict ownership check.

    Args:
        document_id: ID of document to delete.
        current_user: Authenticated caller.
        db: Scoped database session.

    Returns:
        DocumentDeleteResponse: Deletion confirmation.

    Raises:
        HTTPException: 404 if document does not exist or is not owned by current user.
    """
    stmt = select(Document).where(
        Document.id == document_id,
        Document.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()

    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    # Delete vector chunks strictly matching document_id AND user_id
    chunks_deleted = 0
    try:
        vector_store = ChromaVectorStore()
        chunks_deleted = vector_store.delete_by_document(
            document_id=document_id, user_id=current_user.id
        )
    except Exception as exc:
        logger.exception("Failed to delete vector chunks for document %s", document_id)

    # Delete from document store and DB
    document_store.delete_document(document_id)
    await db.delete(doc)
    await db.commit()

    return DocumentDeleteResponse(
        document_id=document_id,
        filename=doc.filename,
        chunks_deleted=chunks_deleted,
    )
