"""In-memory document metadata and chat history stores.

Thread-safe singleton stores used as a temporary persistence layer until
PostgreSQL + SQLAlchemy models are wired up. The interface is intentionally
simple so swapping to a real database requires minimal code changes.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.schemas.chat import ChatHistoryEntry, Citation

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    """Internal metadata record for a tracked document."""

    document_id: str
    filename: str
    user_id: str
    status: str = "processing"  # "processing" | "ready" | "failed"
    chunk_count: int = 0
    error_message: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class InMemoryDocumentStore:
    """Thread-safe in-memory store for document metadata.

    Provides CRUD operations for document records tracked during upload,
    ingestion, and deletion workflows.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._documents: dict[str, DocumentRecord] = {}

    def add_document(
        self,
        filename: str,
        user_id: str = "anonymous",
        document_id: str | None = None,
    ) -> DocumentRecord:
        """Register a new document and return its record.

        Args:
            filename: Original uploaded filename.
            user_id: Owner user ID. Defaults to 'anonymous'.
            document_id: Optional pre-generated ID. Auto-generated if None.

        Returns:
            DocumentRecord: The newly created record.
        """
        doc_id = document_id or str(uuid.uuid4())
        record = DocumentRecord(
            document_id=doc_id,
            filename=filename,
            user_id=user_id,
        )
        with self._lock:
            self._documents[doc_id] = record
        logger.info("Registered document %s (%s) for user %s", doc_id, filename, user_id)
        return record

    def get_document(self, document_id: str) -> DocumentRecord | None:
        """Retrieve a document record by ID.

        Args:
            document_id: Unique document identifier.

        Returns:
            DocumentRecord | None: The record, or None if not found.
        """
        with self._lock:
            return self._documents.get(document_id)

    def list_documents(self, user_id: str | None = None) -> list[DocumentRecord]:
        """List all documents, optionally filtered by user.

        Args:
            user_id: If provided, only return documents owned by this user.

        Returns:
            list[DocumentRecord]: Matching document records, newest first.
        """
        with self._lock:
            records = list(self._documents.values())
        if user_id:
            records = [r for r in records if r.user_id == user_id]
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    def update_status(
        self,
        document_id: str,
        status: str,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> bool:
        """Update the ingestion status of a document.

        Args:
            document_id: Unique document identifier.
            status: New status value ('processing', 'ready', 'failed').
            chunk_count: Number of indexed chunks (set on success).
            error_message: Error description (set on failure).

        Returns:
            bool: True if the document was found and updated.
        """
        with self._lock:
            record = self._documents.get(document_id)
            if record is None:
                return False
            record.status = status
            if chunk_count is not None:
                record.chunk_count = chunk_count
            if error_message is not None:
                record.error_message = error_message
        logger.info("Updated document %s status to '%s'", document_id, status)
        return True

    def delete_document(self, document_id: str) -> DocumentRecord | None:
        """Remove a document record from the store.

        Args:
            document_id: Unique document identifier.

        Returns:
            DocumentRecord | None: The deleted record, or None if not found.
        """
        with self._lock:
            return self._documents.pop(document_id, None)

    def clear(self) -> None:
        """Remove all document records. Useful for testing."""
        with self._lock:
            self._documents.clear()


class InMemoryChatStore:
    """Thread-safe in-memory store for chat history entries.

    Stores completed Q&A exchanges for retrieval via the history endpoint.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[ChatHistoryEntry] = []

    def add_entry(
        self,
        query: str,
        answer: str,
        citations: list[Citation] | None = None,
        context_found: bool = True,
        entry_id: str | None = None,
    ) -> ChatHistoryEntry:
        """Record a completed chat exchange.

        Args:
            query: User question.
            answer: Model-generated answer.
            citations: Resolved source citations.
            context_found: Whether sufficient context was available.
            entry_id: Optional pre-generated ID. Auto-generated if None.

        Returns:
            ChatHistoryEntry: The persisted entry.
        """
        entry = ChatHistoryEntry(
            id=entry_id or str(uuid.uuid4()),
            query=query,
            answer=answer,
            citations=citations or [],
            context_found=context_found,
        )
        with self._lock:
            self._entries.append(entry)
        return entry

    def list_entries(
        self,
        limit: int = 20,
        offset: int = 0,
        user_id: str | None = None,
    ) -> list[ChatHistoryEntry]:
        """Retrieve chat history entries with pagination.

        Args:
            limit: Maximum entries to return. Defaults to 20.
            offset: Number of entries to skip. Defaults to 0.
            user_id: Unused placeholder for future multi-tenant filtering.

        Returns:
            list[ChatHistoryEntry]: Paginated entries, newest first.
        """
        with self._lock:
            # Return newest first
            ordered = list(reversed(self._entries))
        return ordered[offset : offset + limit]

    def total_count(self) -> int:
        """Return total number of chat history entries.

        Returns:
            int: Total entry count.
        """
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        """Remove all chat entries. Useful for testing."""
        with self._lock:
            self._entries.clear()


# ---------------------------------------------------------------------------
# Module-level singleton instances
# ---------------------------------------------------------------------------

document_store = InMemoryDocumentStore()
chat_store = InMemoryChatStore()
