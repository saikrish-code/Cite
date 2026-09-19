"""Integration tests for document management API endpoints."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_current_user
from app.core.database import AsyncSessionLocal, Base, engine
from app.main import app
from app.models.document import Document
from app.models.user import User
from app.services.document_store import document_store


@pytest.fixture(autouse=True)
async def _setup_db_and_auth():
    """Ensure DB schema and mock authenticated user for document endpoints."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    test_user = User(
        id="test-user-id",
        email="test_doc_user@university.edu",
        hashed_password="hashed_pwd",
        full_name="Doc Tester",
        is_active=True,
    )
    async with AsyncSessionLocal() as session:
        session.add(test_user)
        await session.commit()

    app.dependency_overrides[get_current_user] = lambda: test_user
    document_store.clear()
    yield
    document_store.clear()
    app.dependency_overrides.pop(get_current_user, None)


async def _add_test_doc(doc_id: str, filename: str, user_id: str = "test-user-id") -> None:
    async with AsyncSessionLocal() as session:
        doc = Document(
            id=doc_id,
            user_id=user_id,
            filename=filename,
            file_type="pdf",
            status="ready",
        )
        session.add(doc)
        await session.commit()
    document_store.add_document(filename=filename, user_id=user_id, document_id=doc_id)


@pytest.fixture
async def client() -> AsyncClient:
    """Provide an async HTTP client bound to the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


class TestDocumentUpload:
    """Tests for POST /api/v1/documents/upload."""

    async def test_upload_valid_pdf(self, client: AsyncClient) -> None:
        """Uploading a valid PDF returns 202 with document_id and processing status."""
        # Create a minimal valid-looking file
        fake_pdf = b"%PDF-1.4 fake content for testing purposes"

        with patch(
            "app.api.v1.endpoints.documents._background_ingest"
        ) as mock_ingest:
            response = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("test_paper.pdf", io.BytesIO(fake_pdf), "application/pdf")},
            )

        assert response.status_code == 202
        data = response.json()
        assert data["filename"] == "test_paper.pdf"
        assert data["status"] == "processing"
        assert "document_id" in data
        assert len(data["document_id"]) == 36  # UUID format

    async def test_upload_valid_txt(self, client: AsyncClient) -> None:
        """Uploading a valid TXT file returns 202."""
        fake_txt = b"This is a test document with some content."

        with patch(
            "app.api.v1.endpoints.documents._background_ingest"
        ):
            response = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("notes.txt", io.BytesIO(fake_txt), "text/plain")},
            )

        assert response.status_code == 202
        assert response.json()["filename"] == "notes.txt"

    async def test_upload_valid_docx(self, client: AsyncClient) -> None:
        """Uploading a valid DOCX file returns 202."""
        # DOCX files are ZIP-based, use minimal bytes for testing
        fake_docx = b"PK\x03\x04 fake docx content"

        with patch(
            "app.api.v1.endpoints.documents._background_ingest"
        ):
            response = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("report.docx", io.BytesIO(fake_docx), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            )

        assert response.status_code == 202
        assert response.json()["filename"] == "report.docx"

    async def test_upload_unsupported_extension_returns_400(self, client: AsyncClient) -> None:
        """Uploading a file with unsupported extension returns 400."""
        response = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("image.png", io.BytesIO(b"fake png"), "image/png")},
        )

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    async def test_upload_empty_file_returns_400(self, client: AsyncClient) -> None:
        """Uploading an empty file returns 400."""
        response = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
        )

        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    async def test_upload_oversized_file_returns_413(self, client: AsyncClient) -> None:
        """Uploading a file exceeding MAX_UPLOAD_SIZE_MB returns 413."""
        # Create content larger than the configured max (patch to 1 byte for test)
        with patch("app.api.v1.endpoints.documents.settings") as mock_settings:
            mock_settings.allowed_extensions_set = {".txt"}
            mock_settings.max_upload_bytes = 10  # 10 bytes limit
            mock_settings.MAX_UPLOAD_SIZE_MB = 0

            response = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("big.txt", io.BytesIO(b"x" * 100), "text/plain")},
            )

        assert response.status_code == 413
        assert "exceeds" in response.json()["detail"].lower()

    async def test_upload_with_custom_user_id(self, client: AsyncClient) -> None:
        """Upload automatically associates document with current authenticated user ID."""
        fake_txt = b"Content for user test."

        with patch(
            "app.api.v1.endpoints.documents._background_ingest"
        ):
            response = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("doc.txt", io.BytesIO(fake_txt), "text/plain")},
            )

        assert response.status_code == 202
        doc_id = response.json()["document_id"]
        record = document_store.get_document(doc_id)
        assert record is not None
        assert record.user_id == "test-user-id"

    async def test_upload_registers_in_document_store(self, client: AsyncClient) -> None:
        """Uploading a document registers it in the document store."""
        fake_txt = b"Content to index."

        with patch(
            "app.api.v1.endpoints.documents._background_ingest"
        ):
            response = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("paper.txt", io.BytesIO(fake_txt), "text/plain")},
            )

        doc_id = response.json()["document_id"]
        record = document_store.get_document(doc_id)
        assert record is not None
        assert record.filename == "paper.txt"
        assert record.status == "processing"


class TestDocumentList:
    """Tests for GET /api/v1/documents."""

    async def test_list_empty(self, client: AsyncClient) -> None:
        """Listing documents when none exist returns empty list."""
        response = await client.get("/api/v1/documents")

        assert response.status_code == 200
        data = response.json()
        assert data["documents"] == []
        assert data["total"] == 0

    async def test_list_after_upload(self, client: AsyncClient) -> None:
        """Listing documents after upload shows the registered document."""
        await _add_test_doc(
            filename="research.pdf",
            user_id="test-user-id",
            doc_id="test-doc-001",
        )

        response = await client.get("/api/v1/documents")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["documents"][0]["filename"] == "research.pdf"
        assert data["documents"][0]["document_id"] == "test-doc-001"

    async def test_list_filtered_by_user(self, client: AsyncClient) -> None:
        """Listing returns only matching user documents."""
        await _add_test_doc("doc-a", filename="a.pdf", user_id="test-user-id")
        await _add_test_doc("doc-b", filename="b.pdf", user_id="other-user")

        response = await client.get("/api/v1/documents")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["documents"][0]["user_id"] == "test-user-id"

    async def test_list_multiple_documents(self, client: AsyncClient) -> None:
        """Listing returns all registered documents."""
        for i in range(3):
            await _add_test_doc(f"doc-{i}", filename=f"doc_{i}.txt", user_id="test-user-id")

        response = await client.get("/api/v1/documents")

        assert response.status_code == 200
        assert response.json()["total"] == 3


class TestDocumentDelete:
    """Tests for DELETE /api/v1/documents/{document_id}."""

    async def test_delete_existing_document(self, client: AsyncClient) -> None:
        """Deleting an existing document returns success with chunk count."""
        await _add_test_doc("del-doc-1", filename="paper.pdf", user_id="test-user-id")

        with patch(
            "app.api.v1.endpoints.documents.ChromaVectorStore"
        ) as MockVectorStore:
            mock_instance = MagicMock()
            mock_instance.delete_by_document.return_value = 5
            MockVectorStore.return_value = mock_instance

            response = await client.delete("/api/v1/documents/del-doc-1")

        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "del-doc-1"
        assert data["filename"] == "paper.pdf"
        assert data["chunks_deleted"] == 5

        # Verify removed from store
        assert document_store.get_document("del-doc-1") is None

    async def test_delete_nonexistent_returns_404(self, client: AsyncClient) -> None:
        """Deleting a non-existent document returns 404."""
        response = await client.delete("/api/v1/documents/nonexistent-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_delete_handles_vector_store_error(self, client: AsyncClient) -> None:
        """Deletion proceeds even if vector store raises an error."""
        await _add_test_doc("err-doc", filename="broken.pdf", user_id="test-user-id")

        with patch(
            "app.api.v1.endpoints.documents.ChromaVectorStore"
        ) as MockVectorStore:
            mock_instance = MagicMock()
            mock_instance.delete_by_document.side_effect = RuntimeError("ChromaDB down")
            MockVectorStore.return_value = mock_instance

            response = await client.delete("/api/v1/documents/err-doc")

        assert response.status_code == 200
        data = response.json()
        assert data["chunks_deleted"] == 0

        # Document should still be removed from store
        assert document_store.get_document("err-doc") is None
