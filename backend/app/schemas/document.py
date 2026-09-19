"""Pydantic schemas for document ingestion data models."""

from pydantic import BaseModel, ConfigDict, Field


class ChunkMetadata(BaseModel):
    """Metadata attached to every text chunk for citation traceability."""

    model_config = ConfigDict(frozen=True)

    source_filename: str = Field(
        ..., description="Original filename of the uploaded document"
    )
    page_number: int = Field(
        ..., ge=1, description="1-indexed page number where this chunk originates"
    )
    chunk_index: int = Field(
        ..., ge=0, description="0-indexed sequential chunk position within the document"
    )
    char_start: int = Field(
        ..., ge=0, description="Start character offset within the page text"
    )
    char_end: int = Field(
        ..., ge=0, description="End character offset within the page text"
    )


class TextChunk(BaseModel):
    """A single chunk of text with its provenance metadata."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(..., min_length=1, description="The chunk text content")
    metadata: ChunkMetadata = Field(..., description="Provenance metadata")


class PageContent(BaseModel):
    """Extracted text content from a single page of a document."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(..., ge=1, description="1-indexed page number")
    text: str = Field(..., description="Full text content of the page")


class ParsedDocument(BaseModel):
    """Complete parsed representation of an uploaded document."""

    model_config = ConfigDict(frozen=True)

    filename: str = Field(..., description="Original filename")
    total_pages: int = Field(..., ge=1, description="Total number of pages")
    pages: list[PageContent] = Field(..., description="Ordered list of page contents")


class VectorChunkMetadata(BaseModel):
    """Metadata persisted in the vector store for citation and provenance tracking."""

    model_config = ConfigDict(frozen=True)

    user_id: str = Field(..., description="ID of the document owner")
    document_id: str = Field(..., description="Unique ID of the parent document")
    filename: str = Field(..., description="Original filename of the document")
    page: int = Field(..., ge=1, description="1-indexed page number")
    chunk_index: int = Field(
        ..., ge=0, description="0-indexed sequential chunk position within document"
    )
    char_start: int | None = Field(
        default=None, description="Start character offset within page"
    )
    char_end: int | None = Field(
        default=None, description="End character offset within page"
    )

    @classmethod
    def from_chunk(
        cls, chunk: TextChunk, document_id: str, user_id: str
    ) -> "VectorChunkMetadata":
        """Construct vector metadata from an ingestion TextChunk.

        Args:
            chunk: Text chunk produced by chunker.
            document_id: Unique document identifier.
            user_id: Owner user identifier.

        Returns:
            VectorChunkMetadata: Structured vector store metadata.
        """
        return cls(
            user_id=user_id,
            document_id=document_id,
            filename=chunk.metadata.source_filename,
            page=chunk.metadata.page_number,
            chunk_index=chunk.metadata.chunk_index,
            char_start=chunk.metadata.char_start,
            char_end=chunk.metadata.char_end,
        )

    def to_chroma_dict(self) -> dict[str, str | int | float | bool]:
        """Convert metadata to flat dictionary compatible with ChromaDB.

        Returns:
            dict[str, str | int | float | bool]: Primitive scalar mapping.
        """
        payload: dict[str, str | int | float | bool] = {
            "user_id": self.user_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "page": self.page,
            "chunk_index": self.chunk_index,
        }
        if self.char_start is not None:
            payload["char_start"] = self.char_start
        if self.char_end is not None:
            payload["char_end"] = self.char_end
        return payload


class VectorSearchResult(BaseModel):
    """A single retrieved chunk match from vector similarity search."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(
        ..., description="Unique chunk identifier in the vector store"
    )
    text: str = Field(..., description="Text content of the retrieved chunk")
    metadata: VectorChunkMetadata = Field(
        ..., description="Provenance metadata for the chunk"
    )
    score: float = Field(
        ..., description="Relevance similarity score (higher is more relevant)"
    )
