"""Document ingestion pipeline: parsing, chunking, and orchestration.

This module ties together the parser and chunker components into a
single ``IngestionService`` that accepts raw file bytes and produces
retrieval-ready text chunks with full provenance metadata.
"""

from __future__ import annotations

from app.schemas.document import ParsedDocument, TextChunk
from app.services.chunker import BaseChunker, RecursiveTokenChunker
from app.services.parsers import BaseParser, get_parser


class IngestionService:
    """Orchestrates document ingestion: parsing → chunking.

    The service is configured with a chunker strategy at construction time.
    The parser is selected automatically based on the file extension.

    Args:
        chunker: Chunking strategy to use. Defaults to
            ``RecursiveTokenChunker`` with 800-char chunks and 100-char overlap.
    """

    def __init__(self, chunker: BaseChunker | None = None) -> None:
        self.chunker = chunker or RecursiveTokenChunker()

    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        """Parse a document file into structured page content.

        Args:
            file_bytes: Raw file content.
            filename: Original filename with extension (used to select parser).

        Returns:
            ParsedDocument: Parsed document with page-level text.

        Raises:
            ValueError: If the file type is unsupported or the document is empty.
        """
        parser: BaseParser = get_parser(filename)
        return parser.parse(file_bytes, filename)

    def chunk(self, document: ParsedDocument) -> list[TextChunk]:
        """Chunk a parsed document into retrieval units.

        Args:
            document: Previously parsed document.

        Returns:
            list[TextChunk]: Ordered chunks with provenance metadata.
        """
        return self.chunker.chunk(document)

    def ingest(self, file_bytes: bytes, filename: str) -> list[TextChunk]:
        """Full ingestion pipeline: parse then chunk.

        This is the primary entry point for the ingestion service.

        Args:
            file_bytes: Raw file content.
            filename: Original filename with extension.

        Returns:
            list[TextChunk]: Retrieval-ready chunks with metadata.

        Raises:
            ValueError: If the file type is unsupported or the document is empty.
        """
        document = self.parse(file_bytes, filename)
        return self.chunk(document)
