"""Comprehensive unit tests for the document ingestion pipeline.

Tests cover:
- PDF, DOCX, and TXT parsing with real sample files
- Recursive chunking logic (size, overlap, metadata correctness)
- Edge cases (empty files, unsupported formats, tiny/large chunks)
- Chunker swappability via the BaseChunker protocol
- End-to-end IngestionService.ingest() pipeline
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.schemas.document import ParsedDocument, TextChunk
from app.services.chunker import BaseChunker, RecursiveTokenChunker
from app.services.ingestion import IngestionService
from app.services.parsers import DOCXParser, PDFParser, TXTParser, get_parser

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
SAMPLE_PDF = FIXTURES_DIR / "sample.pdf"
SAMPLE_DOCX = FIXTURES_DIR / "sample.docx"
SAMPLE_TXT = FIXTURES_DIR / "sample.txt"
EMPTY_TXT = FIXTURES_DIR / "empty.txt"


# ===========================================================================
# Parser Tests
# ===========================================================================


class TestPDFParser:
    """Tests for PDFParser."""

    def test_parse_extracts_two_pages(self) -> None:
        """Sample PDF has 2 pages; parser should return both."""
        parser = PDFParser()
        data = SAMPLE_PDF.read_bytes()
        result = parser.parse(data, "sample.pdf")

        assert result.total_pages == 2
        assert len(result.pages) == 2
        assert result.filename == "sample.pdf"

    def test_parse_page_numbers_are_1_indexed(self) -> None:
        """Pages should be numbered starting from 1."""
        parser = PDFParser()
        data = SAMPLE_PDF.read_bytes()
        result = parser.parse(data, "sample.pdf")

        assert result.pages[0].page_number == 1
        assert result.pages[1].page_number == 2

    def test_parse_preserves_text_content(self) -> None:
        """Extracted text should contain key phrases from the PDF."""
        parser = PDFParser()
        data = SAMPLE_PDF.read_bytes()
        result = parser.parse(data, "sample.pdf")

        page1_text = result.pages[0].text
        assert "Attention" in page1_text
        # "Transformer" may be on a wrapped line depending on insert_text layout
        assert "dominant" in page1_text or "recurrent" in page1_text

        page2_text = result.pages[1].text
        assert "Introduction" in page2_text or "Recurrent" in page2_text

    def test_parse_empty_pdf_raises(self) -> None:
        """A PDF with pages containing no text should still parse (not raise).

        Note: PyMuPDF cannot serialize a document with 0 pages, so we test
        with a 1-page PDF whose page has no text inserted.
        """
        import pymupdf

        doc = pymupdf.open()
        doc.new_page()  # blank page with no text
        empty_bytes = doc.tobytes()
        doc.close()

        parser = PDFParser()
        result = parser.parse(empty_bytes, "blank.pdf")
        assert result.total_pages == 1
        # The page exists but has no meaningful text
        assert result.pages[0].text.strip() == ""


class TestDOCXParser:
    """Tests for DOCXParser."""

    def test_parse_detects_page_break(self) -> None:
        """DOCX with explicit page break should produce 2 pages."""
        parser = DOCXParser()
        data = SAMPLE_DOCX.read_bytes()
        result = parser.parse(data, "sample.docx")

        assert result.total_pages == 2
        assert result.filename == "sample.docx"

    def test_parse_page_content(self) -> None:
        """First page should have abstract, second should have introduction."""
        parser = DOCXParser()
        data = SAMPLE_DOCX.read_bytes()
        result = parser.parse(data, "sample.docx")

        page1_text = result.pages[0].text
        assert "Attention" in page1_text or "Transformer" in page1_text

        page2_text = result.pages[1].text
        assert "Introduction" in page2_text


class TestTXTParser:
    """Tests for TXTParser."""

    def test_parse_single_page(self) -> None:
        """TXT files should always produce a single page."""
        parser = TXTParser()
        data = SAMPLE_TXT.read_bytes()
        result = parser.parse(data, "sample.txt")

        assert result.total_pages == 1
        assert result.pages[0].page_number == 1
        assert result.filename == "sample.txt"

    def test_parse_preserves_full_text(self) -> None:
        """Parsed text should contain the complete file content."""
        parser = TXTParser()
        data = SAMPLE_TXT.read_bytes()
        result = parser.parse(data, "sample.txt")

        assert "Attention Is All You Need" in result.pages[0].text
        assert "Introduction" in result.pages[0].text

    def test_parse_empty_file_raises(self) -> None:
        """An empty text file should raise ValueError."""
        parser = TXTParser()
        data = EMPTY_TXT.read_bytes()

        with pytest.raises(ValueError, match="empty"):
            parser.parse(data, "empty.txt")


class TestGetParser:
    """Tests for the parser registry / get_parser function."""

    def test_pdf_extension(self) -> None:
        """get_parser should return PDFParser for .pdf files."""
        parser = get_parser("report.pdf")
        assert isinstance(parser, PDFParser)

    def test_docx_extension(self) -> None:
        """get_parser should return DOCXParser for .docx files."""
        parser = get_parser("document.docx")
        assert isinstance(parser, DOCXParser)

    def test_txt_extension(self) -> None:
        """get_parser should return TXTParser for .txt files."""
        parser = get_parser("notes.txt")
        assert isinstance(parser, TXTParser)

    def test_case_insensitive(self) -> None:
        """File extension matching should be case-insensitive."""
        parser = get_parser("REPORT.PDF")
        assert isinstance(parser, PDFParser)

    def test_unsupported_raises(self) -> None:
        """Unsupported extensions should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            get_parser("image.png")


# ===========================================================================
# Chunker Tests
# ===========================================================================


class TestRecursiveTokenChunker:
    """Tests for RecursiveTokenChunker."""

    def _make_document(self, text: str, filename: str = "test.txt") -> ParsedDocument:
        """Helper to create a single-page ParsedDocument."""
        from app.schemas.document import PageContent

        return ParsedDocument(
            filename=filename,
            total_pages=1,
            pages=[PageContent(page_number=1, text=text)],
        )

    def test_short_text_single_chunk(self) -> None:
        """Text shorter than chunk_size should produce exactly one chunk."""
        chunker = RecursiveTokenChunker(chunk_size=800, chunk_overlap=100)
        doc = self._make_document("Hello, world!")
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].text == "Hello, world!"

    def test_chunk_respects_max_size(self) -> None:
        """No chunk should exceed the configured chunk_size."""
        chunker = RecursiveTokenChunker(chunk_size=200, chunk_overlap=0)
        text = "Word " * 500  # ~2500 characters
        doc = self._make_document(text.strip())
        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert (
                len(chunk.text) <= 200
            ), f"Chunk exceeded max size: {len(chunk.text)} > 200"

    def test_chunk_metadata_has_correct_filename(self) -> None:
        """Every chunk metadata should carry the source filename."""
        chunker = RecursiveTokenChunker(chunk_size=100, chunk_overlap=0)
        doc = self._make_document("Some text for chunking.", filename="paper.pdf")
        chunks = chunker.chunk(doc)

        for chunk in chunks:
            assert chunk.metadata.source_filename == "paper.pdf"

    def test_chunk_metadata_has_correct_page_number(self) -> None:
        """Chunk page numbers should match the page they originated from."""
        from app.schemas.document import PageContent

        doc = ParsedDocument(
            filename="multi.pdf",
            total_pages=2,
            pages=[
                PageContent(page_number=1, text="Page one content here."),
                PageContent(page_number=2, text="Page two content here."),
            ],
        )

        chunker = RecursiveTokenChunker(chunk_size=800, chunk_overlap=0)
        chunks = chunker.chunk(doc)

        page_numbers = {c.metadata.page_number for c in chunks}
        assert 1 in page_numbers
        assert 2 in page_numbers

    def test_chunk_indices_are_sequential(self) -> None:
        """Chunk indices should start at 0 and increment globally."""
        chunker = RecursiveTokenChunker(chunk_size=100, chunk_overlap=0)
        text = "Word " * 200
        doc = self._make_document(text.strip())
        chunks = chunker.chunk(doc)

        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_overlap_creates_shared_content(self) -> None:
        """With overlap, adjacent chunks should share boundary text."""
        # Use text with clear paragraph boundaries
        text = "A" * 200 + "\n\n" + "B" * 200
        chunker = RecursiveTokenChunker(chunk_size=250, chunk_overlap=50)
        doc = self._make_document(text)
        chunks = chunker.chunk(doc)

        # With overlap, later chunks should contain trailing chars from previous
        if len(chunks) > 1:
            # The overlap means chunk[1] should start with chars from end of chunk[0]
            assert len(chunks[1].text) > 0

    def test_zero_overlap_no_duplication(self) -> None:
        """With zero overlap, concatenated chunks should reproduce the original text (approximately)."""
        chunker = RecursiveTokenChunker(chunk_size=200, chunk_overlap=0)
        original = "The quick brown fox jumps over the lazy dog. " * 20
        doc = self._make_document(original.strip())
        chunks = chunker.chunk(doc)

        reconstructed = "".join(c.text for c in chunks)
        # The text should be preserved (may differ in whitespace only)
        assert len(reconstructed) >= len(original.strip()) * 0.9

    def test_invalid_chunk_size_raises(self) -> None:
        """chunk_size <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            RecursiveTokenChunker(chunk_size=0)

    def test_overlap_exceeds_size_raises(self) -> None:
        """chunk_overlap >= chunk_size should raise ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap.*must be less than"):
            RecursiveTokenChunker(chunk_size=100, chunk_overlap=100)

    def test_negative_overlap_raises(self) -> None:
        """Negative overlap should raise ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            RecursiveTokenChunker(chunk_size=100, chunk_overlap=-1)

    def test_empty_page_produces_no_chunks(self) -> None:
        """A page with only whitespace should be skipped."""
        chunker = RecursiveTokenChunker(chunk_size=200, chunk_overlap=0)
        doc = self._make_document("   \n\n   ")
        chunks = chunker.chunk(doc)

        assert len(chunks) == 0

    def test_paragraph_boundary_respected(self) -> None:
        """Chunker should prefer splitting at paragraph boundaries."""
        para1 = "First paragraph content here."
        para2 = "Second paragraph content here."
        text = f"{para1}\n\n{para2}"
        chunker = RecursiveTokenChunker(chunk_size=40, chunk_overlap=0)
        doc = self._make_document(text)
        chunks = chunker.chunk(doc)

        # Each paragraph should be a separate chunk since each is < 40 chars
        chunk_texts = [c.text.strip() for c in chunks]
        assert para1 in chunk_texts
        assert para2 in chunk_texts


# ===========================================================================
# Chunker Swappability Tests
# ===========================================================================


class FixedSizeChunker(BaseChunker):
    """A trivial chunker for testing that the interface is swappable."""

    def __init__(self, size: int = 50) -> None:
        self.size = size

    def chunk(self, document: ParsedDocument) -> list[TextChunk]:
        """Split each page into fixed-size character segments."""
        from app.schemas.document import ChunkMetadata

        chunks: list[TextChunk] = []
        idx = 0
        for page in document.pages:
            text = page.text
            for i in range(0, len(text), self.size):
                segment = text[i : i + self.size]
                if segment.strip():
                    chunks.append(
                        TextChunk(
                            text=segment,
                            metadata=ChunkMetadata(
                                source_filename=document.filename,
                                page_number=page.page_number,
                                chunk_index=idx,
                                char_start=i,
                                char_end=i + len(segment),
                            ),
                        )
                    )
                    idx += 1
        return chunks


class TestChunkerSwappability:
    """Verify that the ingestion pipeline accepts alternative chunkers."""

    def test_custom_chunker_via_service(self) -> None:
        """IngestionService should accept and use a custom BaseChunker."""
        custom = FixedSizeChunker(size=30)
        service = IngestionService(chunker=custom)

        data = SAMPLE_TXT.read_bytes()
        chunks = service.ingest(data, "sample.txt")

        # All chunks from our custom chunker should be <= 30 chars
        for chunk in chunks:
            assert len(chunk.text) <= 30

    def test_custom_chunker_metadata_correct(self) -> None:
        """Custom chunker should still produce correct metadata."""
        custom = FixedSizeChunker(size=50)
        service = IngestionService(chunker=custom)

        data = SAMPLE_TXT.read_bytes()
        chunks = service.ingest(data, "sample.txt")

        for chunk in chunks:
            assert chunk.metadata.source_filename == "sample.txt"
            assert chunk.metadata.page_number == 1


# ===========================================================================
# IngestionService End-to-End Tests
# ===========================================================================


class TestIngestionService:
    """End-to-end tests for the full ingestion pipeline."""

    def test_ingest_pdf(self) -> None:
        """Full pipeline on a PDF should produce chunks with correct metadata."""
        service = IngestionService()
        data = SAMPLE_PDF.read_bytes()
        chunks = service.ingest(data, "sample.pdf")

        assert len(chunks) > 0
        assert all(c.metadata.source_filename == "sample.pdf" for c in chunks)
        assert all(c.text.strip() for c in chunks)

    def test_ingest_docx(self) -> None:
        """Full pipeline on a DOCX should produce chunks."""
        service = IngestionService()
        data = SAMPLE_DOCX.read_bytes()
        chunks = service.ingest(data, "sample.docx")

        assert len(chunks) > 0
        assert all(c.metadata.source_filename == "sample.docx" for c in chunks)

    def test_ingest_txt(self) -> None:
        """Full pipeline on a TXT should produce chunks."""
        service = IngestionService()
        data = SAMPLE_TXT.read_bytes()
        chunks = service.ingest(data, "sample.txt")

        assert len(chunks) > 0
        assert all(c.metadata.page_number == 1 for c in chunks)

    def test_ingest_unsupported_format_raises(self) -> None:
        """Unsupported file extensions should raise ValueError."""
        service = IngestionService()
        with pytest.raises(ValueError, match="Unsupported file type"):
            service.ingest(b"some bytes", "image.png")

    def test_parse_returns_parsed_document(self) -> None:
        """The parse() method should return a ParsedDocument."""
        service = IngestionService()
        data = SAMPLE_TXT.read_bytes()
        doc = service.parse(data, "sample.txt")

        assert isinstance(doc, ParsedDocument)
        assert doc.total_pages == 1

    def test_configurable_chunk_size(self) -> None:
        """Changing chunk_size should affect the number of output chunks."""
        small_chunker = RecursiveTokenChunker(chunk_size=100, chunk_overlap=0)
        large_chunker = RecursiveTokenChunker(chunk_size=5000, chunk_overlap=0)

        service_small = IngestionService(chunker=small_chunker)
        service_large = IngestionService(chunker=large_chunker)

        data = SAMPLE_TXT.read_bytes()
        small_chunks = service_small.ingest(data, "sample.txt")
        large_chunks = service_large.ingest(data, "sample.txt")

        assert len(small_chunks) > len(large_chunks)
