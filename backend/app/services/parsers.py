"""Document parsers: extract text page-by-page from PDF, DOCX, and TXT files."""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path

from app.schemas.document import PageContent, ParsedDocument


class BaseParser(ABC):
    """Abstract base class for document parsers.

    Each parser must extract text from a specific file format, returning
    a list of PageContent objects preserving the original page structure.
    """

    @abstractmethod
    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        """Parse a document from raw bytes into structured page content.

        Args:
            file_bytes: Raw bytes of the uploaded file.
            filename: Original filename including extension.

        Returns:
            ParsedDocument: Parsed pages with metadata.

        Raises:
            ValueError: If the document is empty or cannot be parsed.
        """


class PDFParser(BaseParser):
    """Extract text from PDF files page-by-page using PyMuPDF (fitz)."""

    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        """Parse a PDF document into page-level text content.

        Args:
            file_bytes: Raw PDF bytes.
            filename: Original PDF filename.

        Returns:
            ParsedDocument: Extracted pages with page numbers.

        Raises:
            ValueError: If the PDF has no pages or cannot be read.
        """
        import pymupdf

        doc = pymupdf.open(stream=file_bytes, filetype="pdf")

        if doc.page_count == 0:
            doc.close()
            raise ValueError(f"PDF '{filename}' contains no pages.")

        pages: list[PageContent] = []
        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)
            text = page.get_text("text")
            pages.append(PageContent(page_number=page_idx + 1, text=text))

        total_pages = doc.page_count
        doc.close()

        return ParsedDocument(
            filename=filename,
            total_pages=total_pages,
            pages=pages,
        )


class DOCXParser(BaseParser):
    """Extract text from DOCX files using python-docx.

    DOCX files do not have a native page concept; pagination is
    determined by the rendering engine. This parser treats each
    paragraph as belonging to page 1 unless explicit page-break
    markers (``w:lastRenderedPageBreak``) are found in the XML,
    in which case it increments the page counter.
    """

    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        """Parse a DOCX document, approximating page boundaries via page breaks.

        Args:
            file_bytes: Raw DOCX bytes.
            filename: Original DOCX filename.

        Returns:
            ParsedDocument: Extracted pages with best-effort page numbers.

        Raises:
            ValueError: If the document contains no text content.
        """
        import docx
        from docx.oxml.ns import qn

        doc = docx.Document(io.BytesIO(file_bytes))

        current_page = 1
        page_texts: dict[int, list[str]] = {1: []}

        for paragraph in doc.paragraphs:
            # Check for page breaks in the XML runs
            for run in paragraph.runs:
                if run._element.xml.find(qn("w:lastRenderedPageBreak")) != -1:
                    current_page += 1
                    if current_page not in page_texts:
                        page_texts[current_page] = []
                # Also check for explicit break elements in the run
                for br in run._element.findall(qn("w:br")):
                    if br.get(qn("w:type")) == "page":
                        current_page += 1
                        if current_page not in page_texts:
                            page_texts[current_page] = []

            if current_page not in page_texts:
                page_texts[current_page] = []
            page_texts[current_page].append(paragraph.text)

        pages = [
            PageContent(
                page_number=page_num,
                text="\n".join(lines),
            )
            for page_num, lines in sorted(page_texts.items())
        ]

        full_text = "".join(p.text for p in pages)
        if not full_text.strip():
            raise ValueError(f"DOCX '{filename}' contains no text content.")

        return ParsedDocument(
            filename=filename,
            total_pages=len(pages),
            pages=pages,
        )


class TXTParser(BaseParser):
    """Extract text from plain-text files.

    Plain text files have no pages. The entire content is treated as
    a single page (page_number=1) to maintain a consistent interface.
    """

    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        """Parse a plain-text file as a single-page document.

        Args:
            file_bytes: Raw text file bytes.
            filename: Original filename.

        Returns:
            ParsedDocument: Single-page document containing the full text.

        Raises:
            ValueError: If the text file is empty.
        """
        text = file_bytes.decode("utf-8", errors="replace")

        if not text.strip():
            raise ValueError(f"Text file '{filename}' is empty.")

        return ParsedDocument(
            filename=filename,
            total_pages=1,
            pages=[PageContent(page_number=1, text=text)],
        )


# ---------------------------------------------------------------------------
# Registry: file extension -> parser instance
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: dict[str, BaseParser] = {
    ".pdf": PDFParser(),
    ".docx": DOCXParser(),
    ".txt": TXTParser(),
}


def get_parser(filename: str) -> BaseParser:
    """Return the appropriate parser for a given filename.

    Args:
        filename: Original filename with extension.

    Returns:
        BaseParser: Parser instance for the file type.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = Path(filename).suffix.lower()
    parser = SUPPORTED_EXTENSIONS.get(ext)
    if parser is None:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS.keys()))
        raise ValueError(f"Unsupported file type '{ext}'. Supported types: {supported}")
    return parser
