"""Text chunking strategies for splitting parsed documents into retrieval units.

This module defines an abstract ``BaseChunker`` protocol and a concrete
``RecursiveTokenChunker`` implementation. The chunker is designed to be
swappable: you can register additional strategies (semantic chunking,
sentence-window, etc.) and compare retrieval quality across them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.document import ChunkMetadata, ParsedDocument, TextChunk


class BaseChunker(ABC):
    """Abstract interface for document chunking strategies.

    Subclass this to implement alternative chunking algorithms
    (e.g., sentence-window, semantic clustering, or agentic chunking)
    and swap them in the ingestion pipeline without changing callers.
    """

    @abstractmethod
    def chunk(self, document: ParsedDocument) -> list[TextChunk]:
        """Split a parsed document into retrieval-ready text chunks.

        Args:
            document: A fully parsed document with page-level text.

        Returns:
            list[TextChunk]: Ordered list of chunks with provenance metadata.
        """


class RecursiveTokenChunker(BaseChunker):
    """Recursive character-level chunker with configurable size and overlap.

    Splits text hierarchically using a sequence of separators, attempting
    the most semantically meaningful boundary first (double newline for
    paragraph breaks, then single newline, then sentence-ending period,
    then space, then character-level). This preserves natural paragraph
    and sentence structure as much as possible.

    Args:
        chunk_size: Target maximum number of characters per chunk.
            Default 800 corresponds to roughly 200 tokens (at ~4 chars/token
            for English), which sits in the optimal retrieval window for
            dense embedding models like BGE and MiniLM.
        chunk_overlap: Number of characters to overlap between consecutive
            chunks. Default 100 (~25 tokens) provides enough context
            continuity at boundaries to avoid splitting key sentences.
        separators: Ordered list of separator strings to try when splitting.
            Tried from first (most semantic) to last (least semantic).
    """

    DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, document: ParsedDocument) -> list[TextChunk]:
        """Split every page of the document into overlapping text chunks.

        Chunks are produced per-page so that page_number metadata is always
        accurate. The chunk_index is global across the entire document.

        Args:
            document: Parsed document with page-level text.

        Returns:
            list[TextChunk]: Chunks ordered by page, then position.
        """
        all_chunks: list[TextChunk] = []
        global_chunk_index = 0

        for page in document.pages:
            text = page.text
            if not text.strip():
                continue

            raw_chunks = self._recursive_split(text, self.separators)
            merged = self._merge_with_overlap(raw_chunks)

            # Track character offsets for each merged chunk
            search_start = 0
            for chunk_text in merged:
                char_start = text.find(chunk_text.strip()[:50], search_start)
                if char_start == -1:
                    char_start = search_start
                char_end = char_start + len(chunk_text)
                search_start = max(search_start, char_start + 1)

                all_chunks.append(
                    TextChunk(
                        text=chunk_text,
                        metadata=ChunkMetadata(
                            source_filename=document.filename,
                            page_number=page.page_number,
                            chunk_index=global_chunk_index,
                            char_start=char_start,
                            char_end=char_end,
                        ),
                    )
                )
                global_chunk_index += 1

        return all_chunks

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the most semantic separator available.

        Tries the first separator; any segment still exceeding chunk_size
        is recursively split with the remaining (finer-grained) separators.

        Args:
            text: Text to split.
            separators: Remaining separators to try, ordered coarse to fine.

        Returns:
            list[str]: Segments each at most chunk_size characters.
        """
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # Last resort: hard-cut at chunk_size
            return self._hard_split(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return self._hard_split(text)

        splits = text.split(separator)
        segments: list[str] = []
        current = ""

        for piece in splits:
            candidate = piece if not current else current + separator + piece

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    segments.append(current)
                if len(piece) > self.chunk_size:
                    # Piece itself is too large: recurse with finer separators
                    sub_segments = self._recursive_split(piece, remaining_separators)
                    segments.extend(sub_segments)
                    current = ""
                else:
                    current = piece

        if current:
            segments.append(current)

        return segments

    def _hard_split(self, text: str) -> list[str]:
        """Split text at exact chunk_size boundaries as a last resort.

        Args:
            text: Text to split.

        Returns:
            list[str]: Segments of at most chunk_size characters.
        """
        return [
            text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)
        ]

    def _merge_with_overlap(self, segments: list[str]) -> list[str]:
        """Merge segments and apply sliding-window overlap.

        Consecutive chunks share ``chunk_overlap`` characters at the
        boundary so that retrieval does not miss information split
        across chunk edges.

        Args:
            segments: Pre-split text segments.

        Returns:
            list[str]: Merged chunks with overlapping context.
        """
        if not segments:
            return []

        if self.chunk_overlap == 0:
            return [s for s in segments if s.strip()]

        merged: list[str] = []
        for segment in segments:
            if not segment.strip():
                continue

            if merged and self.chunk_overlap > 0:
                prev = merged[-1]
                overlap_text = prev[-self.chunk_overlap :]
                candidate = overlap_text + segment
                if len(candidate) <= self.chunk_size:
                    merged.append(candidate)
                else:
                    merged.append(segment)
            else:
                merged.append(segment)

        return merged
