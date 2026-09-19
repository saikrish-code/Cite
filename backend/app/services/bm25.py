"""BM25 sparse keyword search service for lexical passage retrieval.

Implements Okapi BM25 with Lucene-standard non-negative IDF formulation:
    IDF(q) = ln(1 + (N - n + 0.5) / (n + 0.5))
    score(D, Q) = sum_{q in Q} IDF(q) * [f(q, D) * (k1 + 1)] / [f(q, D) + k1 * (1 - b + b * (|D| / avgdl))]

This formulation guarantees non-negative scores even on small document sets (N <= 3),
delivering exact term matching and frequency-based ranking to complement dense vector search.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any

from app.core.config import settings
from app.schemas.document import VectorSearchResult

logger = logging.getLogger(__name__)

# Basic alphanumeric word tokenizer
TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize_text(text: str) -> list[str]:
    """Tokenize a text string into lowercased lexical tokens.

    Args:
        text: Input string.

    Returns:
        list[str]: Cleaned, lowercased word tokens.
    """
    if not text:
        return []
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


class BM25OkapiScorer:
    """Okapi BM25 scorer with non-negative Lucene IDF smoothing."""

    def __init__(
        self,
        corpus: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avg_doc_len = (
            sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 1.0
        )
        self.doc_freqs: list[Counter[str]] = [Counter(doc) for doc in corpus]

        # Calculate non-negative document frequencies and IDFs
        df: Counter[str] = Counter()
        for doc in corpus:
            df.update(set(doc))

        self.idf: dict[str, float] = {
            term: math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def get_scores(self, query: list[str]) -> list[float]:
        """Calculate BM25 relevance scores for all documents given tokenized query."""
        scores = [0.0] * self.corpus_size
        for term in query:
            if term not in self.idf:
                continue
            term_idf = self.idf[term]
            for i, doc_freq in enumerate(self.doc_freqs):
                freq = doc_freq.get(term, 0)
                if freq > 0:
                    numerator = freq * (self.k1 + 1.0)
                    denominator = freq + self.k1 * (
                        1.0 - self.b + self.b * (self.doc_lengths[i] / self.avg_doc_len)
                    )
                    scores[i] += term_idf * (numerator / denominator)
        return scores


class BM25SearchService:
    """Sparse lexical search service using Okapi BM25.

    Args:
        k1: BM25 term frequency saturation parameter. Defaults to settings.BM25_K1 (1.5).
        b: BM25 document length normalization parameter. Defaults to settings.BM25_B (0.75).
    """

    def __init__(
        self,
        k1: float | None = None,
        b: float | None = None,
    ) -> None:
        self.k1 = k1 if k1 is not None else settings.BM25_K1
        self.b = b if b is not None else settings.BM25_B

    def search(
        self,
        query: str,
        chunks: list[VectorSearchResult],
        top_k: int = 20,
    ) -> list[VectorSearchResult]:
        """Perform BM25 keyword search across a provided candidate pool of chunks.

        Args:
            query: Natural language or keyword query string.
            chunks: Candidate pool of VectorSearchResult objects (pre-scoped by user_id).
            top_k: Maximum number of top matching passages to return.

        Returns:
            list[VectorSearchResult]: Ranked list of results ordered by descending BM25 score.
        """
        cleaned_query = query.strip()
        tokenized_query = tokenize_text(cleaned_query)

        if not tokenized_query or not chunks:
            return []

        # Tokenize corpus
        tokenized_corpus = [tokenize_text(chunk.text) for chunk in chunks]

        # Guard against corpus where all documents are completely empty
        if not any(tokenized_corpus):
            return []

        try:
            scorer = BM25OkapiScorer(tokenized_corpus, k1=self.k1, b=self.b)
            scores = scorer.get_scores(tokenized_query)
        except Exception as exc:
            logger.warning("BM25 scoring failed: %s", exc)
            return []

        # Pair each chunk with its BM25 score
        scored_results: list[tuple[float, VectorSearchResult]] = []
        for score, chunk in zip(scores, chunks):
            # Only include passages that have at least some term overlap (score > 0)
            if score > 0.0:
                scored_chunk = VectorSearchResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=round(float(score), 4),
                )
                scored_results.append((float(score), scored_chunk))

        # Sort descending by BM25 score
        scored_results.sort(key=lambda item: item[0], reverse=True)

        return [res for _, res in scored_results[:top_k]]
