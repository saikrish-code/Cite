"""Neural Cross-Encoder reranker service for deep passage re-scoring.

Takes top candidate passages retrieved via dense vector and lexical BM25 search
and performs joint cross-attention scoring using ms-marco-MiniLM-L-6-v2.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from abc import ABC, abstractmethod
from typing import Any

# Prevent broken torchcodec from attempting to load non-existent ffmpeg dlls on Windows/Python 3.14
if "torchcodec" not in sys.modules:
    sys.modules["torchcodec"] = None

from app.core.config import settings
from app.schemas.document import VectorSearchResult

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract interface defining the contract for neural passage rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[VectorSearchResult],
        top_k: int = 4,
    ) -> list[VectorSearchResult]:
        """Synchronously score and reorder candidate passages relative to the query.

        Args:
            query: Question or search query.
            candidates: Candidate passages from earlier retrieval stages.
            top_k: Number of highest-scoring passages to return.

        Returns:
            list[VectorSearchResult]: Reranked candidates with updated relevance scores.
        """

    async def arerank(
        self,
        query: str,
        candidates: list[VectorSearchResult],
        top_k: int = 4,
    ) -> list[VectorSearchResult]:
        """Asynchronously rerank candidates, offloading model inference to a worker thread.

        Args:
            query: Question or search query.
            candidates: Candidate passages.
            top_k: Number of top passages to return.

        Returns:
            list[VectorSearchResult]: Reranked passages.
        """
        return await asyncio.to_thread(self.rerank, query, candidates, top_k)


class CrossEncoderReranker(BaseReranker):
    """Neural cross-encoder reranker leveraging ms-marco-MiniLM-L-6-v2.

    Computes full cross-attention between (query, passage) token pairs, capturing
    fine-grained semantic and lexical interactions beyond bi-encoder vector dot products.

    Args:
        model_name: HuggingFace model identifier. Defaults to settings.RERANKER_MODEL_NAME.
        device: Device to place the model on ('cpu', 'cuda', etc.). Defaults to auto-select.
        max_length: Maximum sequence length for tokenization. Defaults to 512.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self.device = device
        self.max_length = max_length
        self._tokenizer: Any = None
        self._model: Any = None

    def _ensure_model_loaded(self) -> None:
        """Lazily load the tokenizer and sequence classification model on first inference."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading CrossEncoder model: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        if self.device:
            self._model.to(self.device)
        self._model.eval()

    def rerank(
        self,
        query: str,
        candidates: list[VectorSearchResult],
        top_k: int = 4,
    ) -> list[VectorSearchResult]:
        """Score (query, chunk.text) pairs and return top_k candidates sorted by cross-encoder score.

        Args:
            query: Search query.
            candidates: Candidate passages from earlier retrieval stages.
            top_k: Number of passages to return.

        Returns:
            list[VectorSearchResult]: Reranked passages with cross-encoder scores.
        """
        if not candidates:
            return []

        cleaned_query = query.strip()
        if not cleaned_query:
            return candidates[:top_k]

        # Single candidate optimization: no reordering needed
        if len(candidates) == 1:
            return candidates[:top_k]

        try:
            self._ensure_model_loaded()
            import torch

            pairs = [(cleaned_query, cand.text) for cand in candidates]
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            if self.device:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                if logits.shape[1] == 1:
                    raw_scores = logits.squeeze(-1).tolist()
                else:
                    raw_scores = logits[:, 1].tolist()

            scores = raw_scores if isinstance(raw_scores, list) else [raw_scores]

            # Rebuild results with cross-encoder relevance scores
            rescored: list[tuple[float, VectorSearchResult]] = []
            for score, cand in zip(scores, candidates):
                rescored_cand = VectorSearchResult(
                    chunk_id=cand.chunk_id,
                    text=cand.text,
                    metadata=cand.metadata,
                    score=round(float(score), 4),
                )
                rescored.append((float(score), rescored_cand))

            rescored.sort(key=lambda item: item[0], reverse=True)
            return [res for _, res in rescored[:top_k]]

        except Exception as exc:
            logger.warning(
                "CrossEncoder reranking failed (%s). Falling back to candidate order.",
                exc,
            )
            return candidates[:top_k]


class MockReranker(BaseReranker):
    """Deterministic mock reranker for unit testing without downloading neural weights."""

    def __init__(self, score_multiplier: float = 1.0) -> None:
        self.score_multiplier = score_multiplier

    def rerank(
        self,
        query: str,
        candidates: list[VectorSearchResult],
        top_k: int = 4,
    ) -> list[VectorSearchResult]:
        """Simulate reranking by sorting candidates based on exact query word overlap."""
        if not candidates:
            return []

        query_words = set(query.lower().split())
        scored: list[tuple[float, VectorSearchResult]] = []
        for idx, cand in enumerate(candidates):
            overlap = sum(1 for w in query_words if w in cand.text.lower())
            mock_score = round(float(overlap + (cand.score or 0.0)) * self.score_multiplier, 4)
            scored.append(
                (
                    mock_score,
                    VectorSearchResult(
                        chunk_id=cand.chunk_id,
                        text=cand.text,
                        metadata=cand.metadata,
                        score=mock_score,
                    ),
                )
            )

        scored.sort(key=lambda x: x[0], reverse=True)
        return [res for _, res in scored[:top_k]]
