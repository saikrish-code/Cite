"""Hybrid retrieval pipeline combining dense vector search, BM25, RRF, and Cross-Encoder reranking.

Provides Reciprocal Rank Fusion (RRF) rank aggregation, per-stage latency tracking,
strict multi-tenant isolation, and toggleable retrieval modes ("vector_only", "hybrid", "hybrid_rerank").
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

from app.core.config import settings
from app.schemas.document import VectorSearchResult
from app.services.bm25 import BM25SearchService
from app.services.reranker import BaseReranker, CrossEncoderReranker
from app.services.vector_store import BaseVectorStore, ChromaVectorStore

logger = logging.getLogger(__name__)

RetrievalMode = Literal["vector_only", "hybrid", "hybrid_rerank"]


@dataclass
class RetrievalLatencyMetrics:
    """Latency profiling measurements for each retrieval stage in milliseconds."""

    vector_ms: float = 0.0
    bm25_ms: float = 0.0
    rrf_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0
    mode: str = "hybrid_rerank"

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a JSON-serializable dictionary."""
        return asdict(self)


def reciprocal_rank_fusion(
    dense_results: list[VectorSearchResult],
    sparse_results: list[VectorSearchResult],
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[VectorSearchResult]:
    """Merge and rank passages from dense and sparse retrieval using Reciprocal Rank Fusion (RRF).

    Formula for passage d:
        RRF_score(d) = sum_{system m} (weight_m / (k + rank_m(d)))
    where rank_m(d) is 1-indexed.

    Args:
        dense_results: Ordered results from dense vector search.
        sparse_results: Ordered results from sparse BM25 search.
        k: Smoothing constant to control impact of high ranks (standard is 60).
        dense_weight: Multiplier weight for dense vector rankings.
        sparse_weight: Multiplier weight for BM25 lexical rankings.

    Returns:
        list[VectorSearchResult]: Fused and deduplicated results ordered by descending RRF score.
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, VectorSearchResult] = {}

    # Accumulate dense ranks
    for rank, item in enumerate(dense_results, start=1):
        chunk_id = item.chunk_id
        doc_map[chunk_id] = item
        contribution = dense_weight / (k + rank)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + contribution

    # Accumulate sparse BM25 ranks
    for rank, item in enumerate(sparse_results, start=1):
        chunk_id = item.chunk_id
        if chunk_id not in doc_map:
            doc_map[chunk_id] = item
        contribution = sparse_weight / (k + rank)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + contribution

    # Sort descending by cumulative RRF score
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    fused_results: list[VectorSearchResult] = []
    for chunk_id, score in sorted_items:
        original = doc_map[chunk_id]
        fused_results.append(
            VectorSearchResult(
                chunk_id=original.chunk_id,
                text=original.text,
                metadata=original.metadata,
                score=round(score, 6),
            )
        )

    return fused_results


class HybridRetriever:
    """Multi-stage hybrid retrieval coordinator.

    Supports:
    - "vector_only": Pure semantic search (baseline)
    - "hybrid": Dense semantic + BM25 keyword search merged via RRF
    - "hybrid_rerank": Hybrid RRF followed by Cross-Encoder neural reranking

    Args:
        vector_store: Vector store instance. Defaults to ChromaVectorStore.
        bm25_service: BM25 lexical search service. Defaults to BM25SearchService.
        reranker: Neural reranker instance. Defaults to CrossEncoderReranker.
        default_mode: Default retrieval strategy. Defaults to settings.RETRIEVAL_MODE.
        vector_top_k: Number of dense vectors to retrieve for fusion. Defaults to settings.RETRIEVAL_VECTOR_TOP_K.
        bm25_top_k: Number of BM25 passages to retrieve for fusion. Defaults to settings.RETRIEVAL_BM25_TOP_K.
        rrf_k: Reciprocal Rank Fusion constant. Defaults to settings.RRF_K.
        reranker_top_k: Final number of reranked passages. Defaults to settings.RERANKER_TOP_K.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore | None = None,
        bm25_service: BM25SearchService | None = None,
        reranker: BaseReranker | None = None,
        default_mode: str | None = None,
        vector_top_k: int | None = None,
        bm25_top_k: int | None = None,
        rrf_k: int | None = None,
        reranker_top_k: int | None = None,
    ) -> None:
        self.vector_store = vector_store or ChromaVectorStore()
        self.bm25_service = bm25_service or BM25SearchService()
        self.reranker = reranker or CrossEncoderReranker()
        self.default_mode = default_mode or settings.RETRIEVAL_MODE
        self.vector_top_k = vector_top_k or settings.RETRIEVAL_VECTOR_TOP_K
        self.bm25_top_k = bm25_top_k or settings.RETRIEVAL_BM25_TOP_K
        self.rrf_k = rrf_k or settings.RRF_K
        self.reranker_top_k = reranker_top_k or settings.RERANKER_TOP_K

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
        mode: str | None = None,
    ) -> tuple[list[VectorSearchResult], RetrievalLatencyMetrics]:
        """Synchronously retrieve passages using the requested hybrid strategy.

        Args:
            query: Question or search query.
            k: Final number of passages to return. Defaults to self.reranker_top_k.
            filters: Metadata filters strictly isolating user_id and document_id.
            mode: Retrieval mode override ("vector_only", "hybrid", "hybrid_rerank").

        Returns:
            tuple[list[VectorSearchResult], RetrievalLatencyMetrics]:
                - Ranked list of passages.
                - Measured latencies per pipeline stage.
        """
        active_mode = (mode or self.default_mode).lower().strip()
        final_k = k or self.reranker_top_k
        metrics = RetrievalLatencyMetrics(mode=active_mode)
        t_total_start = time.perf_counter()

        cleaned_query = query.strip()
        if not cleaned_query:
            metrics.total_ms = round((time.perf_counter() - t_total_start) * 1000, 2)
            return [], metrics

        # ---------------------------------------------------------------------
        # 1. Mode: "vector_only"
        # ---------------------------------------------------------------------
        if active_mode == "vector_only":
            t_vec_start = time.perf_counter()
            results = self.vector_store.similarity_search(
                query=cleaned_query,
                k=final_k,
                filters=filters,
            )
            metrics.vector_ms = round((time.perf_counter() - t_vec_start) * 1000, 2)
            metrics.total_ms = round((time.perf_counter() - t_total_start) * 1000, 2)

            logger.info(
                "Retrieval [vector_only]: vector=%.1fms (%d) | total=%.1fms",
                metrics.vector_ms,
                len(results),
                metrics.total_ms,
            )
            return results, metrics

        # ---------------------------------------------------------------------
        # 2. Dense Vector Retrieval (for hybrid & hybrid_rerank)
        # ---------------------------------------------------------------------
        t_vec_start = time.perf_counter()
        dense_results = self.vector_store.similarity_search(
            query=cleaned_query,
            k=self.vector_top_k,
            filters=filters,
        )
        metrics.vector_ms = round((time.perf_counter() - t_vec_start) * 1000, 2)

        # ---------------------------------------------------------------------
        # 3. Sparse BM25 Keyword Search
        # ---------------------------------------------------------------------
        t_bm25_start = time.perf_counter()
        candidate_chunks: list[VectorSearchResult] = []
        if hasattr(self.vector_store, "get_chunks"):
            try:
                retrieved_chunks = self.vector_store.get_chunks(filters=filters)
                if isinstance(retrieved_chunks, list):
                    candidate_chunks = [
                        c for c in retrieved_chunks if isinstance(c, VectorSearchResult)
                    ]
            except Exception as exc:
                logger.debug("Failed to retrieve candidate chunks for BM25: %s", exc)

        sparse_results = self.bm25_service.search(
            query=cleaned_query,
            chunks=candidate_chunks,
            top_k=self.bm25_top_k,
        )
        metrics.bm25_ms = round((time.perf_counter() - t_bm25_start) * 1000, 2)

        # ---------------------------------------------------------------------
        # 4. Reciprocal Rank Fusion (RRF)
        # ---------------------------------------------------------------------
        t_rrf_start = time.perf_counter()
        fused_results = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=self.rrf_k,
        )
        metrics.rrf_ms = round((time.perf_counter() - t_rrf_start) * 1000, 2)

        if not fused_results:
            metrics.total_ms = round((time.perf_counter() - t_total_start) * 1000, 2)
            return [], metrics

        # If mode is "hybrid" (without reranker), return top_k directly
        if active_mode == "hybrid":
            final_results = fused_results[:final_k]
            metrics.total_ms = round((time.perf_counter() - t_total_start) * 1000, 2)

            logger.info(
                "Retrieval [hybrid]: vector=%.1fms (%d), bm25=%.1fms (%d), rrf=%.1fms (%d) | total=%.1fms",
                metrics.vector_ms,
                len(dense_results),
                metrics.bm25_ms,
                len(sparse_results),
                metrics.rrf_ms,
                len(final_results),
                metrics.total_ms,
            )
            return final_results, metrics

        # ---------------------------------------------------------------------
        # 5. Cross-Encoder Neural Reranking (mode: "hybrid_rerank")
        # ---------------------------------------------------------------------
        t_rerank_start = time.perf_counter()
        # Feed top candidates into the cross-encoder
        rerank_pool = fused_results[: self.vector_top_k]
        final_results = self.reranker.rerank(
            query=cleaned_query,
            candidates=rerank_pool,
            top_k=final_k,
        )
        metrics.rerank_ms = round((time.perf_counter() - t_rerank_start) * 1000, 2)
        metrics.total_ms = round((time.perf_counter() - t_total_start) * 1000, 2)

        logger.info(
            "Retrieval [hybrid_rerank]: vector=%.1fms (%d), bm25=%.1fms (%d), rrf=%.1fms (%d), rerank=%.1fms (%d) | total=%.1fms",
            metrics.vector_ms,
            len(dense_results),
            metrics.bm25_ms,
            len(sparse_results),
            metrics.rrf_ms,
            len(fused_results),
            metrics.rerank_ms,
            len(final_results),
            metrics.total_ms,
        )
        return final_results, metrics

    async def aretrieve(
        self,
        query: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
        mode: str | None = None,
    ) -> tuple[list[VectorSearchResult], RetrievalLatencyMetrics]:
        """Asynchronously retrieve passages using worker thread offloading."""
        return await asyncio.to_thread(self.retrieve, query, k, filters, mode)
