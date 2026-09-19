"""Unit tests for Reciprocal Rank Fusion (RRF) algorithm."""

import pytest

from app.schemas.document import VectorChunkMetadata, VectorSearchResult
from app.services.hybrid_retriever import reciprocal_rank_fusion


def _make_result(chunk_id: str, text: str = "sample text") -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        text=text,
        metadata=VectorChunkMetadata(
            document_id="doc1",
            user_id="user1",
            filename="paper.pdf",
            page=1,
            chunk_index=0,
        ),
        score=1.0,
    )


class TestReciprocalRankFusion:
    """Tests for RRF score merging and ranking."""

    def test_both_lists_ranked_first_has_highest_score(self) -> None:
        dense = [_make_result("d1"), _make_result("d2"), _make_result("d3")]
        sparse = [_make_result("d1"), _make_result("d4"), _make_result("d2")]

        fused = reciprocal_rank_fusion(dense, sparse, k=60)

        assert len(fused) == 4
        # d1 was rank 1 in dense (1/61) and rank 1 in sparse (1/61) -> total 2/61 (~0.032787)
        assert fused[0].chunk_id == "d1"
        expected_d1_score = round(1.0 / 61.0 + 1.0 / 61.0, 6)
        assert fused[0].score == expected_d1_score

    def test_item_in_both_beats_single_appearance(self) -> None:
        dense = [_make_result("only_dense_1"), _make_result("both_rank2")]
        sparse = [_make_result("only_sparse_1"), _make_result("both_rank2")]

        fused = reciprocal_rank_fusion(dense, sparse, k=60)

        # both_rank2 has 1/(60+2) + 1/(60+2) = 2/62 = 0.032258
        # only_dense_1 has 1/(60+1) = 1/61 = 0.016393
        # only_sparse_1 has 1/(60+1) = 1/61 = 0.016393
        assert fused[0].chunk_id == "both_rank2"

    def test_disjoint_lists_are_combined(self) -> None:
        dense = [_make_result("d1"), _make_result("d2")]
        sparse = [_make_result("s1"), _make_result("s2")]

        fused = reciprocal_rank_fusion(dense, sparse, k=60)

        assert len(fused) == 4
        chunk_ids = [f.chunk_id for f in fused]
        assert set(chunk_ids) == {"d1", "d2", "s1", "s2"}

    def test_empty_lists_handled(self) -> None:
        assert reciprocal_rank_fusion([], []) == []

        dense = [_make_result("d1")]
        fused = reciprocal_rank_fusion(dense, [])
        assert len(fused) == 1
        assert fused[0].chunk_id == "d1"

    def test_custom_weights(self) -> None:
        dense = [_make_result("d1")]
        sparse = [_make_result("s1")]

        # Dense weighted 2.0x vs sparse 1.0x
        fused = reciprocal_rank_fusion(dense, sparse, k=60, dense_weight=2.0, sparse_weight=1.0)
        assert fused[0].chunk_id == "d1"
        assert fused[0].score > fused[1].score
