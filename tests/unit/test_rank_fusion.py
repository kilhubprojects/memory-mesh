"""Unit tests for reciprocal_rank_fusion."""

from __future__ import annotations

from memorymesh.search.rank_fusion import reciprocal_rank_fusion


class TestRRF:
    def test_single_list_preserves_order(self) -> None:
        result = reciprocal_rank_fusion([["a", "b", "c"]])
        ids = [doc_id for doc_id, _ in result]
        assert ids == ["a", "b", "c"]

    def test_document_in_both_lists_scores_higher(self) -> None:
        # "b" is rank-1 in list2 and rank-3 in list1 — should beat "a" (rank-1 in list1 only)
        result = reciprocal_rank_fusion([["a", "b", "c"], ["b", "x", "y"]])
        scores = dict(result)
        assert scores["b"] > scores["a"]

    def test_scores_positive(self) -> None:
        result = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
        for _, score in result:
            assert score > 0.0

    def test_all_ids_present_in_result(self) -> None:
        result = reciprocal_rank_fusion([["a", "b"], ["c", "d"]])
        ids = {doc_id for doc_id, _ in result}
        assert ids == {"a", "b", "c", "d"}

    def test_empty_lists_returns_empty(self) -> None:
        assert reciprocal_rank_fusion([]) == []

    def test_empty_inner_lists_returns_empty(self) -> None:
        assert reciprocal_rank_fusion([[], []]) == []

    def test_k_parameter_affects_scores(self) -> None:
        # Lower k → higher scores (top rank matters more)
        r_low = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
        r_high = dict(reciprocal_rank_fusion([["a", "b"]], k=100))
        assert r_low["a"] > r_high["a"]

    def test_result_sorted_descending(self) -> None:
        result = reciprocal_rank_fusion([["a", "b", "c"], ["c", "b", "a"]])
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_known_scores(self) -> None:
        # Single list, k=60: rank 1 → 1/61, rank 2 → 1/62
        result = dict(reciprocal_rank_fusion([["x", "y"]], k=60))
        assert abs(result["x"] - 1 / 61) < 1e-12
        assert abs(result["y"] - 1 / 62) < 1e-12

    def test_duplicate_within_list_uses_first_occurrence(self) -> None:
        # "a" appears twice — second occurrence at rank 2 adds extra score
        result = dict(reciprocal_rank_fusion([["a", "a"]]))
        # Score = 1/(60+1) + 1/(60+2)
        expected = 1 / 61 + 1 / 62
        assert abs(result["a"] - expected) < 1e-12
