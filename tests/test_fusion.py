import pytest

from aml_reg_rag.retrieval.fusion import RRF_K, reciprocal_rank_fusion


def test_single_ranking_keeps_order_with_rrf_scores():
    fused = reciprocal_rank_fusion([["a", "b"]])
    assert [key for key, _ in fused] == ["a", "b"]
    assert fused[0][1] == pytest.approx(1 / (RRF_K + 1))


def test_items_in_several_rankings_accumulate():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["c", "b"]]))
    assert fused["b"] == pytest.approx(2 / (RRF_K + 2))
    assert fused["c"] == pytest.approx(1 / (RRF_K + 3) + 1 / (RRF_K + 1))
    assert max(fused, key=lambda key: fused[key]) == "c"


def test_ties_keep_first_seen_order():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    assert [key for key, _ in fused] == ["a", "b"]
