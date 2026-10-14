import itertools

import pytest

from aml_reg_rag.models import ChunkRef
from aml_reg_rag.retrieval.expand import ReferenceExpander
from aml_reg_rag.retrieval.pipeline import RetrievalOptions, RetrievalPipeline
from aml_reg_rag.retrieval.rerank import Rerank
from aml_reg_rag.retrieval.rewrite import Rewrite
from tests.fakes import FakeIndex, retrieved

A = retrieved("G0380252:3:p1:k1-4", "金融機構", law_name="金融機構防制洗錢辦法", score=0.9)
B = retrieved("G0380268:3:p1:k1-4", "虛擬資產服務商", score=0.8)
C = retrieved("I0030047:12", "罰鍰", law_name="資恐防制法", score=0.7)
D = retrieved("G0380131:5", "金融機構定義", law_name="洗錢防制法", score=0.6)


class FakeBm25:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def search(self, query, n):
        self.queries.append(query)
        return self.results.get(query, [])[:n]


class FakeRewriter:
    def __init__(self, queries, fallback=None):
        self.result = Rewrite(queries, fallback, 10, 2)

    def rewrite(self, question):
        return self.result


class FakeReranker:
    def __init__(self, scores, fallback=None):
        self.scores = scores
        self.fallback = fallback
        self.seen = None
        self.question = None

    def score(self, question, candidates):
        self.question = question
        self.seen = [chunk.chunk_id for chunk in candidates]
        return Rerank(self.scores, self.fallback, 30, 5)


def options(hybrid=False, rewrite=False, rerank=False, candidates=20, expand_refs=False):
    return RetrievalOptions(
        hybrid=hybrid,
        rewrite=rewrite,
        rerank=rerank,
        candidates=candidates,
        expand_refs=expand_refs,
        expand_limit=3,
    )


def ids(retrieval):
    return [chunk.chunk_id for chunk in retrieval.chunks]


def test_vector_only_matches_v1_order_and_uses_rrf_scores():
    vector = FakeIndex([A, B, C])
    result = RetrievalPipeline(vector, options()).retrieve("問題", 2)
    assert ids(result) == [A.chunk_id, B.chunk_id]
    assert vector.queries == [("問題", 20)]
    assert result.queries == ["問題"]
    assert result.chunks[0].score == pytest.approx(1 / 61)
    trace = result.traces[0]
    assert (trace.vector_rank, trace.vector_score, trace.bm25_rank, trace.final_rank) == (
        1,
        0.9,
        None,
        1,
    )
    assert result.traces[2].final_rank is None


def test_hybrid_fuses_bm25_into_ranking():
    bm25 = FakeBm25({"問題": [(C, 12.5), (B, 3.0)]})
    result = RetrievalPipeline(FakeIndex([A, B]), options(hybrid=True), bm25=bm25).retrieve(
        "問題", 3
    )
    # B 在兩個排名都出現；A 與 C 同分時，先出現在向量排名的 A 在前
    assert ids(result) == [B.chunk_id, A.chunk_id, C.chunk_id]
    trace = {t.chunk_id: t for t in result.traces}[C.chunk_id]
    assert (trace.vector_rank, trace.bm25_rank, trace.bm25_score) == (None, 1, 12.5)


def test_rewrite_adds_candidates_from_subqueries():
    class ByQuery(FakeIndex):
        def search(self, query, k):
            self.queries.append((query, k))
            return {"原問題": [A, B], "子查詢": [C, D]}[query]

    vector = ByQuery([])
    rewriter = FakeRewriter(["原問題", "子查詢"])
    result = RetrievalPipeline(vector, options(rewrite=True), rewriter=rewriter).retrieve(
        "原問題", 4
    )
    assert result.queries == ["原問題", "子查詢"]
    assert [query for query, _ in vector.queries] == ["原問題", "子查詢"]
    assert set(ids(result)) == {A.chunk_id, B.chunk_id, C.chunk_id, D.chunk_id}
    assert (result.input_tokens, result.output_tokens) == (10, 2)


class ByQuery(FakeIndex):
    def search(self, query, k):
        self.queries.append((query, k))
        return {"原問題": [A, B, C], "子查詢": [C, D]}[query]


def test_rewrite_hybrid_keeps_best_vector_rank_across_queries():
    bm25 = FakeBm25({"原問題": [(D, 4.0)], "子查詢": [(A, 2.0)]})
    result = RetrievalPipeline(
        ByQuery([]),
        options(hybrid=True, rewrite=True),
        bm25=bm25,
        rewriter=FakeRewriter(["原問題", "子查詢"]),
    ).retrieve("原問題", 4)
    trace = {t.chunk_id: t for t in result.traces}[C.chunk_id]
    assert trace.vector_rank == 1
    assert bm25.queries == ["原問題", "子查詢"]


def test_reranker_receives_original_question():
    reranker = FakeReranker({A.chunk_id: 1, B.chunk_id: 1, C.chunk_id: 1, D.chunk_id: 1})
    RetrievalPipeline(
        ByQuery([]),
        options(rewrite=True, rerank=True),
        rewriter=FakeRewriter(["原問題", "子查詢"]),
        reranker=reranker,
    ).retrieve("原問題", 2)
    assert reranker.question == "原問題"


def test_rerank_reorders_and_ties_keep_rrf_order():
    reranker = FakeReranker({A.chunk_id: 0, B.chunk_id: 3, C.chunk_id: 3})
    result = RetrievalPipeline(FakeIndex([A, B, C]), options(rerank=True), reranker=reranker)
    retrieval = result.retrieve("問題", 2)
    assert reranker.seen == [A.chunk_id, B.chunk_id, C.chunk_id]
    assert ids(retrieval) == [B.chunk_id, C.chunk_id]
    assert [chunk.score for chunk in retrieval.chunks] == [3.0, 3.0]
    assert {t.chunk_id: t.rerank_score for t in retrieval.traces}[A.chunk_id] == 0


def test_rerank_fallback_keeps_rrf_order():
    reranker = FakeReranker(None, fallback="TimeoutError")
    retrieval = RetrievalPipeline(
        FakeIndex([A, B]), options(rerank=True), reranker=reranker
    ).retrieve("問題", 2)
    assert ids(retrieval) == [A.chunk_id, B.chunk_id]
    assert retrieval.rerank_fallback == "TimeoutError"
    assert retrieval.traces[0].rerank_score is None


def test_rewrite_fallback_is_recorded():
    rewriter = FakeRewriter(["問題"], fallback="輸出無法解析")
    retrieval = RetrievalPipeline(
        FakeIndex([A]), options(rewrite=True), rewriter=rewriter
    ).retrieve("問題", 1)
    assert retrieval.rewrite_fallback == "輸出無法解析"


def test_candidates_cover_k_and_limit_rerank_input():
    reranker = FakeReranker({A.chunk_id: 1, B.chunk_id: 1})
    vector = FakeIndex([A, B, C, D])
    pipeline = RetrievalPipeline(vector, options(rerank=True, candidates=2), reranker=reranker)
    pipeline.retrieve("問題", 1)
    assert vector.queries == [("問題", 2)]
    assert reranker.seen == [A.chunk_id, B.chunk_id]
    RetrievalPipeline(vector, options(candidates=2)).retrieve("問題", 3)
    assert vector.queries[-1] == ("問題", 3)


@pytest.mark.parametrize(
    ("hybrid", "rewrite", "rerank"), list(itertools.product([False, True], repeat=3))
)
def test_every_combination_returns_k_chunks_with_traces(hybrid, rewrite, rerank):
    pipeline = RetrievalPipeline(
        FakeIndex([A, B, C]),
        options(hybrid, rewrite, rerank),
        bm25=FakeBm25({"問題": [(D, 5.0)]}) if hybrid else None,
        rewriter=FakeRewriter(["問題"]) if rewrite else None,
        reranker=FakeReranker({A.chunk_id: 1, B.chunk_id: 2, C.chunk_id: 0, D.chunk_id: 3})
        if rerank
        else None,
    )
    retrieval = pipeline.retrieve("問題", 2)
    assert len(retrieval.chunks) == 2
    assert sorted(t.final_rank for t in retrieval.traces if t.final_rank) == [1, 2]
    expected_tokens = (10 if rewrite else 0) + (30 if rerank else 0)
    assert retrieval.input_tokens == expected_tokens
    assert retrieval.output_tokens == (2 if rewrite else 0) + (5 if rerank else 0)


def test_missing_component_is_rejected():
    with pytest.raises(ValueError, match="hybrid、rerank"):
        RetrievalPipeline(FakeIndex([A]), options(hybrid=True, rerank=True))
    with pytest.raises(ValueError, match="expand_refs"):
        RetrievalPipeline(FakeIndex([A]), options(expand_refs=True))


def test_expand_refs_appends_cited_chunks_after_top_k():
    refs = [
        ChunkRef(src_chunk_id=B.chunk_id, target_chunk_id=D.chunk_id, kind="definition", raw="")
    ]
    expander = ReferenceExpander(refs, {D.chunk_id: D}, limit=3)
    pipeline = RetrievalPipeline(FakeIndex([A, B, C]), options(expand_refs=True), expander=expander)
    retrieval = pipeline.retrieve("問題", 2)
    assert ids(retrieval) == [A.chunk_id, B.chunk_id, D.chunk_id]
    assert retrieval.chunks[2].expanded_from == B.chunk_id
    assert all(trace.chunk_id != D.chunk_id for trace in retrieval.traces)


def test_options_from_settings_and_overrides(settings):
    defaults = RetrievalOptions.from_settings(settings)
    assert (defaults.hybrid, defaults.rewrite, defaults.rerank) == (True, False, True)
    assert (defaults.expand_refs, defaults.expand_limit) == (False, 3)
    overridden = RetrievalOptions.from_settings(
        settings, hybrid=False, rewrite=False, rerank=False, expand_refs=True
    )
    assert (overridden.hybrid, overridden.rewrite, overridden.rerank) == (False, False, False)
    assert overridden.expand_refs is True
    assert overridden.candidates == settings.retrieval_candidates
