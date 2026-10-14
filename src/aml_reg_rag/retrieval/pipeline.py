from dataclasses import dataclass, field
from typing import Protocol

from aml_reg_rag.config import Settings
from aml_reg_rag.models import ChunkTrace, Retrieval, RetrievedChunk
from aml_reg_rag.retrieval.bm25 import Bm25Index
from aml_reg_rag.retrieval.expand import ReferenceExpander
from aml_reg_rag.retrieval.fusion import reciprocal_rank_fusion
from aml_reg_rag.retrieval.index import ChunkIndex, Database, load_chunks, load_refs
from aml_reg_rag.retrieval.rerank import LlmReranker, make_reranker
from aml_reg_rag.retrieval.rewrite import QueryRewriter, make_rewriter


class Retriever(Protocol):
    def retrieve(self, question: str, k: int) -> Retrieval: ...


@dataclass(frozen=True)
class RetrievalOptions:
    hybrid: bool
    rewrite: bool
    rerank: bool
    candidates: int
    expand_refs: bool
    expand_limit: int

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        hybrid: bool | None = None,
        rewrite: bool | None = None,
        rerank: bool | None = None,
        expand_refs: bool | None = None,
    ) -> "RetrievalOptions":
        return cls(
            hybrid=settings.retrieval_hybrid if hybrid is None else hybrid,
            rewrite=settings.retrieval_rewrite if rewrite is None else rewrite,
            rerank=settings.retrieval_rerank if rerank is None else rerank,
            candidates=settings.retrieval_candidates,
            expand_refs=settings.retrieval_expand_refs if expand_refs is None else expand_refs,
            expand_limit=settings.retrieval_expand_limit,
        )


@dataclass
class _Stage:
    vector_rank: int | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None


@dataclass
class _Pool:
    chunks: dict[str, RetrievedChunk] = field(default_factory=dict)
    stages: dict[str, _Stage] = field(default_factory=dict)

    def add(self, chunk: RetrievedChunk) -> _Stage:
        self.chunks.setdefault(chunk.chunk_id, chunk)
        return self.stages.setdefault(chunk.chunk_id, _Stage())


class RetrievalPipeline:
    def __init__(
        self,
        vector: ChunkIndex,
        options: RetrievalOptions,
        *,
        bm25: Bm25Index | None = None,
        rewriter: QueryRewriter | None = None,
        reranker: LlmReranker | None = None,
        expander: ReferenceExpander | None = None,
    ) -> None:
        missing = [
            name
            for name, enabled, part in (
                ("hybrid", options.hybrid, bm25),
                ("rewrite", options.rewrite, rewriter),
                ("rerank", options.rerank, reranker),
                ("expand_refs", options.expand_refs, expander),
            )
            if enabled and part is None
        ]
        if missing:
            raise ValueError(f"檢索管線缺少元件：{'、'.join(missing)}")
        self._vector = vector
        self._options = options
        self._bm25 = bm25
        self._rewriter = rewriter
        self._reranker = reranker
        self._expander = expander

    def retrieve(self, question: str, k: int) -> Retrieval:
        size = max(self._options.candidates, k)
        input_tokens = output_tokens = 0
        queries, rewrite_fallback = [question], None
        if self._rewriter is not None and self._options.rewrite:
            rewrite = self._rewriter.rewrite(question)
            queries, rewrite_fallback = rewrite.queries, rewrite.fallback
            input_tokens += rewrite.input_tokens
            output_tokens += rewrite.output_tokens

        pool = _Pool()
        per_query = [self._rank_query(query, size, pool) for query in queries]
        fused = reciprocal_rank_fusion(per_query)[:size]
        candidates = [pool.chunks[chunk_id] for chunk_id, _ in fused]
        rrf_scores = dict(fused)

        rerank_scores, rerank_fallback = None, None
        if self._reranker is not None and self._options.rerank and candidates:
            rerank = self._reranker.score(question, candidates)
            rerank_scores, rerank_fallback = rerank.scores, rerank.fallback
            input_tokens += rerank.input_tokens
            output_tokens += rerank.output_tokens
        if rerank_scores is not None:
            # sorted 為穩定排序，同分時保留 RRF 名次
            candidates = sorted(candidates, key=lambda chunk: -rerank_scores[chunk.chunk_id])

        final = candidates[:k]
        final_ranks = {chunk.chunk_id: rank for rank, chunk in enumerate(final, start=1)}
        chunks = [
            chunk.model_copy(
                update={
                    "score": float(rerank_scores[chunk.chunk_id])
                    if rerank_scores is not None
                    else rrf_scores[chunk.chunk_id]
                }
            )
            for chunk in final
        ]
        if self._expander is not None and self._options.expand_refs:
            # 補入的段落排在檢索結果之後，不參與排名，也不寫入 traces
            chunks.extend(self._expander.expand(chunks))
            chunks = self._expander.annotate(chunks)
        traces = [
            ChunkTrace(
                chunk_id=chunk.chunk_id,
                **vars(pool.stages[chunk.chunk_id]),
                rrf_score=rrf_scores[chunk.chunk_id],
                rerank_score=None if rerank_scores is None else rerank_scores[chunk.chunk_id],
                final_rank=final_ranks.get(chunk.chunk_id),
            )
            for chunk in candidates
        ]
        return Retrieval(
            chunks=chunks,
            queries=queries,
            traces=traces,
            rewrite_fallback=rewrite_fallback,
            rerank_fallback=rerank_fallback,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _rank_query(self, query: str, size: int, pool: _Pool) -> list[str]:
        rankings = []
        vector_ids = []
        for rank, chunk in enumerate(self._vector.search(query, size), start=1):
            stage = pool.add(chunk)
            if stage.vector_rank is None or rank < stage.vector_rank:
                stage.vector_rank, stage.vector_score = rank, chunk.score
            vector_ids.append(chunk.chunk_id)
        rankings.append(vector_ids)
        if self._bm25 is not None and self._options.hybrid:
            bm25_ids = []
            for rank, (chunk, score) in enumerate(self._bm25.search(query, size), start=1):
                stage = pool.add(chunk)
                if stage.bm25_rank is None or rank < stage.bm25_rank:
                    stage.bm25_rank, stage.bm25_score = rank, score
                bm25_ids.append(chunk.chunk_id)
            rankings.append(bm25_ids)
        return [chunk_id for chunk_id, _ in reciprocal_rank_fusion(rankings)]


def make_pipeline(
    db: Database, table: str, vector: ChunkIndex, settings: Settings, options: RetrievalOptions
) -> RetrievalPipeline:
    entries = load_chunks(db, table) if options.hybrid or options.expand_refs else []
    expander = None
    if options.expand_refs:
        chunks = {chunk.chunk_id: chunk for chunk, _ in entries}
        expander = ReferenceExpander(load_refs(db, table), chunks, options.expand_limit)
    return RetrievalPipeline(
        vector,
        options,
        bm25=Bm25Index(entries) if options.hybrid else None,
        rewriter=make_rewriter(settings.llm_light_model, settings.llm_timeout)
        if options.rewrite
        else None,
        reranker=make_reranker(settings.llm_light_model, settings.llm_timeout)
        if options.rerank
        else None,
        expander=expander,
    )
