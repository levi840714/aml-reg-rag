import re
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.runnables import Runnable
from pydantic import BaseModel

from aml_reg_rag.evaluation.dataset import CuratedQuestion, SyntheticQuestion
from aml_reg_rag.evaluation.judge import judge_answer
from aml_reg_rag.evaluation.metrics import (
    AnswerResult,
    GenerationSummary,
    RetrievalResult,
    RetrievalSummary,
    UnstableQuestion,
    article_ranking,
    consistency,
    group_by,
    refusal_outcome,
    summarize_generation,
    summarize_retrieval,
)
from aml_reg_rag.graph.state import AskState, labeled_chunks
from aml_reg_rag.llm import describe_error
from aml_reg_rag.models import Answer, RetrievedChunk
from aml_reg_rag.retrieval.index import IndexMeta
from aml_reg_rag.retrieval.pipeline import Retriever

AskFn = Callable[[str, int], AskState]
Progress = Callable[[str], None]


class EvalReport(BaseModel):
    created_at: datetime
    model: str | None
    judge_model: str | None
    k: int
    repeat: int
    index: IndexMeta
    settings: dict[str, str | int | bool]
    reviewed: int
    unreviewed: int
    retrieval: dict[str, RetrievalSummary]
    generation: dict[str, GenerationSummary]
    consistency: float | None
    unstable: list[UnstableQuestion]
    retrieval_results: list[RetrievalResult]
    answer_results: list[AnswerResult]


def run_retrieval(
    retriever: Retriever, curated: list[CuratedQuestion], synthetic: list[SyntheticQuestion], k: int
) -> list[RetrievalResult]:
    items = [
        *(
            (q.id, "curated", q.category, q.expected_articles, q.question)
            for q in curated
            if q.category != "refusal"
        ),
        *((q.id, "synthetic", "synthetic", q.expected_articles, q.question) for q in synthetic),
    ]
    results = []
    for qid, dataset, category, expected, question in items:
        started = time.perf_counter()
        try:
            retrieval = retriever.retrieve(question, k)
        except Exception as exc:  # 單題失敗不中斷整批評測
            results.append(
                RetrievalResult(
                    question_id=qid,
                    dataset=dataset,
                    category=category,
                    expected=expected,
                    ranked=[],
                    top_score=None,
                    error=describe_error(exc),
                )
            )
            continue
        chunks = retrieval.chunks
        results.append(
            RetrievalResult(
                question_id=qid,
                dataset=dataset,
                category=category,
                expected=expected,
                ranked=article_ranking(chunks),
                top_score=chunks[0].score if chunks else None,
                queries=len(retrieval.queries),
                rewrite_fallback=retrieval.rewrite_fallback is not None,
                rerank_fallback=retrieval.rerank_fallback is not None,
                input_tokens=retrieval.input_tokens,
                output_tokens=retrieval.output_tokens,
                seconds=time.perf_counter() - started,
            )
        )
    return results


def run_generation(
    ask: AskFn,
    judge: Runnable[Any, Any],
    curated: list[CuratedQuestion],
    k: int,
    repeat: int,
    progress: Progress | None = None,
) -> list[AnswerResult]:
    results: list[AnswerResult] = []
    total = len(curated) * repeat
    for run in range(1, repeat + 1):
        for question in curated:
            if progress:
                progress(f"[{len(results) + 1}/{total}] {question.id}")
            results.append(answer_one(ask, judge, question, k, run))
    return results


def answer_one(
    ask: AskFn, judge: Runnable[Any, Any], question: CuratedQuestion, k: int, run: int
) -> AnswerResult:
    started = time.perf_counter()
    result = AnswerResult(question_id=question.id, category=question.category, run=run)
    try:
        state = ask(question.question, k)
    except Exception as exc:  # 記錄後繼續下一題
        result.error = describe_error(exc)
        result.seconds = time.perf_counter() - started
        return result
    result.seconds = time.perf_counter() - started
    result.input_tokens = state["input_tokens"]
    result.output_tokens = state["output_tokens"]
    if state["retrieval"] is not None:
        result.retrieval_input_tokens = state["retrieval"].input_tokens
        result.retrieval_output_tokens = state["retrieval"].output_tokens
    answer = state["answer"]
    if answer is None:
        # 模型有回應但格式錯誤，屬於生成品質問題，計入驗證失敗而非執行錯誤
        result.unparsed = True
        return result
    result.verified = state["verified"]
    cited = cited_articles(answer, labeled_chunks(state["chunks"]))
    result.predicted_answerable = answer.answerable
    result.refusal = refusal_outcome(question.answerable, answer.answerable)
    result.cited_articles = cited
    result.cites_forbidden = any(article in question.forbidden_articles for article in cited)
    if question.answerable and answer.answerable:
        result.cites_expected = any(article in question.expected_articles for article in cited)
        _judge(judge, question, answer, result)
    return result


def cited_articles(answer: Answer, labeled: dict[str, RetrievedChunk]) -> list[str]:
    keys = [
        labeled[citation.source].article_key
        for point in answer.points
        for citation in point.citations
        if citation.source in labeled
    ]
    return list(dict.fromkeys(keys))


def build_report(
    *,
    created_at: datetime,
    model: str | None,
    judge_model: str | None,
    k: int,
    repeat: int,
    index: IndexMeta,
    settings: dict[str, str | int | bool],
    light_model: str,
    curated: list[CuratedQuestion],
    retrieval_results: list[RetrievalResult],
    answer_results: list[AnswerResult],
) -> EvalReport:
    retrieval: dict[str, RetrievalSummary] = {}
    for dataset, items in group_by(retrieval_results, lambda r: r.dataset).items():
        retrieval[dataset] = summarize_retrieval(items, light_model)
        if dataset == "curated":
            for category, group in group_by(items, lambda r: r.category).items():
                retrieval[f"curated/{category}"] = summarize_retrieval(group, light_model)
    generation: dict[str, GenerationSummary] = {}
    if answer_results and model and judge_model:
        generation["all"] = summarize_generation(answer_results, model, judge_model, light_model)
        for category, answers in group_by(answer_results, lambda r: r.category).items():
            generation[category] = summarize_generation(answers, model, judge_model, light_model)
    agreement, unstable = consistency(answer_results)
    reviewed = sum(question.reviewed for question in curated)
    return EvalReport(
        created_at=created_at,
        model=model,
        judge_model=judge_model,
        k=k,
        repeat=repeat,
        index=index,
        settings=settings,
        reviewed=reviewed,
        unreviewed=len(curated) - reviewed,
        retrieval=retrieval,
        generation=generation,
        consistency=agreement,
        unstable=unstable,
        retrieval_results=retrieval_results,
        answer_results=answer_results,
    )


def retrieval_label(settings: dict[str, str | int | bool]) -> str:
    # v1 的結果檔沒有檢索開關，視為只有向量檢索
    flags = [name for name in ("hybrid", "rewrite", "rerank") if settings.get(f"retrieval_{name}")]
    return "+".join(flags) or "vector"


def write_report(eval_dir: Path, report: EvalReport) -> Path:
    eval_dir.mkdir(parents=True, exist_ok=True)
    model = re.sub(r"[^A-Za-z0-9.-]+", "_", report.model or "retrieval")
    stem = f"{report.created_at:%Y%m%d-%H%M%S}-{retrieval_label(report.settings)}-{model}"
    path = eval_dir / f"{stem}.json"
    # ablation 連續執行時可能在同一秒產生結果，加序號避免覆寫
    number = 2
    while path.exists():
        path = eval_dir / f"{stem}-{number}.json"
        number += 1
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def _judge(
    judge: Runnable[Any, Any], question: CuratedQuestion, answer: Answer, result: AnswerResult
) -> None:
    try:
        verdict = judge_answer(judge, question.question, question.key_points, answer)
    except Exception:  # judge 失敗不影響生成結果
        result.judge_failed = True
        return
    result.judge_input_tokens = verdict.input_tokens
    result.judge_output_tokens = verdict.output_tokens
    result.coverage = verdict.coverage
    result.judge_failed = verdict.verdicts is None
