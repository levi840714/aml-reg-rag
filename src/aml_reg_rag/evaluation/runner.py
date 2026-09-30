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
from aml_reg_rag.models import Answer, RetrievedChunk
from aml_reg_rag.retrieval.index import ChunkIndex, IndexMeta

AskFn = Callable[[str, int], AskState]
Progress = Callable[[str], None]


class EvalReport(BaseModel):
    created_at: datetime
    model: str | None
    judge_model: str | None
    k: int
    repeat: int
    index: IndexMeta
    reviewed: int
    unreviewed: int
    retrieval: dict[str, RetrievalSummary]
    generation: dict[str, GenerationSummary]
    consistency: float | None
    unstable: list[UnstableQuestion]
    retrieval_results: list[RetrievalResult]
    answer_results: list[AnswerResult]


def run_retrieval(
    index: ChunkIndex, curated: list[CuratedQuestion], synthetic: list[SyntheticQuestion], k: int
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
        try:
            chunks = index.search(question, k)
        except Exception as exc:  # 單題失敗不中斷整批評測
            results.append(
                RetrievalResult(
                    question_id=qid,
                    dataset=dataset,
                    category=category,
                    expected=expected,
                    ranked=[],
                    top_score=None,
                    error=_describe(exc),
                )
            )
            continue
        results.append(
            RetrievalResult(
                question_id=qid,
                dataset=dataset,
                category=category,
                expected=expected,
                ranked=article_ranking(chunks),
                top_score=chunks[0].score if chunks else None,
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
        result.error = _describe(exc)
        result.seconds = time.perf_counter() - started
        return result
    result.seconds = time.perf_counter() - started
    result.verified = state["verified"]
    result.input_tokens = state["input_tokens"]
    result.output_tokens = state["output_tokens"]
    answer = state["answer"]
    if answer is None:
        result.error = "輸出無法解析"
        return result
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
    curated: list[CuratedQuestion],
    retrieval_results: list[RetrievalResult],
    answer_results: list[AnswerResult],
) -> EvalReport:
    retrieval: dict[str, RetrievalSummary] = {}
    for dataset, items in group_by(retrieval_results, lambda r: r.dataset).items():
        retrieval[dataset] = summarize_retrieval(items)
        if dataset == "curated":
            for category, group in group_by(items, lambda r: r.category).items():
                retrieval[f"curated/{category}"] = summarize_retrieval(group)
    generation: dict[str, GenerationSummary] = {}
    if answer_results and model and judge_model:
        generation["all"] = summarize_generation(answer_results, model, judge_model)
        for category, answers in group_by(answer_results, lambda r: r.category).items():
            generation[category] = summarize_generation(answers, model, judge_model)
    agreement, unstable = consistency(answer_results)
    reviewed = sum(question.reviewed for question in curated)
    return EvalReport(
        created_at=created_at,
        model=model,
        judge_model=judge_model,
        k=k,
        repeat=repeat,
        index=index,
        reviewed=reviewed,
        unreviewed=len(curated) - reviewed,
        retrieval=retrieval,
        generation=generation,
        consistency=agreement,
        unstable=unstable,
        retrieval_results=retrieval_results,
        answer_results=answer_results,
    )


def write_report(eval_dir: Path, report: EvalReport) -> Path:
    eval_dir.mkdir(parents=True, exist_ok=True)
    label = re.sub(r"[^A-Za-z0-9.-]+", "_", report.model or "retrieval")
    path = eval_dir / f"{report.created_at:%Y%m%d-%H%M%S}-{label}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def _describe(exc: Exception) -> str:
    """只取例外類型與狀態碼；訊息可能含請求細節。"""
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    name = type(exc).__name__
    return f"{name}（狀態碼 {code}）" if code else name


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
