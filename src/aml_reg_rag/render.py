from rich.console import Console
from rich.markup import escape
from rich.table import Table

from aml_reg_rag.evaluation.compare import Comparison
from aml_reg_rag.evaluation.runner import EvalReport
from aml_reg_rag.graph.state import AskState, labeled_chunks
from aml_reg_rag.ingest.pipeline import IngestReport
from aml_reg_rag.models import Answer, Retrieval, RetrievedChunk

PREVIEW_CHARS = 40


def render_warnings(console: Console, warnings: list[str]) -> None:
    for warning in warnings:
        console.print(f"[yellow]警告：{escape(warning)}[/]")


def render_ingest(console: Console, report: IngestReport) -> None:
    render_warnings(console, report.warnings)
    console.print(f"已建立索引 [bold]{escape(report.meta.table_name)}[/]")
    console.print(
        f"法規 {report.laws} 部、條文 {report.articles} 條、段落 {report.chunks} 段"
        f"（超過切段上限 {report.oversized} 段）"
    )
    console.print(f"資料來源更新日期：{report.meta.source_update_date}")


def render_retrieval(console: Console, retrieval: Retrieval) -> None:
    fallbacks = [
        f"{name}失敗，已退回（{reason}）"
        for name, reason in (
            ("query rewriting ", retrieval.rewrite_fallback),
            ("rerank ", retrieval.rerank_fallback),
        )
        if reason
    ]
    render_warnings(console, fallbacks)
    if len(retrieval.queries) > 1:
        console.print("查詢：")
        for query in retrieval.queries:
            console.print(f"  - {escape(query)}")
    traces = {trace.chunk_id: trace for trace in retrieval.traces}
    table = Table("排名", "分數", "向量", "BM25", "rerank", "chunk_id", "條文", "內文開頭")
    for rank, chunk in enumerate(retrieval.chunks, start=1):
        trace = traces.get(chunk.chunk_id)
        preview = chunk.body.replace("\n", " ")[:PREVIEW_CHARS]
        table.add_row(
            str(rank),
            f"{chunk.score:.3f}",
            _optional(trace.vector_rank if trace else None),
            _optional(trace.bm25_rank if trace else None),
            _optional(trace.rerank_score if trace else None),
            chunk.chunk_id,
            escape(chunk.title),
            escape(preview),
        )
    console.print(table)


def render_compare(console: Console, comparison: Comparison) -> None:
    table = Table("指標", *(escape(column) for column in comparison.columns))
    for name, values in comparison.rows:
        table.add_row(escape(name), *values)
    console.print(table)


def source_numbers(answer: Answer) -> dict[str, int]:
    """依來源首次出現的順序編號。"""
    numbers: dict[str, int] = {}
    for point in answer.points:
        for citation in point.citations:
            numbers.setdefault(citation.source, len(numbers) + 1)
    return numbers


def render_answer(console: Console, state: AskState, *, show_context: bool = False) -> None:
    if show_context and state["retrieval"] is not None:
        render_retrieval(console, state["retrieval"])
    if not state["verified"]:
        console.print("[bold red]引用驗證失敗：[/]")
        for problem in state["problems"]:
            console.print(f"  - {escape(problem)}")
    answer = state["answer"]
    if answer is None:
        return
    if not answer.answerable:
        _render_refusal(console, answer, state["chunks"])
        return
    numbers = source_numbers(answer)
    for number, point in enumerate(answer.points, start=1):
        marks = "".join(f"[{numbers[citation.source]}]" for citation in point.citations)
        console.print(f"{number}. {escape(point.statement)} {escape(marks)}")
        for citation in point.citations:
            console.print(f"   [dim]「{escape(citation.quote)}」[/]")
    labeled = labeled_chunks(state["chunks"])
    console.print("\n來源：")
    for label, number in numbers.items():
        chunk = labeled.get(label)
        detail = f"{chunk.title} {chunk.source_url}" if chunk else f"{label}（不在檢索結果中）"
        console.print(f"  {escape(f'[{number}]')} {escape(detail)}")


def render_eval(console: Console, report: EvalReport) -> None:
    console.print(
        f"模型：{escape(report.model or '（只評估檢索）')}　"
        f"judge：{escape(report.judge_model or '—')}　"
        f"k={report.k}　repeat={report.repeat}　索引：{escape(report.index.table_name)}"
    )
    console.print(f"人工題：已審核 {report.reviewed} 題、未審核 {report.unreviewed} 題")
    retrieval = Table(
        "檢索",
        "題數",
        "錯誤",
        "退回",
        "Hit@1",
        "Hit@3",
        "Hit@k",
        "Recall@k",
        "MRR",
        "平均秒數",
        "成本",
    )
    for name, summary in report.retrieval.items():
        retrieval.add_row(
            name,
            str(summary.questions),
            str(summary.errors),
            str(summary.fallbacks),
            _pct(summary.hit_at_1),
            _pct(summary.hit_at_3),
            _pct(summary.hit_at_k),
            _pct(summary.recall_at_k),
            f"{summary.mrr:.3f}",
            "—" if summary.avg_seconds is None else f"{summary.avg_seconds:.1f}",
            "—" if summary.cost_usd is None else f"${summary.cost_usd:.4f}",
        )
    console.print(retrieval)
    if report.generation:
        _render_generation(console, report)


def _render_generation(console: Console, report: EvalReport) -> None:
    table = Table(
        "生成",
        "執行",
        "錯誤",
        "無法解析",
        "驗證通過",
        "拒答正確",
        "該答卻拒",
        "該拒卻答",
        "引用預期",
        "引用禁止",
        "要點涵蓋",
        "平均秒數",
        "成本",
    )
    for name, s in report.generation.items():
        table.add_row(
            name,
            str(s.runs),
            str(s.errors),
            str(s.unparsed),
            _pct(s.verified_rate),
            _pct(s.refusal_accuracy),
            str(s.false_refusals),
            str(s.false_answers),
            _pct(s.cites_expected_rate),
            str(s.cites_forbidden),
            _pct(s.coverage),
            "—" if s.avg_seconds is None else f"{s.avg_seconds:.1f}",
            "—" if s.cost_usd is None else f"${s.cost_usd:.4f}",
        )
    console.print(table)
    if report.generation["all"].cost_usd is None:
        render_warnings(
            console, [f"PRICING 缺少 {report.model} 或 {report.judge_model} 的價格，未估算成本"]
        )
    if report.consistency is not None:
        console.print(f"重複執行一致率：{_pct(report.consistency)}")
        for item in report.unstable:
            console.print(
                f"  不穩定：{escape(item.question_id)} {escape(', '.join(item.outcomes))}"
            )


def _render_refusal(console: Console, answer: Answer, chunks: list[RetrievedChunk]) -> None:
    console.print(f"[bold yellow]條文不足以回答：[/]{escape(answer.missing or '')}")
    console.print("檢索到的條文：")
    for chunk in chunks:
        console.print(f"  - {escape(chunk.title)} {escape(chunk.source_url)}")


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _optional(value: int | None) -> str:
    return "—" if value is None else str(value)
