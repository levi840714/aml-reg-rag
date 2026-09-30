import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import typer
from dotenv import load_dotenv
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape

from aml_reg_rag import render
from aml_reg_rag.config import PROJECT_ROOT, Settings
from aml_reg_rag.evaluation.compare import ReportError, compare_reports, load_report
from aml_reg_rag.evaluation.dataset import (
    DatasetError,
    load_curated,
    load_synthetic,
    validate_datasets,
    write_synthetic,
)
from aml_reg_rag.evaluation.generate import generate_questions, make_generator
from aml_reg_rag.evaluation.judge import make_judge
from aml_reg_rag.evaluation.runner import (
    build_report,
    run_generation,
    run_retrieval,
    write_report,
)
from aml_reg_rag.graph.build import build_graph, make_drafter, run_ask
from aml_reg_rag.ingest.pipeline import load_source, run_ingest
from aml_reg_rag.ingest.source import SourceError
from aml_reg_rag.llm import describe_error
from aml_reg_rag.retrieval.embeddings import EmbeddingConfigError, make_embeddings
from aml_reg_rag.retrieval.index import (
    Database,
    DatabaseError,
    IndexMeta,
    IndexNotReadyError,
    describe_target,
    indexed_article_keys,
    is_local,
    open_index,
)
from aml_reg_rag.retrieval.pipeline import RetrievalOptions, Retriever, make_pipeline
from aml_reg_rag.trace import append_trace, ask_record

KNOWN_ERRORS = (
    DatabaseError,
    DatasetError,
    EmbeddingConfigError,
    IndexNotReadyError,
    ReportError,
    SourceError,
)

# 關閉 locals 顯示，避免 traceback 印出含資料庫密碼的 Settings
app = typer.Typer(
    help="台灣防制洗錢法規問答（RAG）", no_args_is_help=True, pretty_exceptions_show_locals=False
)
eval_app = typer.Typer(
    help="評估檢索與回答品質", no_args_is_help=True, pretty_exceptions_show_locals=False
)
app.add_typer(eval_app, name="eval")
console = Console()

HYBRID = typer.Option(None, "--hybrid/--no-hybrid", help="加入 BM25（覆寫 RETRIEVAL_HYBRID）")
REWRITE = typer.Option(None, "--rewrite/--no-rewrite", help="拆解子查詢（覆寫 RETRIEVAL_REWRITE）")
RERANK = typer.Option(None, "--rerank/--no-rerank", help="LLM rerank（覆寫 RETRIEVAL_RERANK）")
REPORT_PATHS = typer.Argument(..., exists=True, dir_okay=False, help="eval 結果檔")


@dataclass(frozen=True)
class OpenedIndex:
    db: Database
    retriever: Retriever
    meta: IndexMeta


@app.callback()
def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    # structured output 不使用 AFC，SDK 仍會印出 AFC 使用建議，對 CLI 使用者只是雜訊
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)


@app.command()
def ingest(
    refresh: bool = typer.Option(False, "--refresh", help="重新下載全國法規資料庫資料"),
    allow_remote: bool = typer.Option(
        False, "--allow-remote", help="允許對非本機資料庫刪表重建索引"
    ),
) -> None:
    """下載法規、切段並重建索引。"""
    with _handle_errors():
        settings = Settings()
        target = describe_target(settings.database_url)
        console.print(f"目標資料庫：{escape(target)}")
        if not is_local(settings.database_url) and not allow_remote:
            console.print(
                f"[red]DATABASE_URL 指向非本機資料庫（{escape(target)}），ingest 會刪表重建；"
                "確定要執行請加 --allow-remote[/]"
            )
            raise typer.Exit(1)
        db = Database.connect(settings.database_url)
        try:
            report = run_ingest(settings, db, make_embeddings(settings), refresh=refresh)
        finally:
            db.close()
        render.render_ingest(console, report)


@app.command()
def search(
    question: str,
    k: int | None = typer.Option(None, "--k", min=1),
    hybrid: bool | None = HYBRID,
    rewrite: bool | None = REWRITE,
    rerank: bool | None = RERANK,
) -> None:
    """只檢索；開啟 rewrite 或 rerank 時會呼叫輕量模型。"""
    with _handle_errors():
        settings = Settings()
        options = RetrievalOptions.from_settings(
            settings, hybrid=hybrid, rewrite=rewrite, rerank=rerank
        )
        with _open_index(settings, options) as opened:
            retrieval = opened.retriever.retrieve(question, k or settings.top_k)
        render.render_retrieval(console, retrieval)


@app.command()
def ask(
    question: str,
    k: int | None = typer.Option(None, "--k", min=1),
    show_context: bool = typer.Option(False, "--show-context", help="一併顯示檢索到的段落"),
    hybrid: bool | None = HYBRID,
    rewrite: bool | None = REWRITE,
    rerank: bool | None = RERANK,
) -> None:
    """依法規條文回答問題。"""
    with _handle_errors():
        settings = Settings()
        options = RetrievalOptions.from_settings(
            settings, hybrid=hybrid, rewrite=rewrite, rerank=rerank
        )
        with _open_index(settings, options) as opened:
            drafter = make_drafter(settings.llm_model, settings.llm_timeout)
            graph = build_graph(opened.retriever, drafter)
            started = time.perf_counter()
            state = run_ask(graph, question, k or settings.top_k)
            seconds = time.perf_counter() - started
        render.render_answer(console, state, show_context=show_context)
        now = datetime.now(UTC)
        record = ask_record(
            state, table=opened.meta.table_name, model=settings.llm_model, seconds=seconds, now=now
        )
        try:
            append_trace(settings.var_dir / "traces", record, now)
        except OSError as exc:
            render.render_warnings(console, [f"無法寫入 trace：{type(exc).__name__}"])
        if not state["verified"]:
            raise typer.Exit(1)


@eval_app.command("generate")
def eval_generate(
    overwrite: bool = typer.Option(False, "--overwrite", help="覆寫已存在的自動生成題"),
) -> None:
    """以輕量模型為每條條文產生一題。"""
    with _handle_errors():
        settings = Settings()
        path = settings.eval_dir / "synthetic.jsonl"
        if path.exists() and not overwrite:
            console.print(f"[red]{escape(str(path))} 已存在，要重新產生請加 --overwrite[/]")
            raise typer.Exit(1)
        source = load_source(settings)
        render.render_warnings(console, source.warnings)
        generator = make_generator(settings.llm_light_model, settings.llm_timeout)
        result = generate_questions(
            source.articles, generator, settings.llm_light_model, datetime.now(UTC)
        )
        write_synthetic(path, result.questions)
        console.print(
            f"已產生 {len(result.questions)} 題（失敗 {result.failures} 題）：{escape(str(path))}"
        )


@eval_app.command("run")
def eval_run(
    retrieval_only: bool = typer.Option(False, "--retrieval-only", help="只評估檢索"),
    model: str | None = typer.Option(None, "--model", help="覆寫生成模型 <provider:model>"),
    k: int | None = typer.Option(None, "--k", min=1),
    repeat: int = typer.Option(1, "--repeat", min=1, help="生成階段每題執行次數"),
    hybrid: bool | None = HYBRID,
    rewrite: bool | None = REWRITE,
    rerank: bool | None = RERANK,
) -> None:
    """執行 eval 並寫入 var/eval/。"""
    with _handle_errors():
        settings = Settings()
        options = RetrievalOptions.from_settings(
            settings, hybrid=hybrid, rewrite=rewrite, rerank=rerank
        )
        top_k = k or settings.top_k
        model_name = None if retrieval_only else model or settings.llm_model
        curated = load_curated(settings.eval_dir / "curated.jsonl")
        synthetic = load_synthetic(settings.eval_dir / "synthetic.jsonl")
        if not synthetic:
            render.render_warnings(console, ["尚未產生自動生成題，可執行 reg eval generate"])
        with _open_index(settings, options) as opened:
            known = indexed_article_keys(opened.db, opened.meta.table_name)
            problems = validate_datasets(curated, synthetic, known)
            if problems:
                raise DatasetError("題庫驗證失敗：\n" + "\n".join(problems))
            retrieval_results = run_retrieval(opened.retriever, curated, synthetic, top_k)
            answer_results = []
            if model_name:
                drafter = make_drafter(model_name, settings.llm_timeout)
                graph = build_graph(opened.retriever, drafter)
                answer_results = run_generation(
                    lambda question, size: run_ask(graph, question, size),
                    make_judge(settings.eval_judge_model, settings.llm_timeout),
                    curated,
                    top_k,
                    repeat,
                    progress=lambda message: console.print(f"[dim]{message}[/]"),
                )
        report = build_report(
            created_at=datetime.now(UTC),
            model=model_name,
            judge_model=settings.eval_judge_model if model_name else None,
            k=top_k,
            repeat=repeat,
            index=opened.meta,
            settings=_report_settings(settings, options),
            light_model=settings.llm_light_model,
            curated=curated,
            retrieval_results=retrieval_results,
            answer_results=answer_results,
        )
        path = write_report(settings.var_dir / "eval", report)
        render.render_eval(console, report)
        console.print(f"結果已寫入 {escape(str(path))}")


@eval_app.command("compare")
def eval_compare(
    paths: list[Path] = REPORT_PATHS,
) -> None:
    """並排比較多個 eval 結果。"""
    with _handle_errors():
        render.render_compare(console, compare_reports([load_report(path) for path in paths]))


def _report_settings(settings: Settings, options: RetrievalOptions) -> dict[str, str | int | bool]:
    return {
        "llm_light_model": settings.llm_light_model,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "chunk_max_chars": settings.chunk_max_chars,
        "top_k": settings.top_k,
        "retrieval_hybrid": options.hybrid,
        "retrieval_rewrite": options.rewrite,
        "retrieval_rerank": options.rerank,
        "retrieval_candidates": options.candidates,
    }


@contextmanager
def _open_index(settings: Settings, options: RetrievalOptions) -> Iterator[OpenedIndex]:
    db = Database.connect(settings.database_url)
    try:
        index, meta, warnings = open_index(db, settings, make_embeddings(settings))
        render.render_warnings(console, warnings)
        retriever = make_pipeline(db, meta.table_name, index, settings, options)
        yield OpenedIndex(db, retriever, meta)
    finally:
        db.close()


@contextmanager
def _handle_errors() -> Iterator[None]:
    try:
        yield
    except KNOWN_ERRORS as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(1) from exc
    except typer.Exit:
        raise
    except ValidationError as exc:
        # 只列欄位名稱，輸入值可能是誤填的 key
        # 跨欄位檢查沒有 loc，改用錯誤說明（由程式產生，不含輸入值）
        fields = ", ".join(
            ".".join(str(part) for part in error["loc"]) or error["msg"] for error in exc.errors()
        )
        console.print(f"[red]設定值不合法：{escape(fields)}[/]")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(
            f"[red]執行失敗：{escape(describe_error(exc))}；請確認 .env 的設定與 API key[/]"
        )
        raise typer.Exit(1) from exc
