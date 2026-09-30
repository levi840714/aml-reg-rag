import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markup import escape

from aml_reg_rag import render
from aml_reg_rag.config import PROJECT_ROOT, Settings
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
from aml_reg_rag.retrieval.embeddings import EmbeddingConfigError, make_embeddings
from aml_reg_rag.retrieval.index import (
    ChunkIndex,
    Database,
    DatabaseError,
    IndexMeta,
    IndexNotReadyError,
    indexed_article_keys,
    open_index,
)
from aml_reg_rag.trace import append_trace, ask_record

KNOWN_ERRORS = (
    DatabaseError,
    DatasetError,
    EmbeddingConfigError,
    IndexNotReadyError,
    SourceError,
)

app = typer.Typer(help="台灣防制洗錢法規問答（RAG）", no_args_is_help=True)
eval_app = typer.Typer(help="評估檢索與回答品質", no_args_is_help=True)
app.add_typer(eval_app, name="eval")
console = Console()


@dataclass(frozen=True)
class OpenedIndex:
    db: Database
    index: ChunkIndex
    meta: IndexMeta


@app.callback()
def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


@app.command()
def ingest(
    refresh: bool = typer.Option(False, "--refresh", help="重新下載全國法規資料庫資料"),
) -> None:
    """下載法規、切段並重建索引。"""
    settings = Settings()
    with _handle_errors():
        db = Database.connect(settings.database_url)
        try:
            report = run_ingest(settings, db, make_embeddings(settings), refresh=refresh)
        finally:
            db.close()
    render.render_ingest(console, report)


@app.command()
def search(question: str, k: int | None = typer.Option(None, "--k", min=1)) -> None:
    """只檢索，不呼叫 LLM。"""
    settings = Settings()
    with _handle_errors(), _open_index(settings) as opened:
        chunks = opened.index.search(question, k or settings.top_k)
    render.render_search(console, chunks)


@app.command()
def ask(
    question: str,
    k: int | None = typer.Option(None, "--k", min=1),
    show_context: bool = typer.Option(False, "--show-context", help="一併顯示檢索到的段落"),
) -> None:
    """依法規條文回答問題。"""
    settings = Settings()
    with _handle_errors(), _open_index(settings) as opened:
        graph = build_graph(opened.index, make_drafter(settings.llm_model, settings.llm_timeout))
        started = time.perf_counter()
        state = run_ask(graph, question, k or settings.top_k)
        seconds = time.perf_counter() - started
    now = datetime.now(UTC)
    record = ask_record(
        state, table=opened.meta.table_name, model=settings.llm_model, seconds=seconds, now=now
    )
    append_trace(settings.var_dir / "traces", record, now)
    render.render_answer(console, state, show_context=show_context)


@eval_app.command("generate")
def eval_generate(
    overwrite: bool = typer.Option(False, "--overwrite", help="覆寫已存在的自動生成題"),
) -> None:
    """以輕量模型為每條條文產生一題。"""
    settings = Settings()
    path = settings.eval_dir / "synthetic.jsonl"
    if path.exists() and not overwrite:
        console.print(f"[red]{path} 已存在，要重新產生請加 --overwrite[/]")
        raise typer.Exit(1)
    with _handle_errors():
        source = load_source(settings)
        render.render_warnings(console, source.warnings)
        generator = make_generator(settings.llm_light_model, settings.llm_timeout)
        result = generate_questions(
            source.articles, generator, settings.llm_light_model, datetime.now(UTC)
        )
    write_synthetic(path, result.questions)
    console.print(f"已產生 {len(result.questions)} 題（失敗 {result.failures} 題）：{path}")


@eval_app.command("run")
def eval_run(
    retrieval_only: bool = typer.Option(False, "--retrieval-only", help="只評估檢索"),
    model: str | None = typer.Option(None, "--model", help="覆寫生成模型 <provider:model>"),
    k: int | None = typer.Option(None, "--k", min=1),
    repeat: int = typer.Option(1, "--repeat", min=1, help="生成階段每題執行次數"),
) -> None:
    """執行 eval 並寫入 var/eval/。"""
    settings = Settings()
    top_k = k or settings.top_k
    model_name = None if retrieval_only else model or settings.llm_model
    with _handle_errors():
        curated = load_curated(settings.eval_dir / "curated.jsonl")
        synthetic = load_synthetic(settings.eval_dir / "synthetic.jsonl")
        if not synthetic:
            render.render_warnings(console, ["尚未產生自動生成題，可執行 reg eval generate"])
        with _open_index(settings) as opened:
            known = indexed_article_keys(opened.db, opened.meta.table_name)
            problems = validate_datasets(curated, synthetic, known)
            if problems:
                raise DatasetError("題庫驗證失敗：\n" + "\n".join(problems))
            retrieval_results = run_retrieval(opened.index, curated, synthetic, top_k)
            answer_results = []
            if model_name:
                graph = build_graph(opened.index, make_drafter(model_name, settings.llm_timeout))
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
            curated=curated,
            retrieval_results=retrieval_results,
            answer_results=answer_results,
        )
    path = write_report(settings.var_dir / "eval", report)
    render.render_eval(console, report)
    console.print(f"結果已寫入 {path}")


@contextmanager
def _open_index(settings: Settings) -> Iterator[OpenedIndex]:
    db = Database.connect(settings.database_url)
    try:
        index, meta, warnings = open_index(db, settings, make_embeddings(settings))
        render.render_warnings(console, warnings)
        yield OpenedIndex(db, index, meta)
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
    except Exception as exc:
        # 外部 API 的錯誤訊息可能帶有請求細節，只顯示例外類型與狀態碼
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        suffix = f"（狀態碼 {status}）" if status else ""
        console.print(f"[red]執行失敗：{type(exc).__name__}{suffix}[/]")
        raise typer.Exit(1) from exc
