from datetime import UTC, date, datetime

from rich.console import Console

from aml_reg_rag.evaluation.metrics import AnswerResult
from aml_reg_rag.evaluation.runner import build_report
from aml_reg_rag.graph.state import initial_state
from aml_reg_rag.models import ChunkTrace, Retrieval
from aml_reg_rag.render import render_answer, render_eval, render_retrieval, source_numbers
from aml_reg_rag.retrieval.index import IndexMeta
from tests.fakes import answer, cite, point, refusal, retrieved

META = IndexMeta(
    table_name="chunks_test",
    embedding_model="google_genai:gemini-embedding-001",
    embedding_dim=1536,
    chunk_max_chars=600,
    source_update_date=date(2026, 9, 18),
    chunk_count=214,
    ingested_at=datetime(2026, 9, 30, tzinfo=UTC),
)
CHUNKS = [retrieved("G0380268:9", "虛擬資產服務商應於確認客戶身分時，取得必要資訊。")]


def output(state) -> str:
    console = Console(record=True, width=200)
    render_answer(console, state)
    return console.export_text()


def test_source_numbers_follow_first_appearance():
    result = answer(
        point("甲", cite("S3", "引述"), cite("S1", "引述")),
        point("乙", cite("S1", "引述"), cite("S2", "引述")),
    )
    assert source_numbers(result) == {"S3": 1, "S1": 2, "S2": 3}


def test_render_answer_lists_points_quotes_and_sources():
    state = initial_state("問題", 6) | {
        "chunks": CHUNKS,
        "answer": answer(point("應確認客戶身分。", cite("S1", "應於確認客戶身分時"))),
        "verified": True,
    }
    text = output(state)
    assert "1. 應確認客戶身分。 [1]" in text
    assert "「應於確認客戶身分時」" in text
    assert (
        "[1] 提供虛擬資產服務之事業或人員防制洗錢及打擊資恐辦法 第 9 條 "
        "https://law.moj.gov.tw/LawClass/LawSingle.aspx?pcode=G0380268&flno=9"
    ) in text


def test_render_refusal_and_failed_verification():
    state = initial_state("問題", 6) | {
        "chunks": CHUNKS,
        "answer": refusal("段落沒有課稅規定"),
        "verified": False,
        "problems": ["拒答時需說明段落缺少哪些規範"],
    }
    text = output(state)
    assert "引用驗證失敗" in text
    assert "條文不足以回答：段落沒有課稅規定" in text
    assert "第 9 條" in text


def test_render_eval_warns_when_pricing_is_missing():
    report = build_report(
        created_at=datetime(2026, 9, 30, tzinfo=UTC),
        model="unknown:model",
        judge_model="google_genai:gemini-3.8-flash",
        k=6,
        repeat=1,
        index=META,
        light_model="google_genai:gemini-3.5-flash-lite",
        settings={},
        curated=[],
        retrieval_results=[],
        answer_results=[
            AnswerResult(question_id="c-001", category="lookup", run=1, input_tokens=100)
        ],
    )
    console = Console(record=True, width=200)
    render_eval(console, report)
    assert "PRICING 缺少 unknown:model" in console.export_text()


def test_render_answer_prints_markup_literally():
    state = initial_state("問題", 6) | {
        "chunks": CHUNKS,
        "answer": answer(point("[bold]x[/] 重點", cite("S1", "[bold]y[/] 引述"))),
        "verified": True,
    }
    text = output(state)
    assert "[bold]x[/] 重點" in text
    assert "「[bold]y[/] 引述」" in text


def test_render_eval_shows_retrieval_errors_column():
    report = build_report(
        created_at=datetime(2026, 9, 30, tzinfo=UTC),
        model=None,
        judge_model=None,
        k=6,
        repeat=1,
        index=META,
        light_model="google_genai:gemini-3.5-flash-lite",
        settings={},
        curated=[],
        retrieval_results=[],
        answer_results=[],
    )
    console = Console(record=True, width=200)
    render_eval(console, report)
    header = next(line for line in console.export_text().splitlines() if "題數" in line)
    assert header.index("題數") < header.index("錯誤") < header.index("Hit@1")


def test_render_generation_shows_unparsed_column():
    report = build_report(
        created_at=datetime(2026, 9, 30, tzinfo=UTC),
        model="google_genai:gemini-3.8-flash",
        judge_model="google_genai:gemini-3.8-flash",
        k=6,
        repeat=1,
        index=META,
        light_model="google_genai:gemini-3.5-flash-lite",
        settings={},
        curated=[],
        retrieval_results=[],
        answer_results=[AnswerResult(question_id="c-001", category="lookup", run=1, unparsed=True)],
    )
    console = Console(record=True, width=250)
    render_eval(console, report)
    header = next(line for line in console.export_text().splitlines() if "執行" in line)
    assert header.index("錯誤") < header.index("無法解析") < header.index("驗證通過")


def test_render_retrieval_shows_queries_stage_ranks_and_fallbacks():
    chunk = retrieved("G0380268:9", "虛擬資產服務商應確認客戶身分。", score=3.0)
    retrieval = Retrieval(
        chunks=[chunk],
        queries=["原問題", "子查詢"],
        traces=[
            ChunkTrace(
                chunk_id="G0380268:9",
                vector_rank=4,
                bm25_rank=None,
                rrf_score=0.02,
                rerank_score=3,
                final_rank=1,
            )
        ],
        rerank_fallback=None,
        rewrite_fallback="TimeoutError",
    )
    console = Console(record=True, width=200)
    render_retrieval(console, retrieval)
    text = console.export_text()
    assert "query rewriting 失敗，已退回（TimeoutError）" in text
    assert "  - 子查詢" in text
    row = next(line for line in text.splitlines() if "G0380268:9" in line)
    assert "│ 4 " in row and "│ — " in row and "│ 3 " in row


def test_render_retrieval_marks_expanded_chunks():
    hit = retrieved("G0400162:28", "承銷公告", score=3.0)
    expanded = retrieved("G0400162:16", "白皮書", score=0.0).model_copy(
        update={"expanded_from": "G0400162:28", "ref_kind": "apply"}
    )
    retrieval = Retrieval(chunks=[hit, expanded], queries=["問題"], traces=[])
    console = Console(record=True, width=200)
    render_retrieval(console, retrieval)
    text = console.export_text()
    assert "補充" in text
    assert "準用 ← G0400162:28" in text
