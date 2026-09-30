from contextlib import contextmanager
from datetime import UTC, date, datetime

import pytest
from rich.console import Console
from typer.testing import CliRunner

from aml_reg_rag import cli
from aml_reg_rag.retrieval.index import DatabaseError, IndexMeta
from tests.fakes import FakeIndex, answer, cite, point, retrieved, scripted

runner = CliRunner()
CHUNK = retrieved("G0380268:9", "虛擬資產服務商應確認客戶身分。")
META = IndexMeta(
    table_name="chunks_test",
    embedding_model="google_genai:gemini-embedding-001",
    embedding_dim=1536,
    chunk_max_chars=600,
    source_update_date=date(2026, 9, 18),
    chunk_count=214,
    ingested_at=datetime(2026, 9, 30, tzinfo=UTC),
)
CURATED_ROW = (
    '{"id": "c-001", "question": "虛擬資產服務商要確認客戶身分嗎？", "category": "lookup", '
    '"answerable": true, "expected_articles": ["G0380268:9"], '
    '"key_points": ["應確認客戶身分"], "source": "claude_draft"}\n'
)


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("EVAL_DIR", str(tmp_path / "eval"))
    monkeypatch.setattr(cli, "console", Console(width=200))
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: None)
    return tmp_path


def open_with(chunks):
    @contextmanager
    def opened(settings):
        yield cli.OpenedIndex(db=None, index=FakeIndex(chunks), meta=META)

    return opened


def open_failing(error: Exception):
    @contextmanager
    def opened(settings):
        raise error
        yield

    return opened


def test_search_prints_ranked_chunks(workspace, monkeypatch):
    monkeypatch.setattr(cli, "_open_index", open_with([CHUNK]))
    result = runner.invoke(cli.app, ["search", "確認客戶身分"])
    assert result.exit_code == 0, result.output
    assert "G0380268:9" in result.output


def test_ask_renders_answer_and_writes_trace(workspace, monkeypatch):
    drafter, _ = scripted(
        [answer(point("應確認客戶身分。", cite("S1", "虛擬資產服務商應確認客戶身分")))]
    )
    monkeypatch.setattr(cli, "_open_index", open_with([CHUNK]))
    monkeypatch.setattr(cli, "make_drafter", lambda model, timeout: drafter)
    result = runner.invoke(cli.app, ["ask", "VASP 要確認客戶身分嗎？"])
    assert result.exit_code == 0, result.output
    assert "應確認客戶身分。" in result.output
    traces = list((workspace / "var" / "traces").glob("*.jsonl"))
    assert len(traces) == 1
    assert "VASP 要確認客戶身分嗎？" in traces[0].read_text(encoding="utf-8")


def test_known_errors_exit_with_message(workspace, monkeypatch):
    monkeypatch.setattr(cli, "_open_index", open_failing(DatabaseError("無法連線資料庫")))
    result = runner.invoke(cli.app, ["search", "問題"])
    assert result.exit_code == 1
    assert "無法連線資料庫" in result.output


def test_unexpected_errors_show_only_type_and_status(workspace, monkeypatch):
    class ApiError(Exception):
        status_code = 429

    monkeypatch.setattr(cli, "_open_index", open_failing(ApiError("key=secret-value")))
    result = runner.invoke(cli.app, ["search", "問題"])
    assert result.exit_code == 1
    assert "ApiError（狀態碼 429）" in result.output
    assert "secret-value" not in result.output


@pytest.mark.parametrize("code", [lambda self: 1, "e3q8"])
def test_non_int_error_codes_are_not_shown(workspace, monkeypatch, code):
    class OddError(Exception):
        pass

    OddError.code = code  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "_open_index", open_failing(OddError("x")))
    result = runner.invoke(cli.app, ["search", "問題"])
    assert result.exit_code == 1
    assert "OddError" in result.output
    assert "狀態碼" not in result.output


def test_ask_exits_nonzero_when_verification_fails(workspace, monkeypatch):
    bad = answer(point("錯誤。", cite("S1", "不存在的引述")))
    drafter, _ = scripted([bad, bad])
    monkeypatch.setattr(cli, "_open_index", open_with([CHUNK]))
    monkeypatch.setattr(cli, "make_drafter", lambda model, timeout: drafter)
    result = runner.invoke(cli.app, ["ask", "問題"])
    assert result.exit_code == 1
    assert "引用驗證失敗" in result.output


def test_ask_warns_when_trace_cannot_be_written(workspace, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    drafter, _ = scripted(
        [answer(point("應確認客戶身分。", cite("S1", "虛擬資產服務商應確認客戶身分")))]
    )
    monkeypatch.setattr(cli, "_open_index", open_with([CHUNK]))
    monkeypatch.setattr(cli, "make_drafter", lambda model, timeout: drafter)
    monkeypatch.setattr(cli, "append_trace", fail)
    result = runner.invoke(cli.app, ["ask", "問題"])
    assert result.exit_code == 0, result.output
    assert "應確認客戶身分。" in result.output
    assert "無法寫入 trace：OSError" in result.output


def test_eval_generate_refuses_to_overwrite(workspace):
    (workspace / "eval").mkdir()
    (workspace / "eval" / "synthetic.jsonl").write_text("", encoding="utf-8")
    result = runner.invoke(cli.app, ["eval", "generate"])
    assert result.exit_code == 1
    assert "--overwrite" in result.output


def test_eval_run_reports_missing_curated(workspace):
    result = runner.invoke(cli.app, ["eval", "run", "--retrieval-only"])
    assert result.exit_code == 1
    assert "找不到人工題庫" in result.output


def test_eval_run_retrieval_only_writes_report(workspace, monkeypatch):
    (workspace / "eval").mkdir()
    (workspace / "eval" / "curated.jsonl").write_text(CURATED_ROW, encoding="utf-8")
    monkeypatch.setattr(cli, "_open_index", open_with([CHUNK]))
    monkeypatch.setattr(cli, "indexed_article_keys", lambda db, table: {"G0380268:9"})
    result = runner.invoke(cli.app, ["eval", "run", "--retrieval-only"])
    assert result.exit_code == 0, result.output
    assert "Hit@1" in result.output
    assert len(list((workspace / "var" / "eval").glob("*-retrieval.json"))) == 1


def test_eval_run_rejects_unknown_articles(workspace, monkeypatch):
    (workspace / "eval").mkdir()
    (workspace / "eval" / "curated.jsonl").write_text(CURATED_ROW, encoding="utf-8")
    monkeypatch.setattr(cli, "_open_index", open_with([CHUNK]))
    monkeypatch.setattr(cli, "indexed_article_keys", lambda db, table: set())
    result = runner.invoke(cli.app, ["eval", "run", "--retrieval-only"])
    assert result.exit_code == 1
    assert "G0380268:9 不在索引中" in result.output
