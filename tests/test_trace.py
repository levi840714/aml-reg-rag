import json
from datetime import UTC, datetime

from aml_reg_rag.graph.state import initial_state
from aml_reg_rag.models import ChunkTrace, Retrieval
from aml_reg_rag.trace import append_trace, ask_record
from tests.fakes import answer, cite, point, retrieved


def test_append_trace_writes_daily_jsonl(tmp_path):
    state = initial_state("問題", 6) | {
        "chunks": [retrieved("G0380268:9", "內文", score=0.81234)],
        "answer": answer(point("論點", cite("S1", "內文內文內文內文"))),
        "verified": True,
        "attempts": 1,
        "input_tokens": 100,
        "output_tokens": 20,
    }
    now = datetime(2026, 9, 30, 8, 0, tzinfo=UTC)
    record = ask_record(
        state, table="chunks_test", model="google_genai:gemini-3.8-flash", seconds=1.234, now=now
    )
    path = append_trace(tmp_path, record, now)
    append_trace(tmp_path, record, now)
    assert path.name == "2026-09-30.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    saved = json.loads(lines[0])
    assert saved["retrieved"] == [{"chunk_id": "G0380268:9", "score": 0.8123}]
    assert saved["top_score"] == 0.8123
    assert saved["answer"]["points"][0]["statement"] == "論點"
    assert saved["seconds"] == 1.23


def test_trace_keeps_retrieval_details_without_bodies(tmp_path):
    chunk = retrieved("G0380268:9", "內文", score=3.0)
    retrieval = Retrieval(
        chunks=[chunk],
        queries=["問題", "子查詢"],
        traces=[ChunkTrace(chunk_id="G0380268:9", vector_rank=2, rrf_score=0.03, rerank_score=3)],
        rerank_fallback=None,
        input_tokens=40,
        output_tokens=8,
    )
    state = initial_state("問題", 6) | {"chunks": [chunk], "retrieval": retrieval}
    now = datetime(2026, 9, 30, tzinfo=UTC)
    record = ask_record(state, table="t", model="m", seconds=1.0, now=now)
    assert record["retrieval"]["queries"] == ["問題", "子查詢"]
    assert record["retrieval"]["traces"][0]["rerank_score"] == 3
    assert "chunks" not in record["retrieval"]
