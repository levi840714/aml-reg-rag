import json
from datetime import UTC, datetime

from aml_reg_rag.graph.state import initial_state
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
