import json
from datetime import UTC, date, datetime

from langchain_core.runnables import RunnableLambda

from aml_reg_rag.evaluation.dataset import CuratedQuestion, SyntheticQuestion
from aml_reg_rag.evaluation.judge import JudgeOutput, KeyPointVerdict
from aml_reg_rag.evaluation.metrics import summarize_generation
from aml_reg_rag.evaluation.runner import (
    answer_one,
    build_report,
    run_generation,
    run_retrieval,
    write_report,
)
from aml_reg_rag.graph.state import initial_state
from aml_reg_rag.retrieval.index import IndexMeta
from tests.fakes import FakeIndex, answer, cite, point, refusal, retrieved, scripted

FLASH = "google_genai:gemini-3.8-flash"
SETTINGS = {
    "llm_light_model": "google_genai:gemini-3.5-flash-lite",
    "embedding_model": "google_genai:gemini-embedding-001",
    "embedding_dim": 1536,
    "chunk_max_chars": 600,
    "top_k": 6,
}
VASP = retrieved("G0380268:9", "虛擬資產服務商應於確認客戶身分時，取得必要資訊。", score=0.8)
BANK = retrieved(
    "G0380252:3:p1:k1-4",
    "金融機構確認客戶身分措施，應依下列規定辦理。",
    law_name="金融機構防制洗錢辦法",
    score=0.9,
)
GOOD = answer(point("VASP 應確認客戶身分。", cite("S1", "應於確認客戶身分時，取得必要資訊")))
META = IndexMeta(
    table_name="chunks_test",
    embedding_model="google_genai:gemini-embedding-001",
    embedding_dim=1536,
    chunk_max_chars=600,
    source_update_date=date(2026, 9, 18),
    chunk_count=214,
    ingested_at=datetime(2026, 9, 30, tzinfo=UTC),
)


def question(**overrides) -> CuratedQuestion:
    row = {
        "id": "c-001",
        "question": "VASP 何時要確認客戶身分？",
        "category": "distractor",
        "answerable": True,
        "expected_articles": ["G0380268:9"],
        "forbidden_articles": ["G0380252:3"],
        "key_points": ["建立業務關係時應確認客戶身分"],
        "source": "claude_draft",
    }
    return CuratedQuestion.model_validate(row | overrides)


def refusal_question() -> CuratedQuestion:
    return question(
        id="c-002",
        category="refusal",
        answerable=False,
        expected_articles=[],
        forbidden_articles=[],
        key_points=[],
    )


def fake_ask(result, chunks=(VASP, BANK)):
    def ask(text, k):
        return initial_state(text, k) | {
            "chunks": list(chunks),
            "answer": result,
            "verified": True,
            "attempts": 1,
            "input_tokens": 100,
            "output_tokens": 20,
        }

    return ask


def judge_with(*covered: bool):
    verdicts = [KeyPointVerdict(covered=c, reason="理由") for c in covered]
    judge, _ = scripted([JudgeOutput(verdicts=verdicts)])
    return judge


def test_run_retrieval_skips_refusal_and_includes_synthetic():
    synthetic = SyntheticQuestion(
        id="s-G0380268:9",
        question="交易所何時確認客戶身分？",
        expected_articles=["G0380268:9"],
        generator_model="m",
        created_at=datetime(2026, 9, 30, tzinfo=UTC),
    )
    results = run_retrieval(
        FakeIndex([BANK, VASP]), [question(), refusal_question()], [synthetic], 6
    )
    assert [(r.question_id, r.dataset, r.category) for r in results] == [
        ("c-001", "curated", "distractor"),
        ("s-G0380268:9", "synthetic", "synthetic"),
    ]
    assert results[0].ranked == ["G0380252:3", "G0380268:9"]
    assert results[0].top_score == 0.9


def test_answer_one_scores_citations_and_coverage():
    result = answer_one(fake_ask(GOOD), judge_with(True), question(), 6, run=1)
    assert result.refusal == "correct"
    assert result.cited_articles == ["G0380268:9"]
    assert (result.cites_expected, result.cites_forbidden) == (True, False)
    assert result.coverage == 1.0
    assert (result.input_tokens, result.judge_input_tokens) == (100, 100)


def test_answer_one_flags_forbidden_citation():
    wrong = answer(point("應確認客戶身分。", cite("S2", "金融機構確認客戶身分措施")))
    result = answer_one(fake_ask(wrong), judge_with(False), question(), 6, run=1)
    assert (result.cites_expected, result.cites_forbidden) == (False, True)
    assert result.coverage == 0.0


def test_answer_one_records_false_answer_without_judging():
    judge, calls = scripted([])
    result = answer_one(fake_ask(GOOD), judge, refusal_question(), 6, run=1)
    assert result.refusal == "false_answer"
    assert result.coverage is None
    assert calls == []


def test_answer_one_records_errors():
    def failing(text, k):
        raise TimeoutError

    result = answer_one(failing, judge_with(True), question(), 6, run=1)
    assert result.error == "TimeoutError"


def test_answer_one_counts_unparsed_answer_as_failed_verification():
    result = answer_one(fake_ask(None), judge_with(True), question(), 6, run=1)
    assert result.unparsed is True
    assert result.error is None
    assert result.verified is False
    assert result.input_tokens == 100
    summary = summarize_generation([result], FLASH, FLASH)
    assert (summary.runs, summary.errors, summary.unparsed) == (1, 0, 1)
    assert summary.verified_rate == 0.0


def test_answer_one_error_includes_status_code_not_message():
    class ApiError(Exception):
        status_code = 429

    def failing(text, k):
        raise ApiError("secret request detail")

    result = answer_one(failing, judge_with(True), question(), 6, run=1)
    assert result.error == "ApiError（狀態碼 429）"


def test_run_retrieval_continues_after_search_failure():
    class FlakyIndex(FakeIndex):
        def search(self, query, k):
            if query == "壞題":
                raise TimeoutError
            return super().search(query, k)

    curated = [question(), question(id="c-004", question="壞題")]
    results = run_retrieval(FlakyIndex([VASP]), curated, [], 6)
    assert [r.question_id for r in results] == ["c-001", "c-004"]
    assert results[0].error is None
    assert (results[1].ranked, results[1].top_score, results[1].error) == (
        [],
        None,
        "TimeoutError",
    )


def test_judge_exception_is_recorded():
    def boom(messages):
        raise ValueError

    result = answer_one(fake_ask(GOOD), RunnableLambda(boom), question(), 6, run=1)
    assert result.judge_failed
    assert result.coverage is None


def test_run_generation_repeats_and_reports_progress():
    messages = []
    judge, _ = scripted([])
    results = run_generation(
        fake_ask(refusal()), judge, [question()], 6, repeat=2, progress=messages.append
    )
    assert [r.run for r in results] == [1, 2]
    assert [r.refusal for r in results] == ["false_refusal", "false_refusal"]
    assert messages == ["[1/2] c-001", "[2/2] c-001"]


def test_build_and_write_report(tmp_path):
    curated = [question(), question(id="c-003", category="lookup", reviewed=True)]
    report = build_report(
        created_at=datetime(2026, 9, 30, 8, 0, tzinfo=UTC),
        model=FLASH,
        judge_model=FLASH,
        k=6,
        repeat=1,
        index=META,
        settings=SETTINGS,
        curated=curated,
        retrieval_results=run_retrieval(FakeIndex([VASP]), curated, [], 6),
        answer_results=[answer_one(fake_ask(GOOD), judge_with(True), question(), 6, run=1)],
    )
    assert set(report.retrieval) == {"curated", "curated/distractor", "curated/lookup"}
    assert set(report.generation) == {"all", "distractor"}
    assert (report.reviewed, report.unreviewed) == (1, 1)
    path = write_report(tmp_path, report)
    assert path.name == "20260930-080000-google_genai_gemini-3.8-flash.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["index"]["chunk_count"] == 214
    assert saved["settings"] == SETTINGS


def test_retrieval_only_report(tmp_path):
    report = build_report(
        created_at=datetime(2026, 9, 30, 8, 0, tzinfo=UTC),
        model=None,
        judge_model=None,
        k=6,
        repeat=1,
        index=META,
        settings=SETTINGS,
        curated=[question()],
        retrieval_results=run_retrieval(FakeIndex([VASP]), [question()], [], 6),
        answer_results=[],
    )
    assert report.generation == {}
    assert write_report(tmp_path, report).name == "20260930-080000-retrieval.json"
