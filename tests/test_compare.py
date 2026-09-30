from datetime import UTC, date, datetime

import pytest

from aml_reg_rag.evaluation.compare import (
    ReportError,
    compare_reports,
    load_report,
    report_label,
)
from aml_reg_rag.evaluation.metrics import GenerationSummary, RetrievalSummary
from aml_reg_rag.evaluation.runner import EvalReport
from aml_reg_rag.retrieval.index import IndexMeta

META = IndexMeta(
    table_name="chunks_test",
    embedding_model="google_genai:gemini-embedding-001",
    embedding_dim=1536,
    chunk_max_chars=600,
    source_update_date=date(2026, 9, 18),
    chunk_count=214,
    ingested_at=datetime(2026, 9, 30, tzinfo=UTC),
)


def summary(hit: float) -> RetrievalSummary:
    return RetrievalSummary(
        questions=10, hit_at_1=hit, hit_at_3=hit, hit_at_k=hit, recall_at_k=hit, mrr=hit
    )


def report(settings, retrieval, model=None, generation=None) -> EvalReport:
    return EvalReport(
        created_at=datetime(2026, 9, 30, 8, 0, tzinfo=UTC),
        model=model,
        judge_model=model,
        k=6,
        repeat=1,
        index=META,
        settings=settings,
        reviewed=0,
        unreviewed=20,
        retrieval=retrieval,
        generation=generation or {},
        consistency=None,
        unstable=[],
        retrieval_results=[],
        answer_results=[],
    )


def test_label_reflects_retrieval_flags_and_model():
    v1 = report({"top_k": 6}, {})
    v2 = report({"retrieval_hybrid": True, "retrieval_rerank": True}, {}, model="m")
    assert report_label(v1) == "vector\n09-30 08:00"
    assert report_label(v2) == "hybrid+rerank · m\n09-30 08:00"


def test_rows_align_groups_across_reports():
    first = report({}, {"curated": summary(0.5)})
    second = report(
        {"retrieval_hybrid": True},
        {"curated": summary(0.75), "synthetic": summary(1.0)},
    )
    comparison = compare_reports([first, second])
    rows = dict(comparison.rows)
    assert rows["curated Hit@1"] == ["50%", "75%"]
    assert rows["synthetic MRR"] == ["—", "1.000"]
    assert not any(name.startswith("生成") for name in rows)


def test_generation_rows_appear_when_any_report_has_them():
    generation = GenerationSummary(
        runs=3,
        errors=0,
        unparsed=0,
        verified_rate=1.0,
        refusal_accuracy=0.9,
        false_refusals=0,
        false_answers=0,
        cites_expected_rate=1.0,
        cites_forbidden=0,
        coverage=0.8,
        judge_failures=0,
        avg_seconds=4.0,
        cost_usd=0.1234,
    )
    rows = dict(
        compare_reports(
            [report({}, {}), report({}, {}, model="m", generation={"all": generation})]
        ).rows
    )
    assert rows["生成 驗證通過"] == ["—", "100%"]
    assert rows["生成 成本"] == ["—", "$0.1234"]


def test_load_report_round_trip_and_invalid_file(tmp_path):
    path = tmp_path / "r.json"
    original = report({"retrieval_rerank": False}, {"curated": summary(0.5)})
    path.write_text(original.model_dump_json(), encoding="utf-8")
    assert load_report(path) == original
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ReportError, match="bad.json"):
        load_report(bad)


def test_load_report_rejects_undecodable_file(tmp_path):
    bad = tmp_path / "binary.json"
    bad.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ReportError, match="binary.json"):
        load_report(bad)
