import json
from datetime import UTC, datetime

import pytest

from aml_reg_rag.evaluation.dataset import (
    CuratedQuestion,
    DatasetError,
    SyntheticQuestion,
    load_curated,
    load_synthetic,
    validate_datasets,
    write_synthetic,
)

KNOWN = {"G0380268:2", "G0380252:3"}


def curated(qid="c-001", category="lookup", **overrides) -> CuratedQuestion:
    row = {
        "id": qid,
        "question": "虛擬資產服務商的定義是什麼？",
        "category": category,
        "answerable": True,
        "expected_articles": ["G0380268:2"],
        "key_points": ["在我國境內為他人從事虛擬資產活動為業"],
        "source": "claude_draft",
    }
    if category == "refusal":
        row |= {"answerable": False, "expected_articles": [], "key_points": []}
    return CuratedQuestion.model_validate(row | overrides)


def synthetic(article="G0380268:2") -> SyntheticQuestion:
    return SyntheticQuestion(
        id=f"s-{article}",
        question="交易所要做哪些事？",
        expected_articles=[article],
        generator_model="google_genai:gemini-3.5-flash-lite",
        created_at=datetime(2026, 9, 30, tzinfo=UTC),
    )


def test_load_curated_reads_jsonl(tmp_path):
    path = tmp_path / "curated.jsonl"
    path.write_text(curated().model_dump_json() + "\n\n", encoding="utf-8")
    assert load_curated(path) == [curated()]


def test_load_curated_missing_file_raises(tmp_path):
    with pytest.raises(DatasetError, match="找不到人工題庫"):
        load_curated(tmp_path / "curated.jsonl")


def test_invalid_json_reports_line_number(tmp_path):
    path = tmp_path / "curated.jsonl"
    path.write_text(curated().model_dump_json() + "\n{bad\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="第 2 行"):
        load_curated(path)


def test_unknown_category_is_rejected(tmp_path):
    path = tmp_path / "curated.jsonl"
    row = json.loads(curated().model_dump_json()) | {"category": "other"}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="第 1 行"):
        load_curated(path)


def test_synthetic_round_trip(tmp_path):
    path = tmp_path / "synthetic.jsonl"
    assert load_synthetic(path) == []
    write_synthetic(path, [synthetic()])
    assert load_synthetic(path) == [synthetic()]


def test_valid_datasets_have_no_problems():
    questions = [curated(), curated("c-002", "refusal")]
    assert validate_datasets(questions, [synthetic()], KNOWN) == []


def test_duplicate_ids_are_reported():
    assert validate_datasets([curated(), curated()], [], KNOWN) == ["題目 id 重複：c-001"]


def test_refusal_rules():
    problems = validate_datasets([curated("c-002", "refusal", answerable=True)], [], KNOWN)
    assert "應拒答題" in problems[0]


def test_answerable_rules():
    problems = validate_datasets([curated(key_points=[])], [], KNOWN)
    assert "須為可回答" in problems[0]


def test_unknown_articles_are_reported():
    problems = validate_datasets(
        [curated(forbidden_articles=["G0380252:99"])], [synthetic("X0000000:1")], KNOWN
    )
    assert problems == [
        "c-001 的條文 G0380252:99 不在索引中",
        "s-X0000000:1 的條文 X0000000:1 不在索引中",
    ]


def test_misspelled_field_is_rejected(tmp_path):
    path = tmp_path / "curated.jsonl"
    row = json.loads(curated().model_dump_json()) | {"forbiden_articles": ["G0380252:3"]}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="第 1 行"):
        load_curated(path)


def test_synthetic_without_expected_articles_is_rejected(tmp_path):
    path = tmp_path / "synthetic.jsonl"
    row = json.loads(synthetic().model_dump_json()) | {"expected_articles": []}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="第 1 行"):
        load_synthetic(path)


def test_blank_question_is_rejected(tmp_path):
    path = tmp_path / "curated.jsonl"
    row = json.loads(curated().model_dump_json()) | {"question": "  "}
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="第 1 行"):
        load_curated(path)
