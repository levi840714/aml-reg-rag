import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

Category = Literal["lookup", "scenario", "cross_law", "cross_ref", "distractor", "refusal"]


class DatasetError(ValueError):
    pass


class CuratedQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    category: Category
    answerable: bool
    expected_articles: list[str] = []
    forbidden_articles: list[str] = []
    key_points: list[str] = []
    source: Literal["claude_draft", "user"]
    reviewed: bool = False
    note: str | None = None


class SyntheticQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_articles: list[str] = Field(min_length=1)
    generator_model: str
    created_at: datetime


def load_curated(path: Path) -> list[CuratedQuestion]:
    if not path.exists():
        raise DatasetError(f"找不到人工題庫：{path}")
    return _read_jsonl(path, CuratedQuestion)


def load_synthetic(path: Path) -> list[SyntheticQuestion]:
    return _read_jsonl(path, SyntheticQuestion) if path.exists() else []


def write_synthetic(path: Path, questions: list[SyntheticQuestion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{q.model_dump_json()}\n" for q in questions), encoding="utf-8")


def validate_datasets(
    curated: list[CuratedQuestion], synthetic: list[SyntheticQuestion], known_articles: set[str]
) -> list[str]:
    counts = Counter([q.id for q in curated] + [q.id for q in synthetic])
    problems = [f"題目 id 重複：{qid}" for qid, count in counts.items() if count > 1]
    for question in curated:
        problems.extend(_curated_rules(question))
    references = [
        *((q.id, key) for q in curated for key in [*q.expected_articles, *q.forbidden_articles]),
        *((q.id, key) for q in synthetic for key in q.expected_articles),
    ]
    problems.extend(
        f"{qid} 的條文 {key} 不在索引中" for qid, key in references if key not in known_articles
    )
    return problems


def _curated_rules(question: CuratedQuestion) -> list[str]:
    if question.category == "refusal":
        if question.answerable or question.expected_articles or question.key_points:
            return [f"{question.id} 為應拒答題，answerable 須為 false，且不可有預期條文與要點"]
        return []
    if not (question.answerable and question.expected_articles and question.key_points):
        return [f"{question.id} 須為可回答，且至少有一個預期條文與要點"]
    return []


def _read_jsonl[Row: BaseModel](path: Path, model: type[Row]) -> list[Row]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(model.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise DatasetError(f"{path.name} 第 {number} 行格式錯誤：{exc}") from exc
    return rows
