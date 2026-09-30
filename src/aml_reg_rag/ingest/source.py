import json
import re
import tomllib
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from aml_reg_rag.models import LawArticle

MOJ_API = "https://law.moj.gov.tw/api"
DATASETS = {"ChLaw": "Ch/Law/json", "ChOrder": "Ch/Order/json"}
DOWNLOAD_TIMEOUT = 120.0

_PCODE = re.compile(r"pcode=([A-Za-z0-9]+)", re.IGNORECASE)
_ARTICLE_NO = re.compile(r"第\s*([\d-]+)\s*條")
_SECTION = re.compile(r"^第\s*\S+\s*節")
_UPDATE_DATE = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})")


class SourceError(RuntimeError):
    pass


class CorpusLaw(BaseModel):
    pcode: str
    name: str


@dataclass(frozen=True)
class FetchResult:
    paths: list[Path]
    warnings: list[str]


@dataclass(frozen=True)
class SourceData:
    update_date: date
    articles: list[LawArticle]
    warnings: list[str]


def load_corpus(path: Path) -> list[CorpusLaw]:
    with path.open("rb") as file:
        data = tomllib.load(file)
    return [CorpusLaw.model_validate(item) for item in data.get("laws", [])]


def fetch_datasets(
    raw_dir: Path, *, refresh: bool = False, client: httpx.Client | None = None
) -> FetchResult:
    raw_dir.mkdir(parents=True, exist_ok=True)
    http = client or httpx.Client(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True)
    paths: list[Path] = []
    warnings: list[str] = []
    try:
        for name, endpoint in DATASETS.items():
            path = raw_dir / f"{name}.json.zip"
            if refresh or not path.exists():
                warning = _download_or_keep_cache(http, f"{MOJ_API}/{endpoint}", path, name)
                if warning:
                    warnings.append(warning)
            paths.append(path)
    finally:
        if client is None:
            http.close()
    return FetchResult(paths, warnings)


def _download_or_keep_cache(client: httpx.Client, url: str, path: Path, name: str) -> str | None:
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        if not path.exists():
            raise SourceError(f"下載 {name} 失敗（{type(exc).__name__}），且沒有快取") from exc
        cached = datetime.fromtimestamp(path.stat().st_mtime).date()
        return f"下載 {name} 失敗，改用 {cached} 的快取"
    partial = path.with_suffix(".part")
    partial.write_bytes(response.content)
    partial.replace(path)
    return None


def read_dataset(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(archive.namelist()[0])
    data: dict[str, Any] = json.loads(raw.decode("utf-8-sig"))
    return data


def parse_laws(datasets: list[dict[str, Any]], corpus: list[CorpusLaw]) -> SourceData:
    found: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        for law in dataset.get("Laws", []):
            match = _PCODE.search(law.get("LawURL", ""))
            if match:
                found[match.group(1)] = law
    missing = [law.pcode for law in corpus if law.pcode not in found]
    if missing:
        raise SourceError(f"來源中找不到以下 pcode：{'、'.join(missing)}")

    articles: list[LawArticle] = []
    warnings: list[str] = []
    for entry in corpus:
        law = found[entry.pcode]
        if law["LawName"] != entry.name:
            warnings.append(
                f"{entry.pcode} 在來源中名為「{law['LawName']}」，"
                f"與 corpus.toml 的「{entry.name}」不同"
            )
        parsed, empty = _parse_articles(entry.pcode, law)
        articles.extend(parsed)
        warnings.extend(f"{law['LawName']} {article_no} 內容為空，已略過" for article_no in empty)
    return SourceData(_update_date(datasets), articles, warnings)


def _parse_articles(pcode: str, law: dict[str, Any]) -> tuple[list[LawArticle], list[str]]:
    articles: list[LawArticle] = []
    empty: list[str] = []
    chapter: str | None = None
    section: str | None = None
    for item in law.get("LawArticles", []):
        content = item.get("ArticleContent", "").strip()
        if item.get("ArticleType") == "C":
            heading = " ".join(content.split())
            if _SECTION.match(heading):
                section = heading
            else:
                chapter, section = heading, None
            continue
        if item.get("ArticleType") != "A":
            continue
        article_no = _article_no(item.get("ArticleNo", ""))
        if not content:
            empty.append(item.get("ArticleNo", article_no))
            continue
        articles.append(
            LawArticle(
                pcode=pcode,
                law_name=law["LawName"],
                law_level=law["LawLevel"],
                law_modified_date=datetime.strptime(law["LawModifiedDate"], "%Y%m%d").date(),
                chapter=" ".join(part for part in (chapter, section) if part) or None,
                article_no=article_no,
                content=content,
            )
        )
    return articles, empty


def _article_no(raw: str) -> str:
    match = _ARTICLE_NO.search(raw)
    if not match:
        raise SourceError(f"無法解析條號：{raw}")
    return match.group(1)


def _update_date(datasets: list[dict[str, Any]]) -> date:
    dates = []
    for dataset in datasets:
        match = _UPDATE_DATE.search(dataset.get("UpdateDate", ""))
        if match:
            year, month, day = (int(part) for part in match.groups())
            dates.append(date(year, month, day))
    if not dates:
        raise SourceError("來源資料缺少 UpdateDate")
    return max(dates)
