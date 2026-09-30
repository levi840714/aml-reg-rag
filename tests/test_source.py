import io
import json
import zipfile
from datetime import date
from pathlib import Path

import httpx
import pytest

from aml_reg_rag.ingest.source import (
    CorpusLaw,
    SourceError,
    fetch_datasets,
    load_corpus,
    parse_laws,
    read_dataset,
)

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS = [
    CorpusLaw(pcode="I0030047", name="資恐防制法"),
    CorpusLaw(pcode="G0380252", name="金融機構防制洗錢辦法"),
    CorpusLaw(pcode="G0400162", name="提供虛擬資產服務之事業或人員洗錢防制登記辦法"),
]


def datasets() -> list[dict]:
    return [
        json.loads((FIXTURES / f"{name}.sample.json").read_text(encoding="utf-8"))
        for name in ("ChLaw", "ChOrder")
    ]


def test_parse_laws_extracts_articles_in_corpus_order():
    source = parse_laws(datasets(), CORPUS)
    assert source.update_date == date(2026, 9, 18)
    assert [a.article_key for a in source.articles] == [
        "I0030047:5",
        "I0030047:5-1",
        "G0380252:3",
        "G0400162:1",
        "G0400162:10",
    ]
    assert source.warnings == []


def test_parse_laws_reads_law_metadata():
    article = parse_laws(datasets(), CORPUS).articles[2]
    assert article.law_name == "金融機構防制洗錢辦法"
    assert article.law_level == "命令"
    assert article.law_modified_date == date(2021, 12, 14)


def test_chapter_and_section_are_combined():
    articles = parse_laws(datasets(), CORPUS).articles
    assert articles[0].chapter is None
    assert articles[3].chapter == "第 一 章 總則"
    assert articles[4].chapter == "第 三 章 管理 第 一 節 虛擬資產服務商"


def test_missing_pcode_raises():
    with pytest.raises(SourceError, match="G9999999"):
        parse_laws(datasets(), [*CORPUS, CorpusLaw(pcode="G9999999", name="不存在")])


def test_renamed_law_warns():
    source = parse_laws(datasets(), [CorpusLaw(pcode="I0030047", name="舊名稱")])
    assert "舊名稱" in source.warnings[0]


def test_empty_article_is_skipped_with_warning():
    data = datasets()
    data[0]["Laws"][0]["LawArticles"].append(
        {"ArticleType": "A", "ArticleNo": "第 6 條", "ArticleContent": "  "}
    )
    source = parse_laws(data, [CORPUS[0]])
    assert [a.article_no for a in source.articles] == ["5", "5-1"]
    assert "第 6 條" in source.warnings[0]


def test_load_corpus(tmp_path):
    path = tmp_path / "corpus.toml"
    path.write_text('[[laws]]\npcode = "G0380131"\nname = "洗錢防制法"\n', encoding="utf-8")
    assert load_corpus(path) == [CorpusLaw(pcode="G0380131", name="洗錢防制法")]


def test_read_dataset_handles_bom(tmp_path):
    path = tmp_path / "ChLaw.json.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ChLaw.json", "﻿" + json.dumps({"Laws": []}))
    assert read_dataset(path) == {"Laws": []}


def client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.json", "{}")
    return buffer.getvalue()


def write_cache(raw_dir: Path) -> None:
    for name in ("ChLaw", "ChOrder"):
        (raw_dir / f"{name}.json.zip").write_bytes(b"cached")


def test_fetch_downloads_missing_datasets(tmp_path):
    requested = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        return httpx.Response(200, content=zip_bytes())

    result = fetch_datasets(tmp_path, client=client(handler))
    assert requested == ["/api/Ch/Law/json", "/api/Ch/Order/json"]
    assert [path.name for path in result.paths] == ["ChLaw.json.zip", "ChOrder.json.zip"]
    assert (tmp_path / "ChLaw.json.zip").read_bytes() == zip_bytes()


def test_fetch_uses_cache_without_refresh(tmp_path):
    write_cache(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("有快取時不應下載")

    assert fetch_datasets(tmp_path, client=client(handler)).warnings == []


def test_fetch_falls_back_to_cache_on_failure(tmp_path):
    write_cache(tmp_path)
    result = fetch_datasets(
        tmp_path, refresh=True, client=client(lambda request: httpx.Response(503))
    )
    assert len(result.warnings) == 2
    assert "快取" in result.warnings[0]
    assert (tmp_path / "ChLaw.json.zip").read_bytes() == b"cached"


def test_fetch_without_cache_raises(tmp_path):
    with pytest.raises(SourceError, match="沒有快取"):
        fetch_datasets(tmp_path, client=client(lambda request: httpx.Response(503)))


def test_fetch_keeps_cache_when_download_is_not_a_zip(tmp_path):
    write_cache(tmp_path)
    result = fetch_datasets(
        tmp_path,
        refresh=True,
        client=client(lambda request: httpx.Response(200, content=b"<html>")),
    )
    assert len(result.warnings) == 2
    assert "快取" in result.warnings[0]
    assert (tmp_path / "ChLaw.json.zip").read_bytes() == b"cached"
    assert not (tmp_path / "ChLaw.json.part").exists()


def test_fetch_non_zip_without_cache_raises(tmp_path):
    with pytest.raises(SourceError, match="不是 zip"):
        fetch_datasets(
            tmp_path, client=client(lambda request: httpx.Response(200, content=b"<html>"))
        )
    assert not (tmp_path / "ChLaw.json.zip").exists()


def test_read_dataset_rejects_invalid_zip(tmp_path):
    path = tmp_path / "ChLaw.json.zip"
    path.write_bytes(b"<html>")
    with pytest.raises(SourceError, match="--refresh"):
        read_dataset(path)


def test_read_dataset_rejects_empty_archive(tmp_path):
    path = tmp_path / "ChLaw.json.zip"
    with zipfile.ZipFile(path, "w"):
        pass
    with pytest.raises(SourceError, match="--refresh"):
        read_dataset(path)


def law_with_headings(*headings: str) -> list[dict]:
    articles = [{"ArticleType": "C", "ArticleContent": text} for text in headings]
    articles.append({"ArticleType": "A", "ArticleNo": "第 1 條", "ArticleContent": "內容"})
    law = {
        "LawLevel": "法律",
        "LawName": "測試法",
        "LawURL": "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=T0000001",
        "LawModifiedDate": "20200101",
        "LawArticles": articles,
    }
    return [{"UpdateDate": "2026/9/18 上午 12:00:00", "Laws": [law]}]


def chapter_of(*headings: str) -> str | None:
    corpus = [CorpusLaw(pcode="T0000001", name="測試法")]
    return parse_laws(law_with_headings(*headings), corpus).articles[0].chapter


def test_kuan_heading_is_a_section_under_the_chapter():
    assert chapter_of("第 一 章 總則", "第 一 款 定義") == "第 一 章 總則 第 一 款 定義"


def test_chapter_heading_starting_with_jie_is_not_a_section():
    assert chapter_of("第 一 章 總則", "第一章節約 x") == "第一章節約 x"


def test_unrecognized_heading_leaves_state_unchanged():
    assert chapter_of("第 一 章 總則", "附則說明") == "第 一 章 總則"
