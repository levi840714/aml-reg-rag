import json
from pathlib import Path

from aml_reg_rag.ingest.chunker import chunk_article
from aml_reg_rag.ingest.source import CorpusLaw, parse_laws
from aml_reg_rag.retrieval.bm25 import Bm25Index
from tests.fakes import retrieved

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS = [
    CorpusLaw(pcode="I0030047", name="資恐防制法"),
    CorpusLaw(pcode="G0380252", name="金融機構防制洗錢辦法"),
    CorpusLaw(pcode="G0400162", name="提供虛擬資產服務之事業或人員洗錢防制登記辦法"),
]


def fixture_index() -> Bm25Index:
    data = [
        json.loads((FIXTURES / f"{name}.sample.json").read_text(encoding="utf-8"))
        for name in ("ChLaw", "ChOrder")
    ]
    chunks = [
        chunk
        for article in parse_laws(data, CORPUS).articles
        for chunk in chunk_article(article, 600)
    ]
    return Bm25Index([(retrieved(c.chunk_id, c.body), c.page_content) for c in chunks])


def test_domain_term_ranks_its_chunk_first():
    results = fixture_index().search("制裁名單陳述意見", 3)
    assert results[0][0].chunk_id == "I0030047:5-1"
    assert results[0][1] > 0


def test_law_name_in_header_is_searchable():
    results = fixture_index().search("洗錢防制登記辦法", 10)
    assert {chunk.chunk_id for chunk, _ in results[:2]} == {"G0400162:1", "G0400162:10"}


def test_results_are_limited_and_exclude_zero_scores():
    index = fixture_index()
    assert len(index.search("制裁名單", 1)) == 1
    assert index.search("完全無關的英文 zzz", 5) == []
