import json
from pathlib import Path

from aml_reg_rag.ingest.chunker import chunk_article, chunk_path, line_level, parse_units
from aml_reg_rag.ingest.source import CorpusLaw, parse_laws
from tests.fakes import article

FIXTURES = Path(__file__).parent / "fixtures"


def test_line_level():
    lines = ["本法所稱洗錢如下：", "一、隱匿", "（一）與客戶", "(二)進行", "1.辦理", "１．辦理"]
    assert [line_level(line) for line in lines] == [0, 1, 2, 2, 3, 3]


def test_parse_units_builds_hierarchy():
    units = parse_units("第一項：\r\n一、甲\r\n（一）子\r\n1.孫\r\n二、乙\r\n第二項。")
    assert [unit.head for unit in units] == ["第一項：", "第二項。"]
    assert [child.head for child in units[0].children] == ["一、甲", "二、乙"]
    assert units[0].children[0].children[0].children[0].head == "1.孫"


def test_short_article_is_one_chunk():
    chunks = chunk_article(article("甲項。\n乙項。"), max_chars=100)
    assert [(c.chunk_id, c.locator, c.body) for c in chunks] == [
        ("G0380268:9", "", "甲項。\n乙項。")
    ]


def test_long_article_merges_adjacent_paragraphs():
    content = "\n".join(["甲" * 10, "乙" * 10, "丙" * 10])
    chunks = chunk_article(article(content), max_chars=21)
    assert [(c.chunk_id, c.locator) for c in chunks] == [
        ("G0380268:9:p1-2", "第 1 項至第 2 項"),
        ("G0380268:9:p3", "第 3 項"),
    ]
    assert chunks[0].body == f"{'甲' * 10}\n{'乙' * 10}"


def test_long_paragraph_splits_by_subparagraph_with_lead_in():
    content = "\n".join(
        ["應辦理下列事項：", "一、" + "甲" * 10, "二、" + "乙" * 10, "三、" + "丙" * 10]
    )
    chunks = chunk_article(article(content), max_chars=35)
    assert [(c.chunk_id, c.locator) for c in chunks] == [
        ("G0380268:9:p1:k1-2", "第 1 項第 1 款至第 2 款"),
        ("G0380268:9:p1:k3", "第 1 項第 3 款"),
    ]
    assert all(c.body.startswith("應辦理下列事項：\n") for c in chunks)


def test_long_subparagraph_splits_by_item_with_both_lead_ins():
    content = "\n".join(
        ["措施如下：", "一、確認身分時：", "（一）" + "甲" * 10, "（二）" + "乙" * 10, "二、短款。"]
    )
    chunks = chunk_article(article(content), max_chars=30)
    assert [(c.chunk_id, c.locator) for c in chunks] == [
        ("G0380268:9:p1:k1:m1", "第 1 項第 1 款第 1 目"),
        ("G0380268:9:p1:k1:m2", "第 1 項第 1 款第 2 目"),
        ("G0380268:9:p1:k2", "第 1 項第 2 款"),
    ]
    assert chunks[1].body == f"措施如下：\n一、確認身分時：\n（二）{'乙' * 10}"


def test_sub_items_are_not_split_and_may_exceed_limit():
    content = "\n".join(["措施：", "一、款：", "（一）" + "甲" * 30, "1.子目一", "（二）短目。"])
    chunks = chunk_article(article(content), max_chars=30)
    assert [c.chunk_id for c in chunks] == ["G0380268:9:p1:k1:m1", "G0380268:9:p1:k1:m2"]
    assert "1.子目一" in chunks[0].body
    assert len(chunks[0].body) > 30


def test_real_article_splits_into_groups_under_limit():
    data = [
        json.loads((FIXTURES / f"{name}.sample.json").read_text(encoding="utf-8"))
        for name in ("ChLaw", "ChOrder")
    ]
    source = parse_laws(data, [CorpusLaw(pcode="G0380252", name="金融機構防制洗錢辦法")])
    article_three = source.articles[0]
    chunks = chunk_article(article_three, max_chars=600)
    assert [c.chunk_id for c in chunks] == [
        "G0380252:3:p1:k1-4",
        "G0380252:3:p1:k5-6",
        "G0380252:3:p1:k7:m1-2",
        "G0380252:3:p1:k7:m3-4",
        "G0380252:3:p1:k8-11",
        "G0380252:3:p1:k12-13",
    ]
    assert all(len(c.body) <= 600 for c in chunks)
    assert all(c.body.startswith("金融機構確認客戶身分措施，應依下列規定辦理：") for c in chunks)
    covered = "\n".join(c.body for c in chunks)
    lines = [line.strip() for line in article_three.content.splitlines() if line.strip()]
    assert all(line in covered for line in lines)


def test_chunk_path_reads_ranges_from_chunk_id():
    assert chunk_path("G0380268:9") == ()
    assert chunk_path("G0380268:9:p1-2:k3") == ((1, 2), (3, 3))
    assert chunk_path("G0380131:5-1:p2") == ((2, 2),)
