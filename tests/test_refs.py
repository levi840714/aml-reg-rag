from aml_reg_rag.ingest.chunker import chunk_article
from aml_reg_rag.ingest.refs import Reference, ReferenceParser, cn_number, link_chunks
from aml_reg_rag.models import ChunkRef, LawArticle
from tests.fakes import article

AML_ACT = "洗錢防制法"
CFT_ACT = "資恐防制法"
REGISTRATION = "提供虛擬資產服務之事業或人員洗錢防制登記辦法"
FRAUD = "金融機構及提供虛擬資產服務之事業或人員防制詐欺犯罪危害應遵循事項辦法"


def aml(article_no: str, content: str = "內容。") -> LawArticle:
    return article(content, pcode="G0380131", article_no=article_no, law_name=AML_ACT)


def cft(article_no: str, content: str = "內容。") -> LawArticle:
    return article(content, pcode="I0030047", article_no=article_no, law_name=CFT_ACT)


def registration(article_no: str, content: str = "內容。") -> LawArticle:
    return article(content, pcode="G0400162", article_no=article_no, law_name=REGISTRATION)


def fraud(article_no: str, content: str = "內容。") -> LawArticle:
    return article(content, pcode="G0380271", article_no=article_no, law_name=FRAUD)


def targets(refs: list[Reference]) -> list[tuple[str, tuple[int | None, ...]]]:
    return [(f"{ref.pcode}:{ref.article_no}", ref.target) for ref in refs]


def parse(citing: LawArticle, *others: LawArticle) -> list[Reference]:
    return ReferenceParser([citing, *others]).parse(citing)


def test_cn_number():
    numbers = ["一", "十", "十五", "二十", "一百十三", "二百零一", "三百六十二"]
    assert [cn_number(text) for text in numbers] == [1, 10, 15, 20, 113, 201, 362]


def test_previous_paragraph():
    refs = parse(aml("9", "甲。\n前項之乙。"))
    assert targets(refs) == [("G0380131:9", (1, None, None))]
    assert refs[0].source == (2, None, None)
    assert refs[0].raw == "前項"


def test_previous_n_paragraphs():
    refs = parse(aml("9", "甲。\n乙。\n前二項規定。"))
    assert targets(refs) == [("G0380131:9", (1, None, None)), ("G0380131:9", (2, None, None))]


def test_previous_paragraph_subparagraph_followed_by_definition():
    refs = parse(aml("9", "甲：\n一、子。\n前項第一款所稱虛擬資產服務商。"))
    assert targets(refs) == [("G0380131:9", (1, 1, None))]
    assert refs[0].kind == "definition"


def test_bare_subparagraph_and_previous_subparagraph_stay_in_current_paragraph():
    refs = parse(aml("9", "甲。\n乙如下：\n一、子。\n二、第一款以外者。\n三、前款以外者。"))
    assert targets(refs) == [("G0380131:9", (2, 1, None)), ("G0380131:9", (2, 2, None))]
    assert [ref.source for ref in refs] == [(2, 2, None), (2, 3, None)]


def test_previous_article_with_chained_paragraphs():
    refs = parse(cft("7", "除前條第一項、第二項所列措施外，不得為下列行為。"), cft("6"))
    assert targets(refs) == [("I0030047:6", (1, None, None)), ("I0030047:6", (2, None, None))]


def test_named_law_and_same_law_chain_through_modifiers():
    citing = article(
        "本辦法依洗錢防制法第五條第二項適用同法第七條第三項、第八條第四項前段、"
        "第十條第三項及資恐防制法第七條第五項規定訂定之。",
        article_no="1",
    )
    others = [aml("5"), aml("7"), aml("8"), aml("10"), cft("7"), article("內容。", article_no="10")]
    refs = parse(citing, *others)
    assert targets(refs) == [
        ("G0380131:5", (2, None, None)),
        ("G0380131:7", (3, None, None)),
        ("G0380131:8", (4, None, None)),
        ("G0380131:10", (3, None, None)),
        ("I0030047:7", (5, None, None)),
    ]
    assert {ref.kind for ref in refs} == {"authority"}


def test_unchained_bare_article_returns_to_own_law():
    citing = cft(
        "12",
        "洗錢防制法第五條第一項至第三項所定之機構、事業或人員違反第七條第一項至第三項規定者，"
        "由中央目的事業主管機關處新臺幣二十萬元以上一百萬元以下罰鍰。",
    )
    refs = parse(citing, aml("5"), aml("7"), cft("7"))
    assert [(target, ref.kind) for target, ref in zip(targets(refs), refs, strict=True)] == [
        (("G0380131:5", (1, None, None)), "definition"),
        (("G0380131:5", (2, None, None)), "definition"),
        (("G0380131:5", (3, None, None)), "definition"),
        (("I0030047:7", (1, None, None)), "penalty"),
        (("I0030047:7", (2, None, None)), "penalty"),
        (("I0030047:7", (3, None, None)), "penalty"),
    ]


def test_article_range_follows_article_order():
    citing = aml("28", "提供第十二條至第十四條受理申報之資料。")
    others = [aml("12"), aml("13"), aml("13-1"), aml("14")]
    assert [key for key, _ in targets(parse(citing, *others))] == [
        "G0380131:12",
        "G0380131:13",
        "G0380131:13-1",
        "G0380131:14",
    ]


def test_external_law_citations_are_dropped_for_the_rest_of_the_line():
    citing = aml(
        "3",
        "本法所稱特定犯罪，指下列各款之罪：\n"
        "一、刑法第一百二十一條、第二百零一條之一第二項及該二項之未遂犯、第五條之罪。\n"
        "二、本法第二十一條之罪。",
    )
    assert targets(parse(citing, aml("5"), aml("21"))) == [("G0380131:21", (None, None, None))]


def test_this_act_alias_points_to_parent_act():
    first = registration("1", "本辦法依洗錢防制法（以下簡稱本法）第六條第二項規定訂定之。")
    citing = registration("2", "本法第五條第一項所定金融機構，依本辦法第三條辦理。")
    refs = parse(citing, first, aml("5"), aml("6"), registration("3"), registration("5"))
    assert targets(refs) == [
        ("G0380131:5", (1, None, None)),
        ("G0400162:3", (None, None, None)),
    ]
    assert targets(parse(first, aml("6"), registration("6"))) == [("G0380131:6", (2, None, None))]


def test_alias_to_external_law_is_dropped():
    first = fraud("1", "本辦法依詐欺犯罪危害防制條例（以下簡稱本條例）第八條訂定之。")
    citing = fraud("30", "依本條例第八條第一項後段規定，強化確認客戶身分。")
    assert parse(citing, first, fraud("8")) == []


def test_mutatis_mutandis_kind():
    citing = registration("28", "應辦理公告。\n第十六條第二項及第三項規定，於前項情形準用之。")
    refs = parse(citing, registration("16"))
    assert [(target, ref.kind) for target, ref in zip(targets(refs), refs, strict=True)] == [
        (("G0400162:16", (2, None, None)), "apply"),
        (("G0400162:16", (3, None, None)), "apply"),
        (("G0400162:28", (1, None, None)), "general"),
    ]


def test_same_article_and_this_article():
    citing = aml("9", "依第六條第一項規定，並依同條第四款辦理。\n本條第一項之規定。")
    assert targets(parse(citing, aml("6"))) == [
        ("G0380131:6", (1, None, None)),
        ("G0380131:6", (1, 4, None)),
        ("G0380131:9", (1, None, None)),
    ]


def test_subparagraph_of_article_without_paragraph_defaults_to_first_paragraph():
    refs = parse(aml("9", "採取第六條第一款第二目之措施。"), aml("6"))
    assert targets(refs) == [("G0380131:6", (1, 1, 2))]


def test_link_chunks_maps_to_covering_chunks_and_drops_self_links():
    content = "\n".join(
        ["甲" * 20 + "：", "一、子。", "前項第一款所稱乙" + "乙" * 10, "前項規定。"]
    )
    chunks = chunk_article(aml("9", content), max_chars=40)
    assert [chunk.chunk_id for chunk in chunks] == ["G0380131:9:p1", "G0380131:9:p2-3"]
    assert link_chunks(chunks) == [
        ChunkRef(
            src_chunk_id="G0380131:9:p2-3",
            target_chunk_id="G0380131:9:p1",
            kind="definition",
            raw="前項第一款",
        )
    ]


def test_link_chunks_resolves_to_subparagraph_chunk():
    target = aml("6", "\n".join(["措施如下：", "一、" + "甲" * 20, "二、" + "乙" * 15, "三、短。"]))
    target_chunks = chunk_article(target, max_chars=30)
    citing_chunks = chunk_article(aml("9", "依第六條第三款及第十條辦理。"), max_chars=600)
    refs = link_chunks([*target_chunks, *citing_chunks])
    assert [chunk.chunk_id for chunk in target_chunks] == [
        "G0380131:6:p1:k1",
        "G0380131:6:p1:k2-3",
    ]
    assert [(ref.src_chunk_id, ref.target_chunk_id) for ref in refs] == [
        ("G0380131:9", "G0380131:6:p1:k2-3")
    ]


def test_same_statute_refers_to_last_named_law_even_without_article():
    citing = aml(
        "14",
        "依臺灣地區與大陸地區人民關係條例相關規定辦理，並依第五條第一項規定，"
        "總價值超過同條例第三十八條第五項所定限額者。",
    )
    assert targets(parse(citing, aml("5"), aml("38"))) == [("G0380131:5", (1, None, None))]


def test_apply_kind_when_mutatis_mutandis_precedes_citation():
    citing = aml(
        "18",
        "甲。\n乙。\n丙。\n丁。\n"
        "依第二十八條所簽訂之條約請求協助之案件，如所涉之犯罪行為符合第三條所列之罪，"
        "亦得準用前四項規定。",
    )
    refs = parse(citing, aml("3"), aml("28"))
    kinds = {(f"{ref.pcode}:{ref.article_no}", ref.kind) for ref in refs}
    assert kinds == {
        ("G0380131:28", "general"),
        ("G0380131:3", "definition"),
        ("G0380131:18", "apply"),
    }
