from aml_reg_rag.models import Chunk, article_url
from tests.fakes import article, retrieved


def test_header_includes_law_chapter_and_locator():
    chunk = Chunk(
        chunk_id="G0380268:9:p1",
        article=article("內容", chapter="第 一 章 總則"),
        locator="第 1 項",
        body="內容",
    )
    assert chunk.header == (
        "【提供虛擬資產服務之事業或人員防制洗錢及打擊資恐辦法｜第 一 章 總則｜第 9 條 第 1 項】"
    )
    assert chunk.page_content == f"{chunk.header}\n內容"


def test_header_without_chapter_or_locator():
    chunk = Chunk(chunk_id="G0380268:9", article=article("內容"), locator="", body="內容")
    assert chunk.header == "【提供虛擬資產服務之事業或人員防制洗錢及打擊資恐辦法｜第 9 條】"


def test_article_key_and_url():
    assert article("內容", pcode="I0030047", article_no="5-1").article_key == "I0030047:5-1"
    assert article_url("I0030047", "5-1") == (
        "https://law.moj.gov.tw/LawClass/LawSingle.aspx?pcode=I0030047&flno=5-1"
    )


def test_retrieved_title():
    chunk = retrieved(
        "G0380252:3:p1:k2-4",
        "內文",
        law_name="金融機構防制洗錢辦法",
        locator="第 1 項第 2 款至第 4 款",
    )
    assert chunk.title == "金融機構防制洗錢辦法 第 3 條第 1 項第 2 款至第 4 款"
