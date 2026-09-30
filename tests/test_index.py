from langchain_core.documents import Document

from aml_reg_rag.retrieval.index import MAX_IDENTIFIER, table_name, to_retrieved


def test_table_name_slugifies_model():
    assert table_name("google_genai:gemini-embedding-001", 1536) == (
        "chunks_google_genai_gemini_embedding_001_d1536"
    )


def test_long_table_name_is_truncated_with_hash():
    name = table_name("openai:" + "x" * 80, 1536)
    assert len(name) == MAX_IDENTIFIER
    assert name.startswith("chunks_openai_xxx")
    assert name != table_name("openai:" + "x" * 81, 1536)


def test_to_retrieved_strips_header_and_converts_distance():
    document = Document(
        id="G0380252:3:p1:k1-4",
        page_content="【金融機構防制洗錢辦法｜第 3 條 第 1 項第 1 款至第 4 款】\n第一行\n第二行",
        metadata={
            "pcode": "G0380252",
            "law_name": "金融機構防制洗錢辦法",
            "article_no": "3",
            "locator": "第 1 項第 1 款至第 4 款",
            "source_url": "https://law.moj.gov.tw/LawClass/LawSingle.aspx?pcode=G0380252&flno=3",
        },
    )
    chunk = to_retrieved(document, 0.25)
    assert chunk.chunk_id == "G0380252:3:p1:k1-4"
    assert chunk.article_key == "G0380252:3"
    assert chunk.body == "第一行\n第二行"
    assert chunk.score == 0.75
