from datetime import date
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from aml_reg_rag.config import Settings
from aml_reg_rag.ingest.chunker import chunk_article
from aml_reg_rag.retrieval.embeddings import EmbeddingConfigError
from aml_reg_rag.retrieval.index import (
    MAX_IDENTIFIER,
    rebuild_index,
    table_name,
    to_retrieved,
)
from tests.fakes import article

INDEX_NAME_SUFFIX = "_hnsw"


class UntouchableDb:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"不應在 embedding 完成前存取資料庫：{name}")


class StubEmbeddings(Embeddings):
    def __init__(self, vectors: list[list[float]] | Exception) -> None:
        self._vectors = vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if isinstance(self._vectors, Exception):
            raise self._vectors
        return self._vectors

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


def test_table_name_slugifies_model():
    assert table_name("google_genai:gemini-embedding-001", 1536) == (
        "chunks_google_genai_gemini_embedding_001_d1536"
    )


def test_long_table_name_is_truncated_with_hash():
    name = table_name("openai:" + "x" * 80, 1536)
    assert len(name) + len(INDEX_NAME_SUFFIX) <= MAX_IDENTIFIER
    assert name.startswith("chunks_openai_xxx")
    assert name != table_name("openai:" + "x" * 81, 1536)


def test_default_model_table_name_is_not_truncated():
    name = table_name("google_genai:gemini-embedding-001", 1536)
    assert name == "chunks_google_genai_gemini_embedding_001_d1536"


def _rebuild_inputs() -> tuple[Settings, list[Any]]:
    settings = Settings(_env_file=None, embedding_dim=8)
    return settings, chunk_article(article("虛擬資產服務商應確認客戶身分。"), 600)


def test_rebuild_propagates_embedding_failure_without_touching_db():
    settings, chunks = _rebuild_inputs()
    embeddings = StubEmbeddings(RuntimeError("quota"))
    with pytest.raises(RuntimeError, match="quota"):
        rebuild_index(
            UntouchableDb(),  # type: ignore[arg-type]
            settings,
            embeddings,
            chunks,
            date(2026, 9, 18),
        )


def test_rebuild_rejects_wrong_dimension_without_touching_db():
    settings, chunks = _rebuild_inputs()
    embeddings = StubEmbeddings([[0.0] * 4])
    with pytest.raises(EmbeddingConfigError):
        rebuild_index(
            UntouchableDb(),  # type: ignore[arg-type]
            settings,
            embeddings,
            chunks,
            date(2026, 9, 18),
        )


def test_rebuild_rejects_empty_chunks_without_touching_db():
    settings, _ = _rebuild_inputs()
    with pytest.raises(ValueError, match="沒有可寫入的段落"):
        rebuild_index(
            UntouchableDb(),  # type: ignore[arg-type]
            settings,
            StubEmbeddings([]),
            [],
            date(2026, 9, 18),
        )


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
