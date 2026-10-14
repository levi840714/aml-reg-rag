"""需要 DATABASE_URL 指向含 pgvector 的 Postgres。"""

from datetime import date

import pytest
from dotenv import load_dotenv
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import text

from aml_reg_rag.config import PROJECT_ROOT, Settings
from aml_reg_rag.ingest.chunker import chunk_article
from aml_reg_rag.models import ChunkRef
from aml_reg_rag.retrieval.bm25 import Bm25Index
from aml_reg_rag.retrieval.index import (
    Database,
    IndexNotReadyError,
    indexed_article_keys,
    is_local,
    load_chunks,
    load_refs,
    open_index,
    rebuild_index,
    refs_table,
    write_refs,
)
from tests.fakes import article

pytestmark = pytest.mark.integration


def test_rebuild_and_search_with_real_postgres():
    load_dotenv(PROJECT_ROOT / ".env")
    settings = Settings(embedding_model="test:deterministic", embedding_dim=8)
    if not is_local(settings.database_url):
        pytest.skip("DATABASE_URL 不是本機資料庫；此測試會刪表重建，只在本機執行")
    embeddings = DeterministicFakeEmbedding(size=8)
    chunks = [
        *chunk_article(article("虛擬資產服務商應確認客戶身分。"), 600),
        *chunk_article(
            article(
                "金融機構應確認客戶身分。",
                pcode="G0380252",
                article_no="3",
                law_name="金融機構防制洗錢辦法",
            ),
            600,
        ),
    ]
    db = Database.connect(settings.database_url)
    try:
        meta = rebuild_index(db, settings, embeddings, chunks, date(2026, 9, 18))
        assert meta.table_name == "chunks_test_deterministic_d8"
        assert meta.chunk_count == 2
        index, opened, warnings = open_index(db, settings, embeddings)
        assert opened.table_name == meta.table_name
        assert warnings == []
        results = index.search(chunks[1].page_content, 2)
        assert results[0].chunk_id == "G0380252:3"
        assert results[0].score == pytest.approx(1.0)
        assert indexed_article_keys(db, meta.table_name) == {"G0380268:9", "G0380252:3"}
        entries = load_chunks(db, meta.table_name)
        assert [chunk.chunk_id for chunk, _ in entries] == ["G0380252:3", "G0380268:9"]
        assert entries[1][1].startswith("【提供虛擬資產服務之事業或人員防制洗錢及打擊資恐辦法")
        # 兩段的語料 IDF 會退化為 0，排名由單元測試以 fixture 驗證，這裡只確認能由資料表建立
        assert Bm25Index(entries).search("虛擬資產服務商", 1) == []
    finally:
        db.close()


def test_write_and_load_refs_with_real_postgres():
    load_dotenv(PROJECT_ROOT / ".env")
    settings = Settings()
    if not is_local(settings.database_url):
        pytest.skip("DATABASE_URL 不是本機資料庫；此測試會刪表重建，只在本機執行")
    table = "chunks_test_refs_d8"
    refs = [
        ChunkRef(
            src_chunk_id="G0400162:28", target_chunk_id="G0400162:16", kind="apply", raw="第十六條"
        ),
        ChunkRef(
            src_chunk_id="I0030047:12", target_chunk_id="I0030047:7", kind="penalty", raw="第七條"
        ),
    ]
    db = Database.connect(settings.database_url)
    try:
        with db.sql.begin() as connection:
            connection.execute(text(f'DROP TABLE IF EXISTS "{refs_table(table)}"'))
        with pytest.raises(IndexNotReadyError, match="reg ingest"):
            load_refs(db, table)
        write_refs(db, table, refs)
        write_refs(db, table, refs)
        assert load_refs(db, table) == refs
    finally:
        with db.sql.begin() as connection:
            connection.execute(text(f'DROP TABLE IF EXISTS "{refs_table(table)}"'))
        db.close()
