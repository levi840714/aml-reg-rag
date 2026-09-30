import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import Column, PGEngine, PGVectorStore
from langchain_postgres.v2.indexes import HNSWIndex
from pydantic import BaseModel
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import DBAPIError, OperationalError

from aml_reg_rag.config import Settings
from aml_reg_rag.models import Chunk, RetrievedChunk, article_url
from aml_reg_rag.retrieval.embeddings import EmbeddingConfigError

MAX_IDENTIFIER = 63
ID_COLUMN = "chunk_id"
METADATA_COLUMNS = [
    Column("pcode", "TEXT", nullable=False),
    Column("law_name", "TEXT", nullable=False),
    Column("law_level", "TEXT", nullable=False),
    Column("article_no", "TEXT", nullable=False),
    Column("chapter", "TEXT"),
    Column("locator", "TEXT", nullable=False),
    Column("law_modified_date", "DATE", nullable=False),
    Column("source_url", "TEXT", nullable=False),
]
META_DDL = """
CREATE TABLE IF NOT EXISTS index_meta (
    table_name TEXT PRIMARY KEY,
    embedding_model TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    chunk_max_chars INTEGER NOT NULL,
    source_update_date DATE NOT NULL,
    chunk_count INTEGER NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL
)
"""
INSERT_META = """
INSERT INTO index_meta VALUES (
    :table_name, :embedding_model, :embedding_dim, :chunk_max_chars,
    :source_update_date, :chunk_count, :ingested_at
)
"""


class DatabaseError(RuntimeError):
    pass


class IndexNotReadyError(RuntimeError):
    pass


class ChunkIndex(Protocol):
    def search(self, query: str, k: int) -> list[RetrievedChunk]: ...


class IndexMeta(BaseModel):
    table_name: str
    embedding_model: str
    embedding_dim: int
    chunk_max_chars: int
    source_update_date: date
    chunk_count: int
    ingested_at: datetime


@dataclass(frozen=True)
class Database:
    vector: PGEngine
    sql: Engine

    @classmethod
    def connect(cls, url: str) -> "Database":
        return cls(PGEngine.from_connection_string(url=url), create_engine(url))

    def close(self) -> None:
        self.sql.dispose()


class PgChunkIndex:
    def __init__(self, store: PGVectorStore) -> None:
        self._store = store

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        results = self._store.similarity_search_with_score(query, k=k)
        return [to_retrieved(document, distance) for document, distance in results]


def table_name(model: str, dim: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    name = f"chunks_{slug}_d{dim}"
    if len(name) <= MAX_IDENTIFIER:
        return name
    digest = hashlib.sha1(name.encode()).hexdigest()[:8]
    return f"{name[: MAX_IDENTIFIER - 9]}_{digest}"


def to_retrieved(document: Document, distance: float) -> RetrievedChunk:
    metadata = document.metadata
    _, _, body = document.page_content.partition("\n")
    return RetrievedChunk(
        chunk_id=str(document.id),
        article_key=f"{metadata['pcode']}:{metadata['article_no']}",
        law_name=metadata["law_name"],
        article_no=metadata["article_no"],
        locator=metadata["locator"],
        source_url=metadata["source_url"],
        body=body,
        score=1 - distance,
    )


def ensure_database(db: Database) -> None:
    try:
        with db.sql.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.execute(text(META_DDL))
    except OperationalError as exc:
        raise DatabaseError(
            "無法連線資料庫，請檢查 DATABASE_URL，或執行 docker compose up -d"
        ) from exc
    except DBAPIError as exc:
        raise DatabaseError(
            "無法啟用 pgvector 擴充，請在資料庫安裝 pgvector，或改用 docker compose 啟動"
        ) from exc


def read_meta(db: Database, table: str) -> IndexMeta | None:
    with db.sql.connect() as connection:
        row = (
            connection.execute(
                text("SELECT * FROM index_meta WHERE table_name = :table"), {"table": table}
            )
            .mappings()
            .first()
        )
    return IndexMeta.model_validate(dict(row)) if row else None


def rebuild_index(
    db: Database,
    settings: Settings,
    embeddings: Embeddings,
    chunks: list[Chunk],
    source_update_date: date,
) -> IndexMeta:
    ensure_database(db)
    actual_dim = len(embeddings.embed_query("維度檢查"))
    if actual_dim != settings.embedding_dim:
        raise EmbeddingConfigError(
            f"embedding 實際維度為 {actual_dim}，與 EMBEDDING_DIM={settings.embedding_dim} 不符"
        )
    table = table_name(settings.embedding_model, settings.embedding_dim)
    with db.sql.begin() as connection:
        connection.execute(
            text("DELETE FROM index_meta WHERE table_name = :table"), {"table": table}
        )
    db.vector.init_vectorstore_table(
        table,
        settings.embedding_dim,
        id_column=Column(ID_COLUMN, "TEXT", nullable=False),
        metadata_columns=[*METADATA_COLUMNS],
        store_metadata=False,
        overwrite_existing=True,
    )
    store = _open_store(db, table, embeddings)
    documents = [Document(page_content=c.page_content, metadata=_metadata(c)) for c in chunks]
    store.add_documents(documents, ids=[chunk.chunk_id for chunk in chunks])
    store.apply_vector_index(HNSWIndex())
    meta = IndexMeta(
        table_name=table,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
        chunk_max_chars=settings.chunk_max_chars,
        source_update_date=source_update_date,
        chunk_count=len(chunks),
        ingested_at=datetime.now(UTC),
    )
    with db.sql.begin() as connection:
        connection.execute(text(INSERT_META), meta.model_dump())
    return meta


def open_index(
    db: Database, settings: Settings, embeddings: Embeddings
) -> tuple[PgChunkIndex, IndexMeta, list[str]]:
    ensure_database(db)
    table = table_name(settings.embedding_model, settings.embedding_dim)
    meta = read_meta(db, table)
    if meta is None:
        raise IndexNotReadyError(
            f"找不到 {settings.embedding_model}（{settings.embedding_dim} 維）的索引，"
            "請先執行 reg ingest"
        )
    warnings = []
    if meta.chunk_max_chars != settings.chunk_max_chars:
        warnings.append(
            f"索引以 CHUNK_MAX_CHARS={meta.chunk_max_chars} 建立，與目前設定 "
            f"{settings.chunk_max_chars} 不同；需要時請重新執行 reg ingest"
        )
    return PgChunkIndex(_open_store(db, table, embeddings)), meta, warnings


def indexed_article_keys(db: Database, table: str) -> set[str]:
    with db.sql.connect() as connection:
        rows = connection.execute(text(f'SELECT DISTINCT pcode, article_no FROM "{table}"'))
        return {f"{pcode}:{article_no}" for pcode, article_no in rows}


def _open_store(db: Database, table: str, embeddings: Embeddings) -> PGVectorStore:
    return PGVectorStore.create_sync(
        engine=db.vector,
        embedding_service=embeddings,
        table_name=table,
        id_column=ID_COLUMN,
        metadata_columns=[column.name for column in METADATA_COLUMNS],
    )


def _metadata(chunk: Chunk) -> dict[str, Any]:
    article = chunk.article
    return {
        "pcode": article.pcode,
        "law_name": article.law_name,
        "law_level": article.law_level,
        "article_no": article.article_no,
        "chapter": article.chapter,
        "locator": chunk.locator,
        "law_modified_date": article.law_modified_date,
        "source_url": article_url(article.pcode, article.article_no),
    }
