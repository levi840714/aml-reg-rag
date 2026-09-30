from dataclasses import dataclass

import httpx
from langchain_core.embeddings import Embeddings

from aml_reg_rag.config import Settings
from aml_reg_rag.ingest.chunker import chunk_article
from aml_reg_rag.ingest.source import (
    SourceData,
    fetch_datasets,
    load_corpus,
    parse_laws,
    read_dataset,
)
from aml_reg_rag.retrieval.index import Database, IndexMeta, rebuild_index


@dataclass(frozen=True)
class IngestReport:
    laws: int
    articles: int
    chunks: int
    oversized: int
    meta: IndexMeta
    warnings: list[str]


def load_source(
    settings: Settings, *, refresh: bool = False, client: httpx.Client | None = None
) -> SourceData:
    corpus = load_corpus(settings.data_dir / "corpus.toml")
    fetched = fetch_datasets(settings.var_dir / "raw", refresh=refresh, client=client)
    source = parse_laws([read_dataset(path) for path in fetched.paths], corpus)
    return SourceData(source.update_date, source.articles, [*fetched.warnings, *source.warnings])


def run_ingest(
    settings: Settings,
    db: Database,
    embeddings: Embeddings,
    *,
    refresh: bool = False,
    client: httpx.Client | None = None,
) -> IngestReport:
    source = load_source(settings, refresh=refresh, client=client)
    chunks = [
        chunk
        for article in source.articles
        for chunk in chunk_article(article, settings.chunk_max_chars)
    ]
    meta = rebuild_index(db, settings, embeddings, chunks, source.update_date)
    return IngestReport(
        laws=len({article.pcode for article in source.articles}),
        articles=len(source.articles),
        chunks=len(chunks),
        oversized=sum(len(chunk.body) > settings.chunk_max_chars for chunk in chunks),
        meta=meta,
        warnings=source.warnings,
    )
