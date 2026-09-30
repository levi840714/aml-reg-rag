import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

from aml_reg_rag.ingest import pipeline
from aml_reg_rag.retrieval.index import IndexMeta

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS_TOML = """[[laws]]
pcode = "I0030047"
name = "資恐防制法"

[[laws]]
pcode = "G0380252"
name = "金融機構防制洗錢辦法"
"""


def prepare(settings) -> None:
    raw_dir = settings.var_dir / "raw"
    raw_dir.mkdir(parents=True)
    for name in ("ChLaw", "ChOrder"):
        data = (FIXTURES / f"{name}.sample.json").read_text(encoding="utf-8")
        with zipfile.ZipFile(raw_dir / f"{name}.json.zip", "w") as archive:
            archive.writestr(f"{name}.json", data)
    settings.data_dir.mkdir(parents=True)
    (settings.data_dir / "corpus.toml").write_text(CORPUS_TOML, encoding="utf-8")


def test_run_ingest_chunks_articles_and_rebuilds_index(settings, monkeypatch):
    prepare(settings)
    captured = {}

    def fake_rebuild(db, current, embeddings, chunks, update_date):
        captured["chunks"] = chunks
        return IndexMeta(
            table_name="chunks_test",
            embedding_model=current.embedding_model,
            embedding_dim=current.embedding_dim,
            chunk_max_chars=current.chunk_max_chars,
            source_update_date=update_date,
            chunk_count=len(chunks),
            ingested_at=datetime.now(UTC),
        )

    monkeypatch.setattr(pipeline, "rebuild_index", fake_rebuild)
    report = pipeline.run_ingest(settings, db=None, embeddings=None)
    assert (report.laws, report.articles, report.chunks, report.oversized) == (2, 3, 8, 0)
    assert report.meta.source_update_date == date(2026, 9, 18)
    assert [c.chunk_id for c in captured["chunks"][:3]] == [
        "I0030047:5",
        "I0030047:5-1",
        "G0380252:3:p1:k1-4",
    ]
