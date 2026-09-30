"""需要 GOOGLE_API_KEY、DATABASE_URL，且已執行 reg ingest。"""

import os

import pytest
from dotenv import load_dotenv

from aml_reg_rag.config import PROJECT_ROOT, Settings
from aml_reg_rag.graph.build import build_graph, make_drafter, run_ask
from aml_reg_rag.retrieval.embeddings import make_embeddings
from aml_reg_rag.retrieval.index import Database, open_index

pytestmark = pytest.mark.integration


def test_ask_with_real_llm():
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("需要 GOOGLE_API_KEY")
    settings = Settings()
    db = Database.connect(settings.database_url)
    try:
        index, _, _ = open_index(db, settings, make_embeddings(settings))
        graph = build_graph(index, make_drafter(settings.llm_model, settings.llm_timeout))
        state = run_ask(graph, "虛擬資產服務商的交易紀錄要保存多久？", settings.top_k)
    finally:
        db.close()
    assert state["chunks"]
    assert state["answer"] is not None
    assert state["verified"], state["problems"]
