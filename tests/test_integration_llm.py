"""需要 GOOGLE_API_KEY、DATABASE_URL，且已執行 reg ingest。"""

import os

import pytest
from dotenv import load_dotenv

from aml_reg_rag.config import PROJECT_ROOT, Settings
from aml_reg_rag.graph.build import build_graph, make_drafter, run_ask
from aml_reg_rag.retrieval.embeddings import make_embeddings
from aml_reg_rag.retrieval.index import Database, open_index
from aml_reg_rag.retrieval.pipeline import RetrievalOptions, make_pipeline
from aml_reg_rag.retrieval.rerank import make_reranker
from aml_reg_rag.retrieval.rewrite import make_rewriter
from tests.fakes import retrieved

pytestmark = pytest.mark.integration


@pytest.fixture
def real_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("需要 GOOGLE_API_KEY")
    return Settings()


def test_ask_with_full_pipeline(real_settings):
    options = RetrievalOptions.from_settings(real_settings, hybrid=True, rewrite=True, rerank=True)
    db = Database.connect(real_settings.database_url)
    try:
        index, meta, _ = open_index(db, real_settings, make_embeddings(real_settings))
        retriever = make_pipeline(db, meta.table_name, index, real_settings, options)
        drafter = make_drafter(real_settings.llm_model, real_settings.llm_timeout)
        state = run_ask(build_graph(retriever, drafter), "虛擬資產服務商的交易紀錄要保存多久？", 6)
    finally:
        db.close()
    assert state["retrieval"] is not None
    assert state["retrieval"].rewrite_fallback is None
    assert state["retrieval"].rerank_fallback is None
    assert state["answer"] is not None
    assert state["verified"], state["problems"]


def test_rewrite_splits_cross_law_question(real_settings):
    rewriter = make_rewriter(real_settings.llm_light_model, real_settings.llm_timeout)
    question = "虛擬資產服務商沒有依資恐防制法通報制裁對象的資產，會被怎麼處罰？"
    result = rewriter.rewrite(question)
    assert result.fallback is None
    assert result.queries[0] == question
    assert 1 <= len(result.queries) <= 3


def test_rerank_prefers_applicable_subject(real_settings):
    reranker = make_reranker(real_settings.llm_light_model, real_settings.llm_timeout)
    bank = retrieved(
        "G0380252:3:p1:k1-4",
        "金融機構確認客戶身分措施，應依下列規定辦理：金融機構辦理一定金額以上交易時，應確認客戶身分。",
        law_name="金融機構防制洗錢辦法",
    )
    vasp = retrieved(
        "G0380268:3:p1:k1-4",
        "虛擬資產服務商確認客戶身分措施，應依下列規定辦理：辦理等值新臺幣三萬元以上之臨時性交易時，應確認客戶身分。",
    )
    result = reranker.score("虛擬資產服務商辦理臨時性交易，多少金額要確認客戶身分？", [bank, vasp])
    assert result.fallback is None
    assert result.scores is not None
    assert result.scores[vasp.chunk_id] > result.scores[bank.chunk_id]
