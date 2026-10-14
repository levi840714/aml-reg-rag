import pytest
from pydantic import ValidationError

from aml_reg_rag.config import Settings


def test_defaults_follow_spec():
    settings = Settings(_env_file=None)
    assert settings.llm_model == "google_genai:gemini-3.8-flash"
    assert settings.llm_light_model == "google_genai:gemini-3.5-flash-lite"
    assert settings.eval_judge_model == "google_genai:gemini-3.8-flash"
    assert settings.embedding_model == "google_genai:gemini-embedding-001"
    assert (settings.embedding_dim, settings.top_k, settings.chunk_max_chars) == (1536, 6, 600)


def test_embedding_dim_above_hnsw_limit_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, embedding_dim=3072)


def test_judge_model_does_not_follow_llm_model(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "google_genai:gemini-3.5-flash-lite")
    settings = Settings(_env_file=None)
    assert settings.llm_model == "google_genai:gemini-3.5-flash-lite"
    assert settings.eval_judge_model == "google_genai:gemini-3.8-flash"


def test_non_positive_llm_timeout_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_timeout=0)


def test_retrieval_defaults_follow_ablation():
    # v2 ablation：rewrite 沒有提升檢索且每題多約 1.1 秒，預設關閉
    settings = Settings(_env_file=None)
    assert (settings.retrieval_hybrid, settings.retrieval_rewrite, settings.retrieval_rerank) == (
        True,
        False,
        True,
    )
    assert settings.retrieval_candidates == 20
    # 引用展開：上限 5 補齊 cross_ref 全部預期條文，平均只比上限 3 多 0.06 條
    assert (settings.retrieval_expand_refs, settings.retrieval_expand_limit) == (False, 5)


def test_candidates_must_cover_top_k():
    with pytest.raises(ValidationError, match="RETRIEVAL_CANDIDATES"):
        Settings(_env_file=None, top_k=10, retrieval_candidates=5)
