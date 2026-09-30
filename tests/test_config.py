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
