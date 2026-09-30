import pytest

from aml_reg_rag.retrieval.embeddings import EmbeddingConfigError, dimension_kwargs


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("google_genai:gemini-embedding-001", {"output_dimensionality": 768}),
        ("openai:text-embedding-3-small", {"dimensions": 768}),
    ],
)
def test_dimension_kwargs_per_provider(model, expected):
    assert dimension_kwargs(model, 768) == expected


def test_unsupported_provider_raises():
    with pytest.raises(EmbeddingConfigError, match="ollama"):
        dimension_kwargs("ollama:bge-m3", 1024)


def test_model_without_provider_raises():
    with pytest.raises(EmbeddingConfigError, match="<provider>:<model>"):
        dimension_kwargs("gemini-embedding-001", 768)
