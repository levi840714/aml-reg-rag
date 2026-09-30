import pytest
from langchain_core.embeddings import Embeddings

from aml_reg_rag.retrieval.embeddings import (
    EMBED_ATTEMPTS,
    EmbeddingConfigError,
    ResilientEmbeddings,
    dimension_kwargs,
)


class FlakyEmbeddings(Embeddings):
    """前 failures 次呼叫丟出例外，之後以文字長度作為向量。"""

    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.batches: list[list[str]] = []
        self.queries = 0

    def _maybe_fail(self) -> None:
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("429")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._maybe_fail()
        self.batches.append(texts)
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries += 1
        self._maybe_fail()
        return [float(len(text))]


def resilient(inner: Embeddings, sleeps: list[float]) -> ResilientEmbeddings:
    return ResilientEmbeddings(inner, sleep=sleeps.append)


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


def test_embed_documents_splits_into_batches_in_order():
    inner = FlakyEmbeddings()
    texts = ["x" * n for n in range(1, 46)]
    vectors = resilient(inner, []).embed_documents(texts)
    assert [len(batch) for batch in inner.batches] == [20, 20, 5]
    assert vectors == [[float(n)] for n in range(1, 46)]


def test_embed_documents_retries_then_succeeds():
    sleeps: list[float] = []
    inner = FlakyEmbeddings(failures=2)
    assert resilient(inner, sleeps).embed_documents(["ab"]) == [[2.0]]
    assert sleeps == [1, 2]


def test_embed_query_retries_then_succeeds():
    sleeps: list[float] = []
    inner = FlakyEmbeddings(failures=1)
    assert resilient(inner, sleeps).embed_query("abc") == [3.0]
    assert sleeps == [1]


def test_retry_exhaustion_reraises_last_error():
    sleeps: list[float] = []
    inner = FlakyEmbeddings(failures=EMBED_ATTEMPTS)
    with pytest.raises(RuntimeError, match="429"):
        resilient(inner, sleeps).embed_query("abc")
    assert inner.queries == EMBED_ATTEMPTS
    assert len(sleeps) == EMBED_ATTEMPTS - 1
