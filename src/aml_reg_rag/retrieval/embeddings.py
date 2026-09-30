import time
from collections.abc import Callable
from typing import Any

from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings

from aml_reg_rag.config import Settings

# 各 provider 指定輸出維度的參數名稱
DIMENSION_PARAMS = {"google_genai": "output_dimensionality", "openai": "dimensions"}
# 套件估算 token 時低估中文，每批 100 段可能超過單次請求上限
EMBED_BATCH_SIZE = 20
EMBED_ATTEMPTS = 4


class EmbeddingConfigError(ValueError):
    pass


class ResilientEmbeddings(Embeddings):
    """分批送出並在失敗時以指數退避重試。"""

    def __init__(self, inner: Embeddings, sleep: Callable[[float], None] = time.sleep) -> None:
        self._inner = inner
        self._sleep = sleep

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            vectors.extend(self._retry(self._inner.embed_documents, batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._retry(self._inner.embed_query, text)

    def _retry[Arg, Result](self, call: Callable[[Arg], Result], arg: Arg) -> Result:
        for attempt in range(EMBED_ATTEMPTS - 1):
            try:
                return call(arg)
            except Exception:
                self._sleep(2**attempt)
        return call(arg)


def dimension_kwargs(model: str, dim: int) -> dict[str, int]:
    provider, _, name = model.partition(":")
    if not name:
        raise EmbeddingConfigError(f"EMBEDDING_MODEL 格式須為 <provider>:<model>：{model}")
    param = DIMENSION_PARAMS.get(provider)
    if param is None:
        supported = "、".join(DIMENSION_PARAMS)
        raise EmbeddingConfigError(f"{provider} 不支援指定 embedding 維度，目前支援：{supported}")
    return {param: dim}


def make_embeddings(settings: Settings) -> Embeddings:
    kwargs: dict[str, Any] = dimension_kwargs(settings.embedding_model, settings.embedding_dim)
    return ResilientEmbeddings(init_embeddings(settings.embedding_model, **kwargs))
