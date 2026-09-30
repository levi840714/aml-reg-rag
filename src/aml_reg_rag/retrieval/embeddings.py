from typing import Any

from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings

from aml_reg_rag.config import Settings

# 各 provider 指定輸出維度的參數名稱
DIMENSION_PARAMS = {"google_genai": "output_dimensionality", "openai": "dimensions"}


class EmbeddingConfigError(ValueError):
    pass


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
    return init_embeddings(settings.embedding_model, **kwargs)
