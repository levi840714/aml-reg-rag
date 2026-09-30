from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable
from pydantic import BaseModel

# 暫時性錯誤（限流、逾時）的重試次數
LLM_MAX_RETRIES = 2


def structured_model(
    model: str, schema: type[BaseModel], *, timeout: float, include_raw: bool = False
) -> Runnable[Any, Any]:
    llm = init_chat_model(model, timeout=timeout, max_retries=LLM_MAX_RETRIES)
    return llm.with_structured_output(schema, include_raw=include_raw)


def token_usage(raw: Any) -> tuple[int, int]:
    """回傳 (input, output) token；Gemini 的 output 已含 thinking token。"""
    usage = getattr(raw, "usage_metadata", None) or {}
    return usage.get("input_tokens", 0), usage.get("output_tokens", 0)
