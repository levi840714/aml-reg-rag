from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable
from pydantic import BaseModel

# 暫時性錯誤（限流、逾時）的重試設定：Google 視為總嘗試次數，OpenAI / Anthropic 視為重試次數
LLM_MAX_RETRIES = 3


def structured_model(
    model: str, schema: type[BaseModel], *, timeout: float, include_raw: bool = False
) -> Runnable[Any, Any]:
    llm = init_chat_model(model, timeout=timeout, max_retries=LLM_MAX_RETRIES)
    return llm.with_structured_output(schema, include_raw=include_raw)


def token_usage(raw: Any) -> tuple[int, int]:
    """回傳 (input, output) token；Gemini 的 output 已含 thinking token。"""
    usage = getattr(raw, "usage_metadata", None) or {}
    return usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def describe_error(exc: Exception) -> str:
    """只取例外類型與整數狀態碼；外部 API 的錯誤訊息可能含請求細節。"""
    status = next(
        (v for n in ("status_code", "code") if isinstance(v := getattr(exc, n, None), int)),
        None,
    )
    name = type(exc).__name__
    return f"{name}（狀態碼 {status}）" if status is not None else name
