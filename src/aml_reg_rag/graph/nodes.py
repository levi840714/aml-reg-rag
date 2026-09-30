from typing import Any

from langchain_core.runnables import Runnable
from langgraph.graph import END

from aml_reg_rag import citations
from aml_reg_rag.graph import prompts
from aml_reg_rag.graph.state import AskState, labeled_chunks
from aml_reg_rag.llm import token_usage
from aml_reg_rag.models import Answer
from aml_reg_rag.retrieval.index import ChunkIndex

# 首次生成加上 1 次重試
MAX_ATTEMPTS = 2


class AskNodes:
    def __init__(self, index: ChunkIndex, drafter: Runnable[Any, Any]) -> None:
        self._index = index
        self._drafter = drafter

    def retrieve(self, state: AskState) -> dict[str, Any]:
        return {"chunks": self._index.search(state["question"], state["k"])}

    def generate(self, state: AskState) -> dict[str, Any]:
        messages = prompts.answer_messages(state["question"], labeled_chunks(state["chunks"]))
        if state["attempts"] > 0:
            messages.append(prompts.retry_message(state["answer"], state["problems"]))
        result = self._drafter.invoke(messages)
        input_tokens, output_tokens = token_usage(result["raw"])
        parsed = result["parsed"]
        return {
            "answer": parsed if isinstance(parsed, Answer) else None,
            "attempts": state["attempts"] + 1,
            "input_tokens": state["input_tokens"] + input_tokens,
            "output_tokens": state["output_tokens"] + output_tokens,
        }

    def verify(self, state: AskState) -> dict[str, Any]:
        bodies = {label: chunk.body for label, chunk in labeled_chunks(state["chunks"]).items()}
        problems = citations.verify(state["answer"], bodies)
        return {"problems": problems, "verified": not problems}


def route_after_verify(state: AskState) -> str:
    if state["verified"] or state["attempts"] >= MAX_ATTEMPTS:
        return END
    return "generate"
