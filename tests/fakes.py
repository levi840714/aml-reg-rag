from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda

from aml_reg_rag.models import (
    Answer,
    ChunkTrace,
    Citation,
    LawArticle,
    Point,
    Retrieval,
    RetrievedChunk,
    article_url,
)

VASP_AML = "提供虛擬資產服務之事業或人員防制洗錢及打擊資恐辦法"
USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}


def article(
    content: str,
    *,
    pcode: str = "G0380268",
    article_no: str = "9",
    law_name: str = VASP_AML,
    chapter: str | None = None,
) -> LawArticle:
    return LawArticle(
        pcode=pcode,
        law_name=law_name,
        law_level="命令",
        law_modified_date=date(2024, 11, 26),
        chapter=chapter,
        article_no=article_no,
        content=content,
    )


def retrieved(
    chunk_id: str,
    body: str,
    *,
    law_name: str = VASP_AML,
    locator: str = "",
    score: float = 0.9,
) -> RetrievedChunk:
    pcode, article_no = chunk_id.split(":")[:2]
    return RetrievedChunk(
        chunk_id=chunk_id,
        article_key=f"{pcode}:{article_no}",
        law_name=law_name,
        article_no=article_no,
        locator=locator,
        source_url=article_url(pcode, article_no),
        body=body,
        score=score,
    )


def cite(source: str, quote: str) -> Citation:
    return Citation(source=source, quote=quote)


def point(statement: str, *citations: Citation) -> Point:
    return Point(statement=statement, citations=list(citations))


def answer(*points: Point) -> Answer:
    return Answer(answerable=True, points=list(points))


def refusal(missing: str = "段落沒有相關規範") -> Answer:
    return Answer(answerable=False, points=[], missing=missing)


class FakeIndex:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        self.queries.append((query, k))
        return self.chunks[:k]


class FakeRetriever:
    """回傳固定段落的 Retriever，並記錄每次查詢。"""

    def __init__(
        self, chunks: list[RetrievedChunk], *, input_tokens: int = 0, output_tokens: int = 0
    ) -> None:
        self.chunks = chunks
        self.queries: list[tuple[str, int]] = []
        self._tokens = (input_tokens, output_tokens)

    def retrieve(self, question: str, k: int) -> Retrieval:
        self.queries.append((question, k))
        chunks = self.chunks[:k]
        traces = [
            ChunkTrace(chunk_id=c.chunk_id, vector_rank=rank, rrf_score=c.score, final_rank=rank)
            for rank, c in enumerate(chunks, start=1)
        ]
        return Retrieval(
            chunks=chunks,
            queries=[question],
            traces=traces,
            input_tokens=self._tokens[0],
            output_tokens=self._tokens[1],
        )


def scripted(outputs: list[Any]) -> tuple[Runnable[Any, Any], list[list[BaseMessage]]]:
    """模擬 with_structured_output(include_raw=True)；None 代表解析失敗。"""
    queue = iter(outputs)
    calls: list[list[BaseMessage]] = []

    def respond(messages: list[BaseMessage]) -> dict[str, Any]:
        calls.append(messages)
        parsed = next(queue)
        return {
            "raw": AIMessage(content="", usage_metadata=USAGE),
            "parsed": parsed,
            "parsing_error": None if parsed is not None else ValueError("invalid output"),
        }

    return RunnableLambda(respond), calls
