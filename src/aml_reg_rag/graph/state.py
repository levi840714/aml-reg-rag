from typing import TypedDict

from aml_reg_rag.models import Answer, Retrieval, RetrievedChunk


class AskState(TypedDict):
    question: str
    k: int
    chunks: list[RetrievedChunk]
    retrieval: Retrieval | None
    answer: Answer | None
    problems: list[str]
    attempts: int
    verified: bool
    input_tokens: int
    output_tokens: int


def initial_state(question: str, k: int) -> AskState:
    return {
        "question": question,
        "k": k,
        "chunks": [],
        "retrieval": None,
        "answer": None,
        "problems": [],
        "attempts": 0,
        "verified": False,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def labeled_chunks(chunks: list[RetrievedChunk]) -> dict[str, RetrievedChunk]:
    return {f"S{number}": chunk for number, chunk in enumerate(chunks, start=1)}
