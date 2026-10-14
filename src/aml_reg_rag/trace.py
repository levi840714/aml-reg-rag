import json
from datetime import datetime
from pathlib import Path
from typing import Any

from aml_reg_rag.graph.state import AskState
from aml_reg_rag.models import RetrievedChunk


def ask_record(
    state: AskState, *, table: str, model: str, seconds: float, now: datetime
) -> dict[str, Any]:
    chunks = state["chunks"]
    answer = state["answer"]
    return {
        "time": now.isoformat(),
        "question": state["question"],
        "index": table,
        "model": model,
        "retrieved": [_retrieved_record(chunk) for chunk in chunks],
        "top_score": round(chunks[0].score, 4) if chunks else None,
        "retrieval": _retrieval_record(state),
        "answer": answer.model_dump() if answer else None,
        "verified": state["verified"],
        "problems": state["problems"],
        "attempts": state["attempts"],
        "input_tokens": state["input_tokens"],
        "output_tokens": state["output_tokens"],
        "seconds": round(seconds, 2),
    }


def append_trace(trace_dir: Path, record: dict[str, Any], now: datetime) -> Path:
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{now:%Y-%m-%d}.jsonl"
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _retrieved_record(chunk: RetrievedChunk) -> dict[str, Any]:
    record: dict[str, Any] = {"chunk_id": chunk.chunk_id, "score": round(chunk.score, 4)}
    if chunk.expanded_from is not None:
        record |= {"expanded_from": chunk.expanded_from, "ref_kind": chunk.ref_kind}
    return record


def _retrieval_record(state: AskState) -> dict[str, Any] | None:
    retrieval = state["retrieval"]
    if retrieval is None:
        return None
    # 段落內文已在 retrieved 與法規資料庫中，trace 只留各階段名次與分數
    return retrieval.model_dump(exclude={"chunks"})
