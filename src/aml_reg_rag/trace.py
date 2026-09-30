import json
from datetime import datetime
from pathlib import Path
from typing import Any

from aml_reg_rag.graph.state import AskState


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
        "retrieved": [{"chunk_id": c.chunk_id, "score": round(c.score, 4)} for c in chunks],
        "top_score": round(chunks[0].score, 4) if chunks else None,
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
