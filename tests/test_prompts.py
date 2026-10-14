from aml_reg_rag.graph.prompts import SYSTEM_PROMPT, format_sources
from aml_reg_rag.graph.state import labeled_chunks
from aml_reg_rag.models import ChunkRelation, RefKind, RetrievedChunk
from tests.fakes import retrieved


def related(chunk: RetrievedChunk, other: str, kind: RefKind, *, expanded: bool) -> RetrievedChunk:
    update: dict[str, object] = {"related": [ChunkRelation(chunk_id=other, kind=kind)]}
    if expanded:
        update |= {"expanded_from": other, "ref_kind": kind}
    return chunk.model_copy(update=update)


def test_relations_are_annotated_with_labels():
    hit = retrieved("G0400162:28", "承銷公告")
    applied_hit = related(retrieved("G0400162:16", "白皮書"), hit.chunk_id, "apply", expanded=False)
    obligation = retrieved("I0030047:7", "不得提款")
    penalty = related(
        retrieved("I0030047:12", "罰鍰"), obligation.chunk_id, "penalty", expanded=True
    )
    sources = format_sources(labeled_chunks([hit, applied_hit, obligation, penalty]))
    headers = [block.splitlines()[0] for block in sources.split("\n\n")]
    assert headers[0].endswith("】")
    assert headers[1].endswith("（S1 準用的條文）")
    assert headers[3].endswith("（補充：處罰 S3 所定義務的條文）")
    assert "準用" in SYSTEM_PROMPT and "補充" in SYSTEM_PROMPT


def test_multiple_relations_are_joined():
    a = retrieved("G0400162:28", "甲")
    b = retrieved("G0400162:29", "乙")
    c = retrieved("G0400162:16", "丙").model_copy(
        update={
            "related": [
                ChunkRelation(chunk_id=a.chunk_id, kind="apply"),
                ChunkRelation(chunk_id=b.chunk_id, kind="definition"),
            ]
        }
    )
    header = format_sources(labeled_chunks([a, b, c])).split("\n\n")[2].splitlines()[0]
    assert header.endswith("（S1 準用的條文；S2 引用的條文）")
