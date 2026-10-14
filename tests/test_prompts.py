from aml_reg_rag.graph.prompts import SYSTEM_PROMPT, format_sources
from aml_reg_rag.graph.state import labeled_chunks
from tests.fakes import retrieved


def test_expanded_chunks_are_annotated_with_the_citing_label():
    hit = retrieved("G0400162:28", "承銷公告")
    obligation = retrieved("I0030047:7", "不得提款")
    applied = retrieved("G0400162:16", "白皮書").model_copy(
        update={"expanded_from": hit.chunk_id, "ref_kind": "apply"}
    )
    penalty = retrieved("I0030047:12", "罰鍰").model_copy(
        update={"expanded_from": obligation.chunk_id, "ref_kind": "penalty"}
    )
    sources = format_sources(labeled_chunks([hit, obligation, applied, penalty]))
    blocks = sources.split("\n\n")
    assert "補充" not in blocks[0]
    assert blocks[2].splitlines()[0].endswith("（補充：S1 準用的條文）")
    assert blocks[3].splitlines()[0].endswith("（補充：處罰 S2 所定義務的條文）")
    assert "補充" in SYSTEM_PROMPT
