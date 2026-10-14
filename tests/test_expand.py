from aml_reg_rag.models import ChunkRef, RefKind, RetrievedChunk
from aml_reg_rag.retrieval.expand import ReferenceExpander
from tests.fakes import retrieved

CHUNK_IDS = [
    "G0400162:28",
    "G0400162:16",
    "G0400162:1",
    "G0380131:5:p1-2",
    "I0030047:7",
    "I0030047:12",
    "G0380268:2:p1",
    "G0380268:2:p2-3",
    "G0380268:3",
    "G0380268:9",
]
CHUNKS = {chunk_id: retrieved(chunk_id, f"{chunk_id} 內文") for chunk_id in CHUNK_IDS}


def ref(src: str, target: str, kind: RefKind) -> ChunkRef:
    return ChunkRef(src_chunk_id=src, target_chunk_id=target, kind=kind, raw="第X條")


def expand(refs: list[ChunkRef], hit_ids: list[str], limit: int = 3) -> list[RetrievedChunk]:
    return ReferenceExpander(refs, CHUNKS, limit).expand([CHUNKS[i] for i in hit_ids])


def test_follows_mutatis_mutandis_and_definition_citations():
    refs = [
        ref("G0400162:28", "G0400162:16", "apply"),
        ref("G0400162:28", "G0400162:1", "authority"),
        ref("I0030047:12", "G0380131:5:p1-2", "definition"),
        ref("I0030047:12", "I0030047:7", "general"),
    ]
    extra = expand(refs, ["G0400162:28", "I0030047:12"])
    assert [(c.chunk_id, c.expanded_from, c.ref_kind) for c in extra] == [
        ("G0400162:16", "G0400162:28", "apply"),
        ("G0380131:5:p1-2", "I0030047:12", "definition"),
    ]


def test_adds_penalty_article_that_cites_the_hit():
    extra = expand([ref("I0030047:12", "I0030047:7", "penalty")], ["I0030047:7"])
    assert [(c.chunk_id, c.expanded_from, c.ref_kind) for c in extra] == [
        ("I0030047:12", "I0030047:7", "penalty")
    ]


def test_follows_general_citation_only_within_the_same_article():
    refs = [
        ref("G0380268:2:p2-3", "G0380268:2:p1", "general"),
        ref("G0380268:2:p2-3", "G0380268:3", "general"),
    ]
    assert [c.chunk_id for c in expand(refs, ["G0380268:2:p2-3"])] == ["G0380268:2:p1"]


def test_skips_hits_and_duplicates_and_respects_limit():
    refs = [
        ref("G0380268:9", "G0380268:3", "definition"),
        ref("G0380268:9", "G0400162:28", "apply"),
        ref("G0400162:28", "G0400162:16", "apply"),
        ref("G0400162:16", "G0380268:3", "definition"),
        ref("G0400162:16", "G0400162:1", "definition"),
    ]
    extra = expand(refs, ["G0380268:9", "G0400162:28", "G0400162:16"], limit=2)
    assert [c.chunk_id for c in extra] == ["G0380268:3", "G0400162:1"]


def test_orders_by_hit_rank_then_kind_priority():
    refs = [
        ref("G0380268:9", "G0380268:3", "definition"),
        ref("G0380268:9", "G0400162:16", "apply"),
        ref("G0400162:28", "G0400162:1", "apply"),
    ]
    extra = expand(refs, ["G0380268:9", "G0400162:28"])
    assert [c.chunk_id for c in extra] == ["G0400162:16", "G0380268:3", "G0400162:1"]


def test_expanded_chunks_have_zero_score_and_missing_chunks_are_ignored():
    refs = [ref("G0400162:28", "G9999999:1", "apply"), ref("G0400162:28", "G0400162:16", "apply")]
    extra = expand(refs, ["G0400162:28"])
    assert [(c.chunk_id, c.score) for c in extra] == [("G0400162:16", 0.0)]


def test_annotate_marks_relations_between_retrieved_chunks():
    refs = [
        ref("G0400162:28", "G0400162:16", "apply"),
        ref("I0030047:12", "I0030047:7", "penalty"),
        ref("G0400162:28", "G0400162:1", "authority"),
        ref("G0380268:9", "G0380268:3", "general"),
        ref("G0400162:28", "G0380268:3", "definition"),
    ]
    hit_ids = ["G0400162:28", "G0400162:16", "I0030047:7", "I0030047:12", "G0400162:1"]
    chunks = ReferenceExpander(refs, CHUNKS, 3).annotate([CHUNKS[i] for i in hit_ids])
    related = {c.chunk_id: [(r.chunk_id, r.kind) for r in c.related] for c in chunks}
    assert related == {
        "G0400162:28": [],
        "G0400162:16": [("G0400162:28", "apply")],
        "I0030047:7": [],
        "I0030047:12": [("I0030047:7", "penalty")],
        "G0400162:1": [],
    }


def test_expanded_chunks_carry_their_relation():
    extra = expand([ref("G0400162:28", "G0400162:16", "apply")], ["G0400162:28"])
    assert [(r.chunk_id, r.kind) for r in extra[0].related] == [("G0400162:28", "apply")]
