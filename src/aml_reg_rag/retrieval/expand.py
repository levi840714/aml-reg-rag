from collections.abc import Mapping, Sequence

from aml_reg_rag.models import ChunkRef, RefKind, RetrievedChunk

_PRIORITY: tuple[RefKind, ...] = ("apply", "definition", "penalty", "general")


class ReferenceExpander:
    """依引用關係，為檢索結果補上相關段落。"""

    def __init__(
        self, refs: Sequence[ChunkRef], chunks: Mapping[str, RetrievedChunk], limit: int
    ) -> None:
        self._chunks = chunks
        self._limit = limit
        self._links: dict[str, list[tuple[str, RefKind]]] = {}
        for ref in refs:
            link = _link(ref)
            if link:
                hit_id, other_id = link
                self._links.setdefault(hit_id, []).append((other_id, ref.kind))
        for links in self._links.values():
            links.sort(key=lambda link: _PRIORITY.index(link[1]))

    def expand(self, hits: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        seen = {hit.chunk_id for hit in hits}
        extra: list[RetrievedChunk] = []
        for hit in hits:
            for other_id, kind in self._links.get(hit.chunk_id, []):
                chunk = self._chunks.get(other_id)
                if chunk is None or other_id in seen:
                    continue
                if len(extra) >= self._limit:
                    return extra
                seen.add(other_id)
                update = {"score": 0.0, "expanded_from": hit.chunk_id, "ref_kind": kind}
                extra.append(chunk.model_copy(update=update))
        return extra


def _link(ref: ChunkRef) -> tuple[str, str] | None:
    """回傳（命中的段落, 要補上的段落）；不展開的引用回傳 None。"""
    # 罰則條文引用義務條文，命中義務時反向補上罰則
    if ref.kind == "penalty":
        return ref.target_chunk_id, ref.src_chunk_id
    # 一般引用多為列舉其他條文，只有同一條內的「前項」等才補
    same_article = _article(ref.src_chunk_id) == _article(ref.target_chunk_id)
    if ref.kind in ("apply", "definition") or (ref.kind == "general" and same_article):
        return ref.src_chunk_id, ref.target_chunk_id
    return None


def _article(chunk_id: str) -> str:
    return ":".join(chunk_id.split(":")[:2])
