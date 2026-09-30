from rank_bm25 import BM25Okapi

from aml_reg_rag.models import RetrievedChunk
from aml_reg_rag.retrieval.tokenize import tokenize


class Bm25Index:
    def __init__(self, entries: list[tuple[RetrievedChunk, str]]) -> None:
        """entries 為（段落, 建索引用的全文）；全文含標頭，法規名稱也能被比對。"""
        self._chunks = [chunk for chunk, _ in entries]
        # rank_bm25 對空語料會除以零，故空索引以 None 表示。
        self._bm25 = BM25Okapi([tokenize(text) for _, text in entries]) if entries else None

    def search(self, query: str, n: int) -> list[tuple[RetrievedChunk, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:n]
        return [(self._chunks[i], float(scores[i])) for i in order if scores[i] > 0]
