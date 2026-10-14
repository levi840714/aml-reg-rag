"""解析條文中的引用，並對應到被引用的段落。"""

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from aml_reg_rag.ingest.chunker import chunk_path, parse_units
from aml_reg_rag.models import Chunk, ChunkRef, LawArticle, RefKind

# 項、款、目；None 表示未指定，涵蓋該層全部
Position = tuple[int | None, int | None, int | None]
WHOLE: Position = (None, None, None)

_DIGITS = {char: value for value, char in enumerate("零一二三四五六七八九")}
_N = "[零一二三四五六七八九十百]+"
_UNIT = re.compile(
    rf"(?:(?P<rel>前(?P<rel_n>{_N})?(?P<rel_level>條|項|款|目))"
    rf"|(?P<self>本條|同條)(?!例)"
    rf"|第(?P<article>{_N})條(?:之(?P<article_sub>{_N}))?)?"
    rf"(?:第(?P<p>{_N})項)?(?:第(?P<k>{_N})款)?(?:第(?P<m>{_N})目)?"
    r"(?:前段|後段|但書|本文)?"
)
_CONNECTORS = {"", "、", "及", "或", "與", "和", "暨", "至"}
_ALIAS = re.compile(r"（以下簡稱(本法|本條例)）")
_ALIAS_SUFFIX = re.compile(r"（以下簡稱[^）]*）$")
_EXTERNAL_SUFFIX = re.compile(r"(法|條例|規則|辦法|準則|規程)$")
_OWN_LAW = ("本辦法", "本法", "本條例")
_CONTEXT_LAW = ("同法", "同條例")
_SENTENCE_END = re.compile(r"[。；：\n]")
_DEFINITION = re.compile(r"^(規定)?之?(所稱|所定|所列|所指)")
_PENALTY = re.compile(r"罰鍰|有期徒刑|罰金|拘役")
_LEVEL_GROUPS = ("p", "k", "m")
_CITING_GROUPS = ("rel", "self", "article", *_LEVEL_GROUPS)


def cn_number(text: str) -> int:
    total, digit = 0, 0
    for char in text:
        if char in _DIGITS:
            digit = _DIGITS[char]
        else:
            total += (digit or 1) * (100 if char == "百" else 10)
            digit = 0
    return total + digit


@dataclass(frozen=True)
class Reference:
    source: Position
    pcode: str
    article_no: str
    target: Position
    kind: RefKind
    raw: str


@dataclass(frozen=True)
class _Target:
    # pcode 為 None 表示語料外的法規
    pcode: str | None
    article_no: str | None
    position: Position
    raw: str


@dataclass
class _Chain:
    start: int
    end: int
    targets: list[_Target]


class ReferenceParser:
    def __init__(self, articles: Sequence[LawArticle]) -> None:
        self._pcodes = {article.law_name: article.pcode for article in articles}
        self._names = sorted(self._pcodes, key=len, reverse=True)
        numbers: dict[str, set[str]] = {}
        for article in articles:
            numbers.setdefault(article.pcode, set()).add(article.article_no)
        self._order = {pcode: sorted(nos, key=_article_sort_key) for pcode, nos in numbers.items()}
        self._aliases: dict[str, dict[str, str | None]] = {}
        for article in articles:
            for match in _ALIAS.finditer(article.content):
                law = self._law_named(article.content[: match.start()])
                self._aliases.setdefault(article.pcode, {})[match.group(1)] = law

    def parse(self, article: LawArticle) -> list[Reference]:
        refs: list[Reference] = []
        for text, source in _lines(article.content):
            for chain in self._chains(article, text, source):
                kind = _classify(text, chain.start, chain.end)
                refs.extend(
                    Reference(source, t.pcode, t.article_no, t.position, kind, t.raw)
                    for t in chain.targets
                    if t.pcode is not None and t.article_no is not None
                )
        return refs

    def _chains(self, article: LawArticle, text: str, source: Position) -> list[_Chain]:
        chains: list[_Chain] = []
        own = _Target(article.pcode, article.article_no, source, "")
        context: _Target | None = None
        prev_end = 0
        for match in _UNIT.finditer(text):
            if not any(match.group(name) for name in _CITING_GROUPS):
                continue
            prefix, connector = self._split_prefix(text[prev_end : match.start()])
            chained = context is not None and prefix is None and connector in _CONNECTORS
            # 外部法規的條號清單常以「之罪」「準用」隔開，同一行內沿用
            sticky = context is not None and context.pcode is None and prefix is None
            base = context if context is not None and (chained or sticky) else own
            law = self._prefix_law(prefix, article.pcode, base, text[: match.start()])
            targets = self._resolve(match, law, article, source, base, context)
            if chained and connector == "至" and targets:
                targets = [*self._expand_range(context, targets[0]), *targets]
            if chained:
                chains[-1].targets.extend(targets)
                chains[-1].end = match.end()
            else:
                chains.append(_Chain(match.start(), match.end(), targets))
            context = targets[-1] if targets else context
            prev_end = match.end()
        return chains

    def _split_prefix(self, gap: str) -> tuple[str | None, str]:
        gap = _ALIAS_SUFFIX.sub("", gap).strip()
        for token in (*self._names, *_OWN_LAW, *_CONTEXT_LAW):
            if gap.endswith(token):
                return token, gap[: -len(token)]
        if _EXTERNAL_SUFFIX.search(gap):
            return "", gap
        return None, gap

    def _prefix_law(self, prefix: str | None, own: str, base: _Target, before: str) -> str | None:
        if prefix is None:
            return base.pcode
        if prefix in self._pcodes:
            return self._pcodes[prefix]
        if prefix == "本辦法":
            return own
        if prefix in ("本法", "本條例"):
            return self._aliases.get(own, {}).get(prefix, own)
        if prefix in _CONTEXT_LAW:
            before = _ALIAS_SUFFIX.sub("", before).rstrip()[: -len(prefix)]
            return self._last_named(before, prefix[1:])
        return None

    def _resolve(
        self,
        match: re.Match[str],
        law: str | None,
        article: LawArticle,
        source: Position,
        base: _Target,
        context: _Target | None,
    ) -> list[_Target]:
        raw = match.group(0)
        levels = [cn_number(v) if v else None for v in match.group(*_LEVEL_GROUPS)]
        if match.group("rel"):
            return self._relative(match, article, source, levels, raw)
        if match.group("self") == "本條":
            return [_Target(article.pcode, article.article_no, _position(levels, WHOLE), raw)]
        if match.group("self") == "同條" and context:
            return [_Target(context.pcode, context.article_no, _position(levels, WHOLE), raw)]
        if match.group("article"):
            number = str(cn_number(match.group("article")))
            if match.group("article_sub"):
                number = f"{number}-{cn_number(match.group('article_sub'))}"
            return [_Target(law, number, _position(levels, WHOLE), raw)]
        return [_Target(base.pcode, base.article_no, _inherit(levels, base.position), raw)]

    def _relative(
        self,
        match: re.Match[str],
        article: LawArticle,
        source: Position,
        levels: list[int | None],
        raw: str,
    ) -> list[_Target]:
        count = cn_number(match.group("rel_n")) if match.group("rel_n") else 1
        level = match.group("rel_level")
        if level == "條":
            order = self._order.get(article.pcode, [])
            index = order.index(article.article_no) if article.article_no in order else 0
            return [
                _Target(article.pcode, number, _position(levels, WHOLE), raw)
                for number in order[max(0, index - count) : index]
            ]
        depth = "項款目".index(level)
        current = source[depth]
        if current is None:
            return []
        targets = []
        for value in range(max(1, current - count), current):
            position = (*source[:depth], value, *levels[depth + 1 :])
            targets.append(_Target(article.pcode, article.article_no, _fill(position), raw))
        return targets

    def _expand_range(self, start: _Target | None, end: _Target) -> Iterator[_Target]:
        pcode = end.pcode
        if start is None or pcode is None or start.pcode != pcode:
            return
        if start.article_no != end.article_no:
            order = self._order.get(pcode, [])
            if start.article_no in order and end.article_no in order:
                first, last = order.index(start.article_no), order.index(end.article_no)
                for number in order[first + 1 : last]:
                    yield _Target(pcode, number, WHOLE, end.raw)
            return
        depth = _deepest(end.position)
        if depth is None or start.position[:depth] != end.position[:depth]:
            return
        lo, hi = start.position[depth], end.position[depth]
        if lo is None or hi is None:
            return
        for value in range(lo + 1, hi):
            position = _fill((*end.position[:depth], value))
            yield _Target(pcode, end.article_no, position, end.raw)

    def _last_named(self, text: str, suffix: str) -> str | None:
        # 「同條例」指前文最近提到的條例，即使它沒有接條號
        end = text.rfind(suffix)
        return self._law_named(text[: end + len(suffix)]) if end >= 0 else None

    def _law_named(self, text: str) -> str | None:
        return next((self._pcodes[name] for name in self._names if text.endswith(name)), None)


def link_chunks(chunks: Sequence[Chunk]) -> list[ChunkRef]:
    by_article: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_article.setdefault(chunk.article.article_key, []).append(chunk)
    articles = [group[0].article for group in by_article.values()]
    parser = ReferenceParser(articles)
    links: dict[tuple[str, str, RefKind], ChunkRef] = {}
    for article in articles:
        for ref in parser.parse(article):
            sources = _covering(by_article[article.article_key], ref.source)
            targets = _covering(by_article.get(f"{ref.pcode}:{ref.article_no}", []), ref.target)
            for src in sources:
                for target in targets:
                    key = (src, target, ref.kind)
                    if src != target and key not in links:
                        links[key] = ChunkRef(
                            src_chunk_id=src, target_chunk_id=target, kind=ref.kind, raw=ref.raw
                        )
    return list(links.values())


def _lines(content: str) -> Iterator[tuple[str, Position]]:
    # 與切段器相同的樹狀位置；目以下的子目併入所屬的目
    for p, paragraph in enumerate(parse_units(content), start=1):
        yield paragraph.head, (p, None, None)
        for k, subparagraph in enumerate(paragraph.children, start=1):
            yield subparagraph.head, (p, k, None)
            for m, item in enumerate(subparagraph.children, start=1):
                yield item.text(), (p, k, m)


def _classify(text: str, start: int, end: int) -> RefKind:
    sentence_start = max((m.end() for m in _SENTENCE_END.finditer(text, 0, start)), default=0)
    sentence_end = _SENTENCE_END.search(text, end)
    before = text[sentence_start:start]
    after = text[end : sentence_end.start() if sentence_end else len(text)]
    if _DEFINITION.match(after):
        return "definition"
    # 「準用第十六條」或「第十六條規定，於前項情形準用之」
    if before.endswith("準用") or (after.startswith("規定") and "準用" in after):
        return "apply"
    if "違反" in before and _PENALTY.search(before + after):
        return "penalty"
    if "依" in before and "訂定" in after:
        return "authority"
    return "general"


def _covering(chunks: Sequence[Chunk], position: Position) -> list[str]:
    return [chunk.chunk_id for chunk in chunks if _covers(chunk_path(chunk.chunk_id), position)]


def _covers(path: tuple[tuple[int, int], ...], position: Position) -> bool:
    return all(
        value is None or lo <= value <= hi for (lo, hi), value in zip(path, position, strict=False)
    )


def _position(levels: list[int | None], fallback: Position) -> Position:
    # 條下直接引用款、目時，條文只有一項
    p, k, m = levels
    if p is None and (k is not None or m is not None):
        p = 1
    return (p, k, m) if any(v is not None for v in (p, k, m)) else fallback


def _inherit(levels: list[int | None], base: Position) -> Position:
    first = next((depth for depth, value in enumerate(levels) if value is not None), 3)
    return _fill((*base[:first], *levels[first:]))


def _fill(values: tuple[int | None, ...]) -> Position:
    padded = (*values, None, None, None)[:3]
    return (padded[0], padded[1], padded[2])


def _deepest(position: Position) -> int | None:
    specified = [depth for depth, value in enumerate(position) if value is not None]
    return specified[-1] if specified else None


def _article_sort_key(article_no: str) -> tuple[int, ...]:
    return tuple(int(part) for part in article_no.split("-"))
