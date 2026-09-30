import re
from dataclasses import dataclass, field

from aml_reg_rag.models import Chunk, LawArticle

# 依序為款、目、子目的行首格式；不符合者為新的一項
_MARKERS = (
    re.compile(r"^[一二三四五六七八九十百]+、"),
    re.compile(r"^[（(][一二三四五六七八九十百]+[）)]"),
    re.compile(r"^\d+[.．]"),
)
_LEVEL_NAMES = ("項", "款", "目")
_ID_PREFIXES = ("p", "k", "m")
# 最多切到目；子目隨所屬目，不再切分
_MAX_SPLIT_DEPTH = len(_LEVEL_NAMES) - 1

Segment = tuple[int, int]


@dataclass
class Unit:
    head: str
    level: int
    children: list["Unit"] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join([self.head, *(child.text() for child in self.children)])


@dataclass(frozen=True)
class Piece:
    path: tuple[Segment, ...]
    body: str


def line_level(line: str) -> int:
    for level, pattern in enumerate(_MARKERS, start=1):
        if pattern.match(line):
            return level
    return 0


def parse_units(content: str) -> list[Unit]:
    roots: list[Unit] = []
    stack: list[Unit] = []
    for line in (raw.strip() for raw in content.splitlines()):
        if not line:
            continue
        unit = Unit(line, line_level(line))
        while stack and stack[-1].level >= unit.level:
            stack.pop()
        (stack[-1].children if stack else roots).append(unit)
        stack.append(unit)
    return roots


def chunk_article(article: LawArticle, max_chars: int) -> list[Chunk]:
    roots = parse_units(article.content)
    whole = "\n".join(unit.text() for unit in roots)
    pieces = [] if len(whole) <= max_chars else _split(roots, [], (), max_chars)
    if len(pieces) <= 1:
        return [Chunk(chunk_id=article.article_key, article=article, locator="", body=whole)]
    return [
        Chunk(
            chunk_id=_chunk_id(article, piece.path),
            article=article,
            locator=_locator(piece.path),
            body=piece.body,
        )
        for piece in pieces
    ]


def _split(
    units: list[Unit], prefix: list[str], fixed: tuple[Segment, ...], max_chars: int
) -> list[Piece]:
    pieces: list[Piece] = []
    group: list[int] = []

    def body(indexes: list[int]) -> str:
        return "\n".join([*prefix, *(units[index - 1].text() for index in indexes)])

    def flush() -> None:
        if group:
            pieces.append(Piece((*fixed, (group[0], group[-1])), body(group)))
            group.clear()

    for index, unit in enumerate(units, start=1):
        too_long = len(body([index])) > max_chars
        if too_long and unit.children and len(fixed) < _MAX_SPLIT_DEPTH:
            flush()
            child_prefix = [*prefix, unit.head]
            pieces.extend(_split(unit.children, child_prefix, (*fixed, (index, index)), max_chars))
        elif group and len(body([*group, index])) <= max_chars:
            group.append(index)
        else:
            flush()
            group.append(index)
    flush()
    return pieces


def _chunk_id(article: LawArticle, path: tuple[Segment, ...]) -> str:
    parts = [f"{_ID_PREFIXES[depth]}{_range(segment)}" for depth, segment in enumerate(path)]
    return ":".join([article.article_key, *parts])


def _range(segment: Segment) -> str:
    start, end = segment
    return str(start) if start == end else f"{start}-{end}"


def _locator(path: tuple[Segment, ...]) -> str:
    parts = []
    for depth, (start, end) in enumerate(path):
        name = _LEVEL_NAMES[depth]
        part = f"第 {start} {name}"
        parts.append(part if start == end else f"{part}至第 {end} {name}")
    return "".join(parts)
