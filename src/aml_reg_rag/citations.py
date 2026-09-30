import re
import unicodedata

from aml_reg_rag.models import Answer

MIN_QUOTE_CHARS = 8
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WHITESPACE.sub("", unicodedata.normalize("NFKC", text))


def verify(answer: Answer | None, bodies: dict[str, str]) -> list[str]:
    """回傳問題清單，空清單代表通過；bodies 為來源標籤（S1…）對應的段落內文。"""
    if answer is None:
        return ["輸出格式錯誤，無法解析為指定結構"]
    if not answer.answerable:
        return _verify_refusal(answer)
    if not answer.points:
        return ["可回答時至少需要一個論點"]
    normalized = {label: normalize(body) for label, body in bodies.items()}
    problems: list[str] = []
    for number, point in enumerate(answer.points, start=1):
        if not point.citations:
            problems.append(f"論點 {number} 沒有引用")
        for citation in point.citations:
            problems.extend(_verify_citation(number, citation.source, citation.quote, normalized))
    return problems


def _verify_refusal(answer: Answer) -> list[str]:
    problems = []
    if answer.points:
        problems.append("拒答時不應有論點")
    if not (answer.missing and answer.missing.strip()):
        problems.append("拒答時需說明段落缺少哪些規範")
    return problems


def _verify_citation(number: int, source: str, quote: str, bodies: dict[str, str]) -> list[str]:
    if source not in bodies:
        return [f"論點 {number} 引用的 {source} 不在本次檢索結果中"]
    target = normalize(quote)
    if sum(ch.isalnum() for ch in target) < MIN_QUOTE_CHARS:
        return [f"論點 {number} 引用 {source} 的原文少於 {MIN_QUOTE_CHARS} 字"]
    if target not in bodies[source]:
        return [f"論點 {number} 引用 {source} 的原文不在該段落中：「{quote}」"]
    return []
