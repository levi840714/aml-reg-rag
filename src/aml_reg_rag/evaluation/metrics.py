from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Literal

from pydantic import BaseModel

from aml_reg_rag.models import RetrievedChunk

# 每百萬 token 美元（input, output），2026-09-30 查詢；Gemini 3.8 Flash 為 2026 年底前優惠價
PRICING: dict[str, tuple[float, float]] = {
    "google_genai:gemini-3.8-flash": (0.75, 3.75),
    "google_genai:gemini-3.5-flash-lite": (0.30, 2.50),
    "google_genai:gemini-3.1-flash-lite": (0.25, 1.50),
    "openai:gpt-6-luna": (0.10, 0.50),
    "openai:gpt-6-sol": (2.00, 10.00),
    "anthropic:claude-sonnet-5": (2.00, 10.00),
    "anthropic:claude-haiku-4-5": (1.00, 5.00),
}

RefusalOutcome = Literal["correct", "false_refusal", "false_answer"]


class RetrievalResult(BaseModel):
    question_id: str
    dataset: Literal["curated", "synthetic"]
    category: str
    expected: list[str]
    ranked: list[str]
    top_score: float | None
    queries: int = 1
    rewrite_fallback: bool = False
    rerank_fallback: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    error: str | None = None


class RetrievalSummary(BaseModel):
    questions: int
    errors: int = 0
    hit_at_1: float
    hit_at_3: float
    hit_at_k: float
    recall_at_k: float
    mrr: float
    # 以下為 v2 新增，預設值讓 v1 的結果檔仍可讀取
    fallbacks: int = 0
    avg_seconds: float | None = None
    cost_usd: float | None = None


class AnswerResult(BaseModel):
    question_id: str
    category: str
    run: int
    verified: bool = False
    predicted_answerable: bool | None = None
    refusal: RefusalOutcome | None = None
    cited_articles: list[str] = []
    cites_expected: bool | None = None
    cites_forbidden: bool = False
    coverage: float | None = None
    judge_failed: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    retrieval_input_tokens: int = 0
    retrieval_output_tokens: int = 0
    seconds: float = 0.0
    unparsed: bool = False
    error: str | None = None


class GenerationSummary(BaseModel):
    runs: int
    errors: int
    unparsed: int
    verified_rate: float | None
    refusal_accuracy: float | None
    false_refusals: int
    false_answers: int
    cites_expected_rate: float | None
    cites_forbidden: int
    coverage: float | None
    judge_failures: int
    avg_seconds: float | None
    cost_usd: float | None


class UnstableQuestion(BaseModel):
    question_id: str
    outcomes: list[str]


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    if input_tokens == 0 and output_tokens == 0:
        return 0.0
    price = PRICING.get(model)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


def article_ranking(chunks: list[RetrievedChunk]) -> list[str]:
    return list(dict.fromkeys(chunk.article_key for chunk in chunks))


def hit_at(ranked: list[str], expected: list[str], n: int) -> bool:
    return any(article in expected for article in ranked[:n])


def recall(ranked: list[str], expected: list[str]) -> float:
    targets = set(expected)
    return len(targets & set(ranked)) / len(targets)


def reciprocal_rank(ranked: list[str], expected: list[str]) -> float:
    for position, article in enumerate(ranked, start=1):
        if article in expected:
            return 1 / position
    return 0.0


def refusal_outcome(expected_answerable: bool, predicted_answerable: bool) -> RefusalOutcome:
    if expected_answerable == predicted_answerable:
        return "correct"
    return "false_refusal" if expected_answerable else "false_answer"


def summarize_retrieval(results: list[RetrievalResult], light_model: str) -> RetrievalSummary:
    done = [r for r in results if r.error is None]
    count = len(done) or 1  # 全數失敗時各指標為 0.0
    return RetrievalSummary(
        questions=len(done),
        errors=len(results) - len(done),
        hit_at_1=sum(hit_at(r.ranked, r.expected, 1) for r in done) / count,
        hit_at_3=sum(hit_at(r.ranked, r.expected, 3) for r in done) / count,
        hit_at_k=sum(hit_at(r.ranked, r.expected, len(r.ranked)) for r in done) / count,
        recall_at_k=sum(recall(r.ranked, r.expected) for r in done) / count,
        mrr=sum(reciprocal_rank(r.ranked, r.expected) for r in done) / count,
        fallbacks=sum(r.rewrite_fallback or r.rerank_fallback for r in done),
        avg_seconds=_mean(r.seconds for r in done),
        cost_usd=cost_usd(
            light_model, sum(r.input_tokens for r in done), sum(r.output_tokens for r in done)
        ),
    )


def summarize_generation(
    results: list[AnswerResult], model: str, judge_model: str, light_model: str
) -> GenerationSummary:
    done = [r for r in results if r.error is None]
    answered = [r for r in done if r.refusal is not None]
    generation_cost = cost_usd(
        model, sum(r.input_tokens for r in done), sum(r.output_tokens for r in done)
    )
    judge_cost = cost_usd(
        judge_model,
        sum(r.judge_input_tokens for r in done),
        sum(r.judge_output_tokens for r in done),
    )
    retrieval_cost = cost_usd(
        light_model,
        sum(r.retrieval_input_tokens for r in done),
        sum(r.retrieval_output_tokens for r in done),
    )
    costs = (generation_cost, judge_cost, retrieval_cost)
    return GenerationSummary(
        runs=len(done),
        errors=len(results) - len(done),
        unparsed=sum(r.unparsed for r in done),
        verified_rate=_rate(done, lambda r: r.verified),
        refusal_accuracy=_rate(answered, lambda r: r.refusal == "correct"),
        false_refusals=sum(r.refusal == "false_refusal" for r in done),
        false_answers=sum(r.refusal == "false_answer" for r in done),
        cites_expected_rate=_rate(
            [r for r in done if r.cites_expected is not None], lambda r: bool(r.cites_expected)
        ),
        cites_forbidden=sum(r.cites_forbidden for r in done),
        coverage=_mean(r.coverage for r in done if r.coverage is not None),
        judge_failures=sum(r.judge_failed for r in done),
        avg_seconds=_mean(r.seconds for r in done),
        cost_usd=None if None in costs else sum(cost for cost in costs if cost is not None),
    )


def consistency(results: list[AnswerResult]) -> tuple[float | None, list[UnstableQuestion]]:
    """重複執行時，每題的拒答判斷與驗證結果是否一致；出錯的執行不計入。"""
    outcomes: dict[str, list[str]] = defaultdict(list)
    for result in results:
        if result.error is None:
            outcomes[result.question_id].append(f"{result.refusal}/{result.verified}")
    repeated = {qid: values for qid, values in outcomes.items() if len(values) > 1}
    if not repeated:
        return None, []
    unstable = [
        UnstableQuestion(question_id=qid, outcomes=values)
        for qid, values in repeated.items()
        if len(set(values)) > 1
    ]
    return 1 - len(unstable) / len(repeated), unstable


def group_by[Item](items: Iterable[Item], key: Callable[[Item], str]) -> dict[str, list[Item]]:
    groups: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    return dict(groups)


def _rate(items: list[AnswerResult], predicate: Callable[[AnswerResult], bool]) -> float | None:
    return sum(predicate(item) for item in items) / len(items) if items else None


def _mean(values: Iterable[float]) -> float | None:
    collected = list(values)
    return sum(collected) / len(collected) if collected else None
