import pytest

from aml_reg_rag.evaluation.metrics import (
    AnswerResult,
    RetrievalResult,
    article_ranking,
    consistency,
    cost_usd,
    hit_at,
    recall,
    reciprocal_rank,
    refusal_outcome,
    summarize_generation,
    summarize_retrieval,
)
from tests.fakes import retrieved

FLASH = "google_genai:gemini-3.8-flash"


def test_article_ranking_dedupes_chunks_of_same_article():
    chunks = [
        retrieved("G0380252:3:p1:k1-4", "甲"),
        retrieved("G0380268:9", "乙"),
        retrieved("G0380252:3:p1:k5-6", "丙"),
    ]
    assert article_ranking(chunks) == ["G0380252:3", "G0380268:9"]


def test_rank_metrics():
    ranked = ["A:1", "B:2", "C:3"]
    assert hit_at(ranked, ["B:2"], 1) is False
    assert hit_at(ranked, ["B:2"], 3) is True
    assert recall(ranked, ["B:2", "D:4"]) == 0.5
    assert reciprocal_rank(ranked, ["C:3"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, ["D:4"]) == 0.0


def test_refusal_outcome():
    assert refusal_outcome(True, True) == "correct"
    assert refusal_outcome(False, False) == "correct"
    assert refusal_outcome(True, False) == "false_refusal"
    assert refusal_outcome(False, True) == "false_answer"


def retrieval(qid, expected, ranked) -> RetrievalResult:
    return RetrievalResult(
        question_id=qid,
        dataset="curated",
        category="lookup",
        expected=expected,
        ranked=ranked,
        top_score=0.9,
    )


def test_summarize_retrieval():
    summary = summarize_retrieval(
        [retrieval("q1", ["A:1"], ["A:1", "B:2"]), retrieval("q2", ["C:3"], ["A:1", "C:3"])]
    )
    assert summary.questions == 2
    assert (summary.hit_at_1, summary.hit_at_3, summary.hit_at_k) == (0.5, 1.0, 1.0)
    assert (summary.recall_at_k, summary.mrr) == (1.0, 0.75)


def test_cost_usd():
    assert cost_usd(FLASH, 1_000_000, 1_000_000) == 4.5
    assert cost_usd("unknown:model", 1, 1) is None


def generation_results() -> list[AnswerResult]:
    return [
        AnswerResult(
            question_id="q1",
            category="lookup",
            run=1,
            verified=True,
            predicted_answerable=True,
            refusal="correct",
            cites_expected=True,
            coverage=1.0,
            input_tokens=1000,
            output_tokens=100,
            judge_input_tokens=500,
            judge_output_tokens=50,
            seconds=2.0,
        ),
        AnswerResult(
            question_id="q2",
            category="refusal",
            run=1,
            verified=True,
            predicted_answerable=True,
            refusal="false_answer",
            cites_forbidden=True,
            seconds=4.0,
        ),
        AnswerResult(question_id="q3", category="lookup", run=1, error="TimeoutError"),
    ]


def test_summarize_generation():
    summary = summarize_generation(generation_results(), FLASH, FLASH)
    assert (summary.runs, summary.errors) == (2, 1)
    assert (summary.verified_rate, summary.refusal_accuracy) == (1.0, 0.5)
    assert (summary.false_refusals, summary.false_answers) == (0, 1)
    assert (summary.cites_expected_rate, summary.cites_forbidden) == (1.0, 1)
    assert (summary.coverage, summary.avg_seconds) == (1.0, 3.0)
    assert summary.cost_usd == pytest.approx((1500 * 0.75 + 150 * 3.75) / 1_000_000)


def test_unknown_model_cost_is_none():
    assert summarize_generation(generation_results(), "unknown:model", FLASH).cost_usd is None


def outcome(qid, run, refusal) -> AnswerResult:
    return AnswerResult(question_id=qid, category="lookup", run=run, verified=True, refusal=refusal)


def test_consistency_flags_unstable_questions():
    results = [
        outcome("q1", 1, "correct"),
        outcome("q1", 2, "correct"),
        outcome("q2", 1, "correct"),
        outcome("q2", 2, "false_refusal"),
    ]
    agreement, unstable = consistency(results)
    assert agreement == 0.5
    assert [item.question_id for item in unstable] == ["q2"]


def test_consistency_needs_repeats():
    assert consistency([outcome("q1", 1, "correct")]) == (None, [])
