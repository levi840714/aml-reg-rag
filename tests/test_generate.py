from datetime import UTC, datetime

from langchain_core.runnables import RunnableLambda

from aml_reg_rag.evaluation.generate import (
    MIN_ARTICLE_CHARS,
    GeneratedQuestion,
    generate_questions,
)
from tests.fakes import article

NOW = datetime(2026, 9, 30, tzinfo=UTC)
MODEL = "google_genai:gemini-3.5-flash-lite"


def test_generates_one_question_per_eligible_article():
    articles = [
        article("甲" * MIN_ARTICLE_CHARS, article_no="2"),
        article("本辦法自發布日施行。", article_no="18"),
    ]
    prompts = []

    def respond(messages):
        prompts.append(messages[-1].content)
        return GeneratedQuestion(question=" 客戶開戶時要做哪些審查？ ")

    result = generate_questions(articles, RunnableLambda(respond), MODEL, NOW)
    assert result.failures == 0
    assert [(q.id, q.question, q.expected_articles) for q in result.questions] == [
        ("s-G0380268:2", "客戶開戶時要做哪些審查？", ["G0380268:2"])
    ]
    assert result.questions[0].generator_model == MODEL
    assert len(prompts) == 1
    assert "甲" in prompts[0]


def test_failed_generations_are_counted():
    articles = [article("甲" * 40, article_no="2"), article("乙" * 40, article_no="3")]

    def respond(messages):
        if "乙" in messages[-1].content:
            raise ValueError("boom")
        return GeneratedQuestion(question="問題？")

    result = generate_questions(articles, RunnableLambda(respond), MODEL, NOW)
    assert [q.id for q in result.questions] == ["s-G0380268:2"]
    assert result.failures == 1
