from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from aml_reg_rag.evaluation import prompts
from aml_reg_rag.evaluation.dataset import SyntheticQuestion
from aml_reg_rag.llm import structured_model
from aml_reg_rag.models import LawArticle

# 施行日期等過短的條文不出題
MIN_ARTICLE_CHARS = 30
MAX_CONCURRENCY = 4


class GeneratedQuestion(BaseModel):
    question: str = Field(description="實務人員會提出、且能由該條文回答的問題")


@dataclass(frozen=True)
class GenerationResult:
    questions: list[SyntheticQuestion]
    failures: int


def eligible_articles(articles: list[LawArticle]) -> list[LawArticle]:
    return [article for article in articles if len(article.content) >= MIN_ARTICLE_CHARS]


def generate_questions(
    articles: list[LawArticle], generator: Runnable[Any, Any], model_name: str, now: datetime
) -> GenerationResult:
    targets = eligible_articles(articles)
    outputs = generator.batch(
        [prompts.question_messages(article) for article in targets],
        config={"max_concurrency": MAX_CONCURRENCY},
        return_exceptions=True,
    )
    questions: list[SyntheticQuestion] = []
    for article, output in zip(targets, outputs, strict=True):
        if isinstance(output, GeneratedQuestion) and output.question.strip():
            questions.append(
                SyntheticQuestion(
                    id=f"s-{article.article_key}",
                    question=output.question.strip(),
                    expected_articles=[article.article_key],
                    generator_model=model_name,
                    created_at=now,
                )
            )
    return GenerationResult(questions, failures=len(targets) - len(questions))


def make_generator(model: str, timeout: float) -> Runnable[Any, Any]:
    return structured_model(model, GeneratedQuestion, timeout=timeout)
