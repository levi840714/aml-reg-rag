from collections import Counter

from aml_reg_rag.config import PROJECT_ROOT
from aml_reg_rag.evaluation.dataset import load_curated, validate_datasets


def test_curated_questions_follow_dataset_rules():
    questions = load_curated(PROJECT_ROOT / "eval" / "curated.jsonl")
    referenced = {key for q in questions for key in [*q.expected_articles, *q.forbidden_articles]}
    assert validate_datasets(questions, [], referenced) == []
    assert Counter(q.category for q in questions) == {
        "lookup": 5,
        "scenario": 5,
        "cross_law": 3,
        "cross_ref": 5,
        "distractor": 4,
        "refusal": 3,
    }
    assert all(q.forbidden_articles for q in questions if q.category == "distractor")
    multi_article = ("cross_law", "cross_ref")
    assert all(len(q.expected_articles) >= 2 for q in questions if q.category in multi_article)
