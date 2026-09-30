from langchain_core.runnables import RunnableLambda

from aml_reg_rag.retrieval.rewrite import QueryRewriter, SubQueries
from tests.fakes import scripted

QUESTION = "虛擬資產服務商沒有通報制裁對象的資產，會被怎麼處罰？"


def rewrite_with(*outputs):
    model, calls = scripted(list(outputs))
    return QueryRewriter(model).rewrite(QUESTION), calls


def test_original_question_comes_first():
    result, calls = rewrite_with(SubQueries(queries=["資恐防制法通報義務", "違反通報義務罰鍰"]))
    assert result.queries == [QUESTION, "資恐防制法通報義務", "違反通報義務罰鍰"]
    assert result.fallback is None
    assert (result.input_tokens, result.output_tokens) == (100, 20)
    assert f"<question>\n{QUESTION}\n</question>" in calls[0][-1].content


def test_subqueries_are_deduplicated_and_capped():
    result, _ = rewrite_with(SubQueries(queries=[QUESTION, " 甲 ", "甲", "乙", "丙"]))
    assert result.queries == [QUESTION, "甲", "乙"]


def test_empty_subqueries_keep_only_the_question():
    result, _ = rewrite_with(SubQueries(queries=[]))
    assert result.queries == [QUESTION]


def test_unparsed_output_falls_back():
    result, _ = rewrite_with(None)
    assert (result.queries, result.fallback) == ([QUESTION], "輸出無法解析")


def test_api_error_falls_back_with_description():
    class ApiError(Exception):
        status_code = 503

    def boom(messages):
        raise ApiError

    result = QueryRewriter(RunnableLambda(boom)).rewrite(QUESTION)
    assert (result.queries, result.fallback) == ([QUESTION], "ApiError（狀態碼 503）")
