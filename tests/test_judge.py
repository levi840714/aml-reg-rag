from aml_reg_rag.evaluation.judge import JudgeOutput, KeyPointVerdict, judge_answer
from tests.fakes import answer, cite, point, scripted

ANSWER = answer(point("需確認客戶身分。", cite("S1", "應於確認客戶身分時")))
KEY_POINTS = ["要點一", "要點二"]


def verdicts(*covered: bool) -> JudgeOutput:
    return JudgeOutput(verdicts=[KeyPointVerdict(covered=c, reason="理由") for c in covered])


def test_coverage_from_verdicts():
    judge, calls = scripted([verdicts(True, False)])
    result = judge_answer(judge, "問題", KEY_POINTS, ANSWER)
    assert result.coverage == 0.5
    assert (result.input_tokens, result.output_tokens) == (100, 20)
    prompt = calls[0][-1].content
    assert "1. 要點一" in prompt
    assert "- 需確認客戶身分。" in prompt


def test_verdict_count_mismatch_is_a_failure():
    judge, _ = scripted([verdicts(True)])
    result = judge_answer(judge, "問題", KEY_POINTS, ANSWER)
    assert result.verdicts is None
    assert result.coverage is None


def test_unparsed_output_is_a_failure():
    judge, _ = scripted([None])
    assert judge_answer(judge, "問題", KEY_POINTS, ANSWER).verdicts is None
