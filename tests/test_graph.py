from aml_reg_rag.graph.build import build_graph, run_ask
from tests.fakes import FakeIndex, answer, cite, point, refusal, retrieved, scripted

QUESTION = "VASP 何時要確認客戶身分？"
CHUNKS = [
    retrieved("G0380268:9", "虛擬資產服務商應於確認客戶身分時，取得必要資訊。"),
    retrieved(
        "G0380252:3:p1:k1-4",
        "金融機構確認客戶身分措施，應依下列規定辦理。",
        law_name="金融機構防制洗錢辦法",
        locator="第 1 項第 1 款至第 4 款",
    ),
]
GOOD = answer(
    point("虛擬資產服務商應確認客戶身分。", cite("S1", "應於確認客戶身分時，取得必要資訊"))
)
BAD = answer(point("虛擬資產服務商應確認客戶身分。", cite("S1", "這段文字不在條文中出現過")))


def ask(outputs, k=2):
    drafter, calls = scripted(outputs)
    index = FakeIndex(CHUNKS)
    state = run_ask(build_graph(index, drafter), QUESTION, k)
    return state, calls, index


def test_verified_answer_finishes_in_one_attempt():
    state, calls, index = ask([GOOD])
    assert state["verified"]
    assert state["attempts"] == 1
    assert state["problems"] == []
    assert index.queries == [(QUESTION, 2)]
    assert (state["input_tokens"], state["output_tokens"]) == (100, 20)


def test_sources_are_labeled_in_prompt():
    _, calls, _ = ask([GOOD])
    prompt = calls[0][-1].content
    assert "[S1] 【提供虛擬資產服務之事業或人員防制洗錢及打擊資恐辦法 第 9 條】" in prompt
    assert "[S2] 【金融機構防制洗錢辦法 第 3 條第 1 項第 1 款至第 4 款】" in prompt
    assert "<question>\nVASP 何時要確認客戶身分？\n</question>" in prompt


def test_two_parse_failures_are_flagged():
    state, calls, _ = ask([None, None])
    assert state["answer"] is None
    assert not state["verified"]
    assert state["attempts"] == 2
    assert len(calls) == 2
    assert state["problems"] == ["輸出格式錯誤，無法解析為指定結構"]


def test_failed_verification_retries_with_problems():
    state, calls, _ = ask([BAD, GOOD])
    assert state["verified"]
    assert state["attempts"] == 2
    retry = calls[1][-1].content
    assert "未通過驗證" in retry
    assert "不在該段落中" in retry
    assert (state["input_tokens"], state["output_tokens"]) == (200, 40)


def test_retry_failure_is_flagged():
    state, calls, _ = ask([BAD, BAD])
    assert not state["verified"]
    assert state["attempts"] == 2
    assert len(calls) == 2
    assert "不在該段落中" in state["problems"][0]


def test_parse_failure_is_retried():
    state, _, _ = ask([None, GOOD])
    assert state["verified"]
    assert state["attempts"] == 2


def test_refusal_is_verified():
    state, _, _ = ask([refusal()])
    assert state["verified"]
    assert not state["answer"].answerable
