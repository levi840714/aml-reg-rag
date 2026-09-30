from langchain_core.runnables import RunnableLambda

from aml_reg_rag.retrieval.rerank import ChunkScore, LlmReranker, RerankOutput
from tests.fakes import retrieved, scripted

QUESTION = "VASP 辦理臨時性交易多少金額要確認客戶身分？"
CANDIDATES = [
    retrieved("G0380252:3:p1:k1-4", "金融機構確認客戶身分措施", law_name="金融機構防制洗錢辦法"),
    retrieved("G0380268:3:p1:k1-4", "虛擬資產服務商確認客戶身分措施"),
    retrieved("G0380268:10", "紀錄保存"),
]


def output(**scores: int) -> RerankOutput:
    return RerankOutput(scores=[ChunkScore(label=k, relevance=v) for k, v in scores.items()])


def test_scores_map_labels_to_chunk_ids():
    model, calls = scripted([output(C1=0, C2=3, C3=1)])
    result = LlmReranker(model).score(QUESTION, CANDIDATES)
    assert result.scores == {
        "G0380252:3:p1:k1-4": 0,
        "G0380268:3:p1:k1-4": 3,
        "G0380268:10": 1,
    }
    assert result.fallback is None
    prompt = calls[0][-1].content
    assert "[C2] 【提供虛擬資產服務之事業或人員防制洗錢及打擊資恐辦法 第 3 條】" in prompt
    assert "適用對象不同" in calls[0][0].content


def test_missing_labels_score_zero_and_unknown_labels_are_ignored():
    model, _ = scripted([output(C2=2, C9=3)])
    result = LlmReranker(model).score(QUESTION, CANDIDATES)
    assert result.scores == {"G0380252:3:p1:k1-4": 0, "G0380268:3:p1:k1-4": 2, "G0380268:10": 0}


def test_unparsed_output_falls_back():
    model, _ = scripted([None])
    result = LlmReranker(model).score(QUESTION, CANDIDATES)
    assert (result.scores, result.fallback) == (None, "輸出無法解析")


def test_api_error_falls_back():
    def boom(messages):
        raise TimeoutError

    result = LlmReranker(RunnableLambda(boom)).score(QUESTION, CANDIDATES)
    assert (result.scores, result.fallback) == (None, "TimeoutError")


def test_labels_are_normalized_before_lookup():
    parsed = RerankOutput(
        scores=[
            ChunkScore(label="c1", relevance=1),
            ChunkScore(label="[C2]", relevance=3),
            ChunkScore(label=" C 3 ", relevance=2),
        ]
    )
    model, _ = scripted([parsed])
    result = LlmReranker(model).score(QUESTION, CANDIDATES)
    assert result.scores == {
        "G0380252:3:p1:k1-4": 1,
        "G0380268:3:p1:k1-4": 3,
        "G0380268:10": 2,
    }


def test_labels_with_trailing_text_only_use_the_chunk_number():
    parsed = RerankOutput(
        scores=[
            ChunkScore(label="C1 【洗錢防制法 第 5 條】", relevance=2),
            ChunkScore(label="C2: 第3條", relevance=3),
        ]
    )
    model, _ = scripted([parsed])
    result = LlmReranker(model).score(QUESTION, CANDIDATES)
    assert result.scores == {
        "G0380252:3:p1:k1-4": 2,
        "G0380268:3:p1:k1-4": 3,
        "G0380268:10": 0,
    }


def test_only_unknown_labels_fall_back_and_keep_tokens():
    model, _ = scripted([output(C9=3)])
    result = LlmReranker(model).score(QUESTION, CANDIDATES)
    assert (result.scores, result.fallback) == (None, "無有效評分")
    assert (result.input_tokens, result.output_tokens) != (0, 0)


def test_empty_scores_fall_back():
    model, _ = scripted([RerankOutput(scores=[])])
    result = LlmReranker(model).score(QUESTION, CANDIDATES)
    assert (result.scores, result.fallback) == (None, "無有效評分")
