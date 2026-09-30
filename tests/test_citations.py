from aml_reg_rag.citations import normalize, verify
from aml_reg_rag.models import Answer
from tests.fakes import answer, cite, point, refusal

BODIES = {
    "S1": "虛擬資產服務商應於\n確認客戶身分時，取得必要資訊。",
    "S2": "其他條文內容，與本題無關的規定。",
}


def test_normalize_unifies_width_and_removes_whitespace():
    assert normalize("ＡＢ　c\n d（１）") == "ABcd(1)"


def test_valid_answer_passes():
    result = answer(point("應確認客戶身分。", cite("S1", "確認客戶身分時，取得必要資訊")))
    assert verify(result, BODIES) == []


def test_quote_across_line_break_passes():
    result = answer(point("應確認客戶身分。", cite("S1", "虛擬資產服務商應於確認客戶身分時")))
    assert verify(result, BODIES) == []


def test_unparsed_answer_is_a_format_error():
    assert verify(None, BODIES) == ["輸出格式錯誤，無法解析為指定結構"]


def test_answerable_without_points_fails():
    assert verify(answer(), BODIES) == ["可回答時至少需要一個論點"]


def test_point_without_citation_fails():
    assert verify(answer(point("沒有引用的論點")), BODIES) == ["論點 1 沒有引用"]


def test_unknown_source_fails():
    result = answer(point("論點", cite("S9", "確認客戶身分時，取得必要資訊")))
    assert verify(result, BODIES) == ["論點 1 引用的 S9 不在本次檢索結果中"]


def test_short_quote_fails():
    result = answer(point("論點", cite("S1", "確認客戶")))
    assert verify(result, BODIES) == ["論點 1 引用 S1 的原文少於 8 字"]


def test_punctuation_does_not_count_towards_quote_length():
    result = answer(point("論點", cite("S1", "。" * 8)))
    assert verify(result, BODIES) == ["論點 1 引用 S1 的原文少於 8 字"]


def test_quote_not_in_cited_source_fails():
    result = answer(point("論點", cite("S2", "確認客戶身分時，取得必要資訊")))
    assert verify(result, BODIES) == [
        "論點 1 引用 S2 的原文不在該段落中：「確認客戶身分時，取得必要資訊」"
    ]


def test_refusal_passes():
    assert verify(refusal(), BODIES) == []


def test_refusal_with_points_and_blank_missing_fails():
    result = Answer(
        answerable=False,
        points=[point("論點", cite("S1", "確認客戶身分時，取得必要資訊"))],
        missing=" ",
    )
    assert verify(result, BODIES) == ["拒答時不應有論點", "拒答時需說明段落缺少哪些規範"]
