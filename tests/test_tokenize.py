from aml_reg_rag.retrieval.tokenize import tokenize


def test_chinese_runs_become_overlapping_bigrams():
    assert tokenize("臨時性交易") == ["臨時", "時性", "性交", "交易"]


def test_single_chinese_character_is_kept():
    assert tokenize("法 規") == ["法", "規"]


def test_alphanumeric_runs_are_kept_whole_and_lowercased():
    assert tokenize("FATF R.16 與第 5-1 條") == ["fatf", "r.16", "與第", "5-1", "條"]


def test_full_width_characters_are_normalized():
    assert tokenize("ＶＡＳＰ　第５條") == ["vasp", "第", "5", "條"]


def test_trailing_punctuation_is_stripped():
    assert tokenize("R.16.") == ["r.16"]
