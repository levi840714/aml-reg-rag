import re
import unicodedata

_CJK = "㐀-䶿一-鿿豈-﫿"
_RUN = re.compile(rf"[{_CJK}]+|[a-z0-9][a-z0-9.\-]*")
_CJK_RUN = re.compile(rf"[{_CJK}]")


def tokenize(text: str) -> list[str]:
    """中文切重疊 bigram，英數字串整段保留；法規專有名詞不在一般斷詞字典中。"""
    tokens: list[str] = []
    for run in _RUN.findall(unicodedata.normalize("NFKC", text).lower()):
        if _CJK_RUN.match(run):
            if len(run) == 1:
                tokens.append(run)
            else:
                tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        else:
            tokens.append(run.rstrip(".-"))
    return tokens
