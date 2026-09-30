from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from aml_reg_rag.models import Answer, LawArticle

QUESTION_SYSTEM = """你是虛擬資產服務商的法遵人員，要為法規查詢系統準備測試問題。
閱讀下方條文，寫一個實務上會提出、且能由這條條文回答的問題。

規則：
1. 以實務人員的口吻描述需求或情境，20 到 80 字。
2. 不得出現條號，不要照抄條文的句子。
3. 以繁體中文撰寫，只寫一個問題。"""

QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [("system", QUESTION_SYSTEM), ("human", "法規：{law_name}\n\n條文：\n{content}")]
)

JUDGE_SYSTEM = """你是法規問答的評審，判斷回答是否涵蓋每一個要點。
- 要點的意思在回答中有明確表達即算涵蓋，不要求用詞相同。
- 回答與要點矛盾，或只含糊帶過，視為未涵蓋。
- 依要點順序逐一輸出判斷，數量必須與要點相同，每個判斷附一句理由。"""

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", JUDGE_SYSTEM),
        ("human", "問題：{question}\n\n要點：\n{key_points}\n\n回答：\n{answer}"),
    ]
)


def question_messages(article: LawArticle) -> list[BaseMessage]:
    return QUESTION_PROMPT.format_messages(law_name=article.law_name, content=article.content)


def judge_messages(question: str, key_points: list[str], answer: Answer) -> list[BaseMessage]:
    points = "\n".join(f"{number}. {point}" for number, point in enumerate(key_points, start=1))
    statements = "\n".join(f"- {point.statement}" for point in answer.points)
    return JUDGE_PROMPT.format_messages(question=question, key_points=points, answer=statements)
