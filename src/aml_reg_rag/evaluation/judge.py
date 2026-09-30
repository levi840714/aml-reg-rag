from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from aml_reg_rag.evaluation import prompts
from aml_reg_rag.llm import structured_model, token_usage
from aml_reg_rag.models import Answer


class KeyPointVerdict(BaseModel):
    covered: bool = Field(description="回答是否涵蓋此要點")
    reason: str = Field(description="一句話說明理由")


class JudgeOutput(BaseModel):
    verdicts: list[KeyPointVerdict] = Field(description="依要點順序的判斷，數量與要點相同")


@dataclass(frozen=True)
class JudgeResult:
    verdicts: list[KeyPointVerdict] | None
    input_tokens: int
    output_tokens: int

    @property
    def coverage(self) -> float | None:
        if not self.verdicts:
            return None
        return sum(verdict.covered for verdict in self.verdicts) / len(self.verdicts)


def judge_answer(
    judge: Runnable[Any, Any], question: str, key_points: list[str], answer: Answer
) -> JudgeResult:
    result = judge.invoke(prompts.judge_messages(question, key_points, answer))
    input_tokens, output_tokens = token_usage(result["raw"])
    parsed = result["parsed"]
    valid = isinstance(parsed, JudgeOutput) and len(parsed.verdicts) == len(key_points)
    return JudgeResult(
        verdicts=parsed.verdicts if valid else None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def make_judge(model: str, timeout: float) -> Runnable[Any, Any]:
    return structured_model(model, JudgeOutput, timeout=timeout, include_raw=True)
