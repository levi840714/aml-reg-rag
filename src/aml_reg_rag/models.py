from datetime import date

from pydantic import BaseModel, Field

LAW_ARTICLE_URL = "https://law.moj.gov.tw/LawClass/LawSingle.aspx?pcode={pcode}&flno={article_no}"


def article_url(pcode: str, article_no: str) -> str:
    return LAW_ARTICLE_URL.format(pcode=pcode, article_no=article_no)


class LawArticle(BaseModel):
    pcode: str
    law_name: str
    law_level: str
    law_modified_date: date
    chapter: str | None
    article_no: str
    content: str

    @property
    def article_key(self) -> str:
        return f"{self.pcode}:{self.article_no}"


class Chunk(BaseModel):
    chunk_id: str
    article: LawArticle
    locator: str
    body: str

    @property
    def header(self) -> str:
        position = f"第 {self.article.article_no} 條"
        if self.locator:
            position = f"{position} {self.locator}"
        parts = [self.article.law_name, self.article.chapter, position]
        return f"【{'｜'.join(part for part in parts if part)}】"

    @property
    def page_content(self) -> str:
        return f"{self.header}\n{self.body}"


class RetrievedChunk(BaseModel):
    chunk_id: str
    article_key: str
    law_name: str
    article_no: str
    locator: str
    source_url: str
    body: str
    score: float

    @property
    def title(self) -> str:
        position = f"第 {self.article_no} 條"
        return f"{self.law_name} {position}{self.locator}"


class Citation(BaseModel):
    source: str = Field(description="引用段落的標籤，例如 S2")
    quote: str = Field(description="從該段落逐字複製的原文，至少 8 個字")


class Point(BaseModel):
    statement: str = Field(description="一個論點，第一個論點直接回答問題")
    citations: list[Citation] = Field(description="支持此論點的引用，至少一個")


class Answer(BaseModel):
    answerable: bool = Field(description="提供的段落是否足以回答問題")
    points: list[Point] = Field(description="回答的論點；無法回答時為空")
    missing: str | None = Field(default=None, description="無法回答時，說明段落缺少哪些規範")
