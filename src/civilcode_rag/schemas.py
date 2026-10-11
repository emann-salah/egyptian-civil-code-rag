from __future__ import annotations

from pydantic import BaseModel


class ArticleRecord(BaseModel):
    article_number: int
    book: str | None = None
    chapter: str | None = None
    section: str | None = None
    topic: str | None = None
    text_ar: str
    text_en: str = ""
    is_repealed: bool = False
    source_page: int | None = None
    citation: str