from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ArticleRecord(BaseModel):
    article_number: int
    book: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    topic: Optional[str] = None
    text_ar: str
    text_en: str = ""
    is_repealed: bool = False
    source_page: Optional[int] = None
    citation: str