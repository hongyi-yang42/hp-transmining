"""Pydantic data models for pipeline records."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class OCRBlock(BaseModel):
    """One text block from OCR or text-layer extraction."""

    page: int = Field(description="Source PDF page number (1-indexed)")
    block_idx: int = Field(description="Block ordinal within the page")
    text: str
    bbox: list[float] = Field(description="[x0, y0, x1, y1] in PDF coordinate units")
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    engine: Literal["paddleocr", "pymupdf"]


class CleanSentence(BaseModel):
    """One cleaned text unit after removing headers/page numbers/footnotes."""

    page: int
    paragraph: int = Field(description="Paragraph ordinal within the chapter")
    text: str
    source_pages: list[int] = Field(
        default_factory=list,
        description="All source PDF pages this unit draws from (for cross-page joins)",
    )
    is_footnote: bool = False


class Segment(BaseModel):
    """One sentence with a stable chapter-scoped ID."""

    id: str = Field(
        description="Format: {book}_{lang}_ch{NN}_p{NNNN}_s{NNN}",
        pattern=r"^[a-z0-9]+_[a-z]+_ch\d{2}_p\d{4}_s\d{3}$",
    )
    chapter: int
    paragraph: int
    sentence: int = Field(description="Sentence ordinal within its paragraph, 1-indexed")
    text: str
    source_pages: list[int] = Field(default_factory=list)


class Alignment(BaseModel):
    """One EN-ZH alignment record."""

    align_id: str = Field(pattern=r"^a\d+$")
    en: list[str] = Field(description="List of English Segment IDs")
    zh: list[str] = Field(description="List of Chinese Segment IDs")
    type: Literal["1:0", "0:1", "1:1", "1:2", "2:1", "2:2"]
    confidence: float = Field(ge=0.0, le=1.0)
    method: Literal["vecalign_labse", "manual"]
    validated: bool = False
