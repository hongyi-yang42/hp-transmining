"""Pydantic data models for pipeline records."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class OCRSpan(BaseModel):
    """One styled span within a text-layer line (pymupdf engine only).

    PaddleOCR emits plain text lines, so its blocks carry no span metadata
    (``OCRBlock.lines`` stays empty). The pymupdf engine populates it so the
    clean stage can separate footnote markers (small superscript digits) and
    note bodies (small-print spans) from body text by font size.
    """

    text: str
    size: float = Field(default=0.0, description="Font size in pt")
    flags: int = Field(default=0, description="PyMuPDF span flags bitfield")


class OCRLine(BaseModel):
    """One rendered line inside a text-layer block."""

    spans: list[OCRSpan] = Field(default_factory=list)


class OCRBlock(BaseModel):
    """One text block from OCR or text-layer extraction."""

    page: int = Field(description="Source PDF page number (1-indexed)")
    block_idx: int = Field(description="Block ordinal within the page")
    text: str
    bbox: list[float] = Field(description="[x0, y0, x1, y1] in PDF coordinate units")
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    engine: Literal["paddleocr", "pymupdf"]
    lines: list[OCRLine] = Field(
        default_factory=list,
        description="Line/span structure of the text layer (pymupdf only). "
        "Invariant when populated: text == '\\n'.join(''.join(span.text for "
        "span in line.spans) for line in lines).",
    )


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
    type: Literal["1:0", "0:1", "1:1", "1:2", "2:1", "1:3", "3:1", "2:2"]
    confidence: float = Field(ge=0.0, le=1.0)
    # Score gap between the chosen DP move's path total and the best competing
    # move at the same cell. Absolute cosine confidence is not discriminative
    # for e5 (all pairs >= 0.69); the margin is the ranking/review signal.
    # None for records that predate margin support or had no competitor.
    margin: float | None = Field(default=None, ge=0.0)
    method: Literal["vecalign_labse", "manual"]
    validated: bool = False
