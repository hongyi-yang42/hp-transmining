"""Pydantic data models for pipeline records."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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


_SEGMENT_ID_LANG_RE = re.compile(r"^[a-z0-9]+_([a-z]+)_ch\d{2}_p\d{4}_s\d{3}$")


def _lang_from_segment_ids(ids: Any) -> str:
    """Language code carried by the first recognizable segment ID, or ""."""
    if isinstance(ids, list):
        for sid in ids:
            if isinstance(sid, str):
                m = _SEGMENT_ID_LANG_RE.match(sid)
                if m:
                    return m.group(1)
    return ""


class Alignment(BaseModel):
    """One cross-language alignment record for an arbitrary src→tgt pair.

    ``src``/``tgt`` are positional sides (the DP writer puts the first
    argument on ``src``); the actual languages live in ``src_lang`` /
    ``tgt_lang`` and in the segment IDs themselves. A ``mode="before"``
    validator still accepts records written by the pre-rename serializer,
    which hard-coded the field names ``en``/``zh`` regardless of language
    and the method value ``vecalign_labse``.
    """

    align_id: str = Field(pattern=r"^a\d+$")
    src: list[str] = Field(description="Source-side Segment IDs")
    tgt: list[str] = Field(description="Target-side Segment IDs")
    # Empty string only on legacy records whose language could not be
    # derived from segment IDs (e.g. an empty 1:0 side); the DP writer
    # always fills both.
    src_lang: str = Field(default="", description="Source language code, e.g. 'de'")
    tgt_lang: str = Field(default="", description="Target language code, e.g. 'zh'")
    # Grouping types up to arity 5 (max_group): the DP emits any
    # di:dj with 1 < di + dj <= max_group + 1; the Literal is the superset
    # for the largest supported cap (AlignmentConfig.max_group, default 5).
    type: Literal[
        "1:0",
        "0:1",
        "1:1",
        "1:2",
        "2:1",
        "1:3",
        "3:1",
        "2:2",
        "1:4",
        "4:1",
        "2:3",
        "3:2",
        "1:5",
        "5:1",
        "2:4",
        "4:2",
        "3:3",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    # Score gap between the chosen DP move's path total and the best competing
    # move at the same cell. Absolute cosine confidence is not discriminative
    # for e5 (all pairs >= 0.69); the margin is the ranking/review signal.
    # None for records that predate margin support or had no competitor.
    margin: float | None = Field(default=None, ge=0.0)
    # "embedding_dp": machine record from this module's DP (regardless of the
    # embedding model — provenance lives in the run manifest and cache meta).
    # "manual": reserved for records a human actually created by hand; the
    # machine writer must never assign it.
    method: Literal["embedding_dp", "manual"]
    # Machine flag: confidence below AlignmentConfig.review_threshold. Such
    # records stay method="embedding_dp" and are queued for human review —
    # they are NOT manual alignments.
    needs_review: bool = False
    validated: bool = False

    @model_validator(mode="before")
    @classmethod
    def _compat_legacy_fields(cls, data: Any) -> Any:
        """Read records from the pre-rename serializer: `en`/`zh` field
        names (used for every language pair) and `vecalign_labse` method;
        derive src/tgt languages from segment IDs when absent."""
        if not isinstance(data, dict):
            return data
        if "src" not in data and "en" in data:
            data["src"] = data.pop("en")
        if "tgt" not in data and "zh" in data:
            data["tgt"] = data.pop("zh")
        if data.get("method") == "vecalign_labse":
            data["method"] = "embedding_dp"
        if not data.get("src_lang"):
            data["src_lang"] = _lang_from_segment_ids(data.get("src"))
        if not data.get("tgt_lang"):
            data["tgt_lang"] = _lang_from_segment_ids(data.get("tgt"))
        return data
