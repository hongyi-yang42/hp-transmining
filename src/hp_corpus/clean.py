"""Cleaning pipeline: OCR JSONL → plain UTF-8 text + separated footnotes.

Rules (applied in order):
1. Drop pages outside configured chapter range.
2. Drop recurring header_patterns — but keep the first occurrence as chapter title.
3. Drop page-number blocks (short digit-only text at top/bottom of page).
4. Drop decorative glyphs (single-char blocks, asterisks separators).
5. Separate footnotes (①②③-prefixed blocks) into _notes.jsonl.
6. Merge line breaks within paragraph; detect paragraph boundaries.
7. Cross-page rejoin when previous page's last block lacks terminator.
8. Conservative normalization (whitespace; no name/punctuation auto-correction).

Paragraph boundary detection uses **bbox x-coordinate as primary signal**:
a block whose left edge is significantly indented relative to the page's
standard left margin starts a new paragraph. Sentence terminators are NOT
used as paragraph boundaries — every Chinese sentence ends with 。, so a
terminator-based heuristic fragments every paragraph. Terminator info is
only consulted for cross-page rejoin (deciding whether a page break ends
a paragraph or wraps).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import CleanSentence, OCRBlock

# Sentence terminators used for cross-page rejoin decisions.
_TERMINATORS_ZH = "。！？；”’》）】》\"'"
_TERMINATORS_EN = ".!?\";'…)]"
_TERMINATORS = _TERMINATORS_ZH + _TERMINATORS_EN

# Decorative single-char blocks (asterisks, ornaments, dots).
_DECORATIVE = {"*", "·", "•", "◇", "◆", "■", "□", "★", "☆", "✦", "✧", "~", "〜", "§", "||"}
_FOOTNOTE_RE = re.compile(r"^[①②③④⑤⑥⑦⑧⑨⑩]")
_PURE_DIGITS_RE = re.compile(r"^\d{1,4}$")

# A block is "indented" if its bbox x0 exceeds the page's body-text cluster
# by a meaningful margin. PaddleOCR's body-text x0 has ~30px jitter on a
# 300-DPI render, and real paragraph first-line indents sit ~90–110px right
# of body. We use gap-based adaptive detection (see _indent_threshold);
# this constant is only a fallback when the gap analysis fails.
_INDENT_X_FALLBACK_PX = 50


@dataclass
class CleanResult:
    sentences: list[CleanSentence]
    footnotes: list[OCRBlock]
    chapter_title: str | None


def _strip_block_text(text: str) -> str:
    # Drop C0/C1 control chars — PyMuPDF sometimes wraps page numbers in
    # private-use markers like "\x91 17 \x91" as decorative side flourishes.
    # Without this, the page-number regex sees "\x9117\x91" and misses it.
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return text.replace(" ", " ").strip()


def _is_decorative(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return all(c in _DECORATIVE or c.isspace() for c in t)


def _looks_like_page_number(text: str) -> bool:
    return bool(_PURE_DIGITS_RE.match(text.strip()))


def _ends_with_terminator(text: str) -> bool:
    if not text:
        return True
    return text.rstrip()[-1] in _TERMINATORS


def clean_blocks(
    blocks: list[OCRBlock],
    config: dict[str, Any],
) -> CleanResult:
    """Run the full cleaning pipeline on a list of OCRBlocks."""
    clean_cfg = config.get("clean", {}) or {}
    chapter_cfg = config["chapter"]
    start_page = chapter_cfg["start_page"]
    end_page = chapter_cfg["end_page"]
    lang = config["lang"]

    header_patterns = [h for h in clean_cfg.get("header_patterns", []) or [] if h]
    footnote_markers = clean_cfg.get("footnote_markers", []) or []
    pd_cfg = clean_cfg.get("paragraph_detection", {}) or {}
    dialogue_markers = pd_cfg.get("dialogue_markers", []) or []
    merge_line_breaks = clean_cfg.get("merge_line_breaks", True)
    # OCR-engine noise filters (PaddleOCR occasionally emits stray detections
    # at extreme bbox positions or with very low confidence; these pollute
    # downstream alignment).
    min_confidence = clean_cfg.get("min_confidence", 0.4)

    # 1. Page range filter
    in_range = [b for b in blocks if start_page <= b.page <= end_page]
    # Sort by page, then block_idx for stable reading order
    in_range.sort(key=lambda b: (b.page, b.block_idx))

    # 1b. Drop blocks with implausible bbox (extreme x0 outlier — these are
    # almost always OCR false positives like stray Latin letters or digits
    # detected at the page margin). A block is an outlier if its x0 is more
    # than 3x the page's median x0.
    in_range = _drop_bbox_outliers(in_range)

    # 2-5. Per-block classification
    seen_headers: set[str] = set()
    chapter_title: str | None = None
    footnotes: list[OCRBlock] = []
    kept: list[OCRBlock] = []
    for b in in_range:
        text = _strip_block_text(b.text)
        if not text:
            continue

        # Confidence filter — drop low-confidence OCR noise (default <0.4).
        # PyMuPDF text-layer blocks always carry confidence=1.0 so this only
        # affects PaddleOCR output.
        if b.confidence < min_confidence:
            continue

        # Footnote separation
        if footnote_markers and _FOOTNOTE_RE.match(text):
            footnotes.append(b.model_copy(update={"text": text}))
            continue

        # Header strip — exact match, with a whitespace-insensitive fallback
        # so typography variants like "THE  BOY  WHO  LIVED" (double-spaced
        # running header) and "C H A P T E R  O N E" (letter-spaced chapter
        # title) still match their normal-space pattern.
        matched_pat = next(
            (
                pat
                for pat in header_patterns
                if text == pat or text.replace(" ", "") == pat.replace(" ", "")
            ),
            None,
        )
        if matched_pat is not None:
            if matched_pat not in seen_headers:
                seen_headers.add(matched_pat)
                if chapter_title is None:
                    chapter_title = text
            continue

        # Page numbers
        if _looks_like_page_number(text):
            continue

        # Decorative
        if _is_decorative(text):
            continue

        # Single-char alphabetic blocks — PyMuPDF sometimes emits a decorative
        # drop cap as its own block (e.g. "M " for the M in "Mr."). Downstream
        # assembly would then prepend it to the wrong paragraph. Drop it.
        if len(text) == 1 and text.isalpha():
            continue

        kept.append(b.model_copy(update={"text": text}))

    # 6-7. Paragraph assembly + cross-page rejoin
    paragraphs = _assemble_paragraphs(kept, lang, dialogue_markers, merge_line_breaks)

    sentences: list[CleanSentence] = []
    for para_idx, (para_text, source_pages) in enumerate(paragraphs, start=1):
        if not para_text.strip():
            continue
        sentences.append(
            CleanSentence(
                page=min(source_pages) if source_pages else 0,
                paragraph=para_idx,
                text=para_text,
                source_pages=sorted(set(source_pages)),
            )
        )

    return CleanResult(sentences=sentences, footnotes=footnotes, chapter_title=chapter_title)


def _drop_bbox_outliers(blocks: list[OCRBlock]) -> list[OCRBlock]:
    """Drop blocks whose x0 is an extreme outlier within their page.

    PaddleOCR occasionally emits stray single-character detections at the
    page edge or in unexpected positions (e.g. misreading a margin mark as
    "X" or "食"). These have bbox x0 values 3–20× larger than typical body
    text. Drop any block whose x0 exceeds 3× the page's median x0.
    """
    by_page: dict[int, list[float]] = defaultdict(list)
    for b in blocks:
        if len(b.bbox) >= 4 and b.bbox[2] > b.bbox[0] > 0:
            by_page[b.page].append(float(b.bbox[0]))
    medians: dict[int, float] = {}
    for page, xs in by_page.items():
        if xs:
            medians[page] = sorted(xs)[len(xs) // 2]
    return [
        b
        for b in blocks
        if b.page not in medians
        or not (len(b.bbox) >= 4 and b.bbox[2] > b.bbox[0] > 0)
        or b.bbox[0] <= 3.0 * medians[b.page]
    ]


def _page_left_margins(blocks: list[OCRBlock]) -> dict[int, float]:
    """Per-page body-text left margin = 10th-percentile of block x0.

    Using p10 (rather than min) is robust to a few stray outer-margin blocks
    like page numbers or running titles that may sit further left than body
    text. Returns page → x0 of typical body-text left edge.
    """
    by_page: dict[int, list[float]] = defaultdict(list)
    for b in blocks:
        if len(b.bbox) >= 4:
            by_page[b.page].append(float(b.bbox[0]))
    margins: dict[int, float] = {}
    for page, xs in by_page.items():
        if not xs:
            continue
        xs_sorted = sorted(xs)
        # p10 with interpolation guard
        k = max(0, min(len(xs_sorted) - 1, int(round(0.10 * (len(xs_sorted) - 1)))))
        margins[page] = xs_sorted[k]
    return margins


def _indent_thresholds(blocks: list[OCRBlock]) -> dict[int, float]:
    """Per-page adaptive indent threshold via largest-gap analysis.

    For each page, sort the unique x0 values and find the largest gap among
    the lower 80% (body cluster). The midpoint of that gap is the natural
    separator between body-text jitter and real paragraph indents. Falls
    back to ``margin + _INDENT_X_FALLBACK_PX`` when no gap ≥ 30px exists
    (page has no clear indent cluster, e.g. title pages).
    """
    margins = _page_left_margins(blocks)
    by_page: dict[int, list[float]] = defaultdict(list)
    for b in blocks:
        if len(b.bbox) >= 4 and b.bbox[2] > b.bbox[0] > 0:
            by_page[b.page].append(float(b.bbox[0]))

    thresholds: dict[int, float] = {}
    for page, xs in by_page.items():
        margin = margins.get(page)
        if margin is None:
            continue
        unique = sorted(set(xs))
        # Restrict to x0 values not too far from body (within 2x margin) so
        # far outliers don't pollute the gap analysis.
        body_cluster = [x for x in unique if x <= 2.5 * margin]
        if len(body_cluster) < 2:
            thresholds[page] = margin + _INDENT_X_FALLBACK_PX
            continue
        # Find largest gap between consecutive body-cluster values
        max_gap = 0.0
        gap_start = 0.0
        for i in range(len(body_cluster) - 1):
            g = body_cluster[i + 1] - body_cluster[i]
            if g > max_gap:
                max_gap = g
                gap_start = body_cluster[i]
        if max_gap >= 30.0:
            # Midpoint of the largest gap is the natural threshold
            thresholds[page] = gap_start + max_gap / 2
        else:
            thresholds[page] = margin + _INDENT_X_FALLBACK_PX
    return thresholds


def _is_indented(block: OCRBlock, threshold: float | None) -> bool:
    """True iff this block's left edge exceeds the page's indent threshold.

    Falls back to leading-whitespace detection when bbox is unavailable or
    unusable (e.g. text-layer blocks with [0,0,0,0]).
    """
    if threshold is not None and len(block.bbox) >= 4 and block.bbox[2] > block.bbox[0] > 0:
        return float(block.bbox[0]) > threshold
    # Whitespace fallback — works for English text layer where blocks carry
    # their leading spaces.
    return block.text.startswith("　　") or block.text.startswith("    ")


def _assemble_paragraphs(
    blocks: list[OCRBlock],
    lang: str,
    dialogue_markers: list[str],
    merge_line_breaks: bool,  # noqa: ARG001 — kept for API symmetry; current impl always merges
) -> list[tuple[str, list[int]]]:
    """Group OCR blocks into paragraphs.

    Boundary signals (highest priority first):
    1. Indentation — block's left edge is noticeably right of the page's
       standard margin (paragraph first-line indent).
    2. Dialogue open-quote at start of block AND running paragraph ends with
       a terminator — typical dialogue-line start.
    3. (English only) Running paragraph ends with a terminator — text-layer
       blocks typically correspond 1:1 to paragraphs, so a terminator signals
       the end of one. NOT used for Chinese because every sentence ends with
       。, which would fragment every paragraph.
    4. Otherwise: continuation of current paragraph (handles line wraps and
       cross-page rejoins).
    """
    paragraphs: list[tuple[str, list[int]]] = []
    cur_text = ""
    cur_pages: list[int] = []

    open_q = dialogue_markers[0] if dialogue_markers else None
    indent_thresholds = _indent_thresholds(blocks)
    # English source uses PyMuPDF text layer where each block is typically
    # already a paragraph; terminator signals paragraph end. Chinese source
    # is PaddleOCR line-by-line, so terminator would over-fragment.
    use_terminator_signal = lang == "en"

    for b in blocks:
        text = b.text.strip()
        if not text:
            continue

        threshold = indent_thresholds.get(b.page)
        new_para = False
        if not cur_text:
            new_para = True
        elif _is_indented(b, threshold):
            new_para = True
        elif open_q and text.startswith(open_q) and _ends_with_terminator(cur_text):
            # New dialogue line: leading quote + previous sentence closed.
            new_para = True
        elif use_terminator_signal and _ends_with_terminator(cur_text):
            # English text-layer: each block ≈ paragraph.
            new_para = True

        if new_para and cur_text:
            paragraphs.append((cur_text, cur_pages))
            cur_text = ""
            cur_pages = []

        # German PaddleOCR is line-by-line like Chinese, but German is
        # whitespace-delimited and uses end-of-line hyphenation. Joining
        # lines without a space produces "warenstolz"; not repairing
        # hyphen+lowercase leaves "Grun-nings". Chinese needs direct
        # concatenation (no inter-word space).
        if cur_text:
            if cur_text.endswith("-") and text and text[0].islower():
                cur_text = cur_text[:-1] + text
            elif lang in ("en", "de"):
                cur_text += " " + text
            else:
                cur_text += text
        else:
            cur_text = text
        cur_pages.append(b.page)

    if cur_text:
        paragraphs.append((cur_text, cur_pages))

    out: list[tuple[str, list[int]]] = []
    for text, pages in paragraphs:
        text_clean = re.sub(r"\s+", " ", text).strip()
        if text_clean:
            out.append((text_clean, list(pages)))
    return out


def write_clean_outputs(
    result: CleanResult,
    output_dir: str | Path,
    book: str,
    lang: str,
    chapter: int,
) -> dict[str, Path]:
    """Write cleaned JSONL + plain text + footnote notes. Returns paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / f"{book}_{lang}_ch{chapter:02d}.jsonl"
    txt_path = out / f"{book}_{lang}_ch{chapter:02d}.txt"
    notes_path = out / f"{book}_{lang}_ch{chapter:02d}_notes.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in result.sentences:
            f.write(s.model_dump_json() + "\n")

    # Plain text: paragraphs separated by blank line. No chapter title heading —
    # the title is metadata, not body text.
    with open(txt_path, "w", encoding="utf-8") as f:
        for s in result.sentences:
            f.write(s.text + "\n\n")

    with open(notes_path, "w", encoding="utf-8") as f:
        for note in result.footnotes:
            f.write(note.model_dump_json() + "\n")

    return {"jsonl": jsonl_path, "text": txt_path, "notes": notes_path}
