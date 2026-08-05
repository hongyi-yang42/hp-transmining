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
"""

from __future__ import annotations

import re
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


@dataclass
class CleanResult:
    sentences: list[CleanSentence]
    footnotes: list[OCRBlock]
    chapter_title: str | None


def _strip_block_text(text: str) -> str:
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

    # 1. Page range filter
    in_range = [b for b in blocks if start_page <= b.page <= end_page]
    # Sort by page, then block_idx for stable reading order
    in_range.sort(key=lambda b: (b.page, b.block_idx))

    # 2-5. Per-block classification
    seen_headers: set[str] = set()
    chapter_title: str | None = None
    footnotes: list[OCRBlock] = []
    kept: list[OCRBlock] = []
    for b in in_range:
        text = _strip_block_text(b.text)
        if not text:
            continue

        # Footnote separation
        if footnote_markers and _FOOTNOTE_RE.match(text):
            footnotes.append(b.model_copy(update={"text": text}))
            continue

        # Header strip — exact match only. Substring match would over-match
        # (e.g. "Harry Potter" appears constantly in body text).
        matched_pat = next((pat for pat in header_patterns if text == pat), None)
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


def _assemble_paragraphs(
    blocks: list[OCRBlock],
    lang: str,
    dialogue_markers: list[str],
    merge_line_breaks: bool,  # noqa: ARG001 — kept for API symmetry; current impl always merges
) -> list[tuple[str, list[int]]]:
    """Group OCR blocks into paragraphs.

    Heuristic:
    - If the running paragraph ends with a sentence terminator, the next block
      starts a new paragraph.
    - If the running paragraph does NOT end with a terminator, the next block
      is treated as a line-wrap continuation (this also handles cross-page joins).
    - Indentation (4+ ASCII spaces or 2 full-width 　) or a leading dialogue
      marker forces a new paragraph regardless.
    """
    paragraphs: list[tuple[str, list[int]]] = []
    cur_text = ""
    cur_pages: list[int] = []

    open_q = dialogue_markers[0] if dialogue_markers else None

    for b in blocks:
        text = b.text.strip()
        if not text:
            continue

        new_para = False
        if not cur_text:
            new_para = True
        elif open_q and text.startswith(open_q):
            new_para = True
        elif text.startswith("　　") or text.startswith("    "):
            new_para = True
        elif _ends_with_terminator(cur_text):
            new_para = True

        if new_para and cur_text:
            paragraphs.append((cur_text, cur_pages))
            cur_text = ""
            cur_pages = []

        if cur_text and lang == "en":
            if cur_text.endswith("-") and text and text[0].islower():
                cur_text = cur_text[:-1] + text
            else:
                cur_text += " " + text
        elif cur_text:
            cur_text += text
        else:
            cur_text = text
        cur_pages.append(b.page)

    if cur_text:
        paragraphs.append((cur_text, cur_pages))

    out: list[tuple[str, list[int]]] = []
    for text, pages in paragraphs:
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            out.append((text, pages))
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
