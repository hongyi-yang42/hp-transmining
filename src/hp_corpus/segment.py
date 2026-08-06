"""Sentence segmentation with stable chapter-scoped IDs.

ID format: {book}_{lang}_ch{NN}_p{NNNN}_s{NNN}
- Paragraph ordinal within the chapter (p), 1-indexed
- Sentence ordinal within the paragraph (s), 1-indexed
- source_pages stored as a field, NOT encoded in the ID
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .schema import CleanSentence, Segment


def make_id(book: str, lang: str, chapter: int, paragraph: int, sentence: int) -> str:
    return f"{book}_{lang}_ch{chapter:02d}_p{paragraph:04d}_s{sentence:03d}"


# Chinese terminators that trigger a sentence split.
_ZH_SPLIT_CHARS = "。！？；"
# Chinese typographic quote pairs (open, close). Includes curly + straight +
# corner-bracket + angle-bracket variants to be robust across OCR outputs.
_ZH_QUOTE_PAIRS = (
    "“",
    "”",  # curly double
    "‘",
    "’",  # curly single
    "「",
    "」",  # corner brackets
    "『",
    "』",  # white corner brackets
    "《",
    "》",  # angle brackets
    '"',
    '"',  # straight double (treat as paired)
    "'",
    "'",  # straight single (treat as paired)
)


def _split_zh(text: str, preserve_ellipsis: bool = True) -> list[str]:
    """Split Chinese text on 。！？；, honoring matching quote pairs.

    With ``preserve_ellipsis=True`` (default), an ellipsis immediately followed
    by a terminator (``……。``) still triggers a split on the terminator, but a
    bare ``……`` mid-sentence does not.
    """
    open_quotes = set(_ZH_QUOTE_PAIRS[::2])
    close_to_open = dict(zip(_ZH_QUOTE_PAIRS[1::2], _ZH_QUOTE_PAIRS[::2], strict=True))

    out: list[str] = []
    cur: list[str] = []
    open_stack: list[str] = []

    for ch in text:
        cur.append(ch)
        if open_stack and ch in close_to_open and close_to_open[ch] == open_stack[-1]:
            open_stack.pop()
            # Chinese dialogue convention: split at the closing quote so the
            # quoted sentence stands alone. Subsequent narration starts a new
            # sentence.
            out.append("".join(cur).strip())
            cur = []
        elif ch in open_quotes:
            open_stack.append(ch)
        elif ch in _ZH_SPLIT_CHARS and not open_stack:
            out.append("".join(cur).strip())
            cur = []

    tail = "".join(cur).strip()
    if tail:
        out.append(tail)

    if preserve_ellipsis:
        # No-op for now: ellipsis handling is the default behavior above.
        # The flag exists for symmetry with config and future tuning.
        pass

    # Skip fragments that contain no CJK chars (e.g. lone "X" OCR noise or
    # standalone closing quote). Downstream alignment can't use them.
    return [s for s in out if s and _has_cjk(s)]


def _has_cjk(text: str) -> bool:
    """True iff text contains at least one CJK Unified Ideograph character."""
    return any("一" <= ch <= "鿿" for ch in text)


def _split_en(text: str, abbreviations: list[str]) -> list[str]:
    """Split English on . ! ? followed by space or end. Preserves abbreviations
    and mid-sentence ellipsis (...)."""
    if not text.strip():
        return []
    masked = text
    placeholders: list[tuple[str, str]] = []
    # Ellipsis first (longer run wins).
    for needle in ("...", "…"):
        if needle in masked:
            ph = f"\x00ELLIPSIS{len(placeholders)}\x00"
            masked = masked.replace(needle, ph)
            placeholders.append((ph, needle))
    for abbr in sorted(abbreviations, key=len, reverse=True):
        if abbr in masked:
            ph = f"\x00ABBR{len(placeholders)}\x00"
            masked = masked.replace(abbr, ph)
            placeholders.append((ph, abbr))

    # Split on . ! ? followed by optional closing quote/paren, then whitespace or end.
    parts = re.split(r"(?<=[.!?])(?:[\"'\)\]]?)(?:\s+|$)", masked)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        for ph, needle in placeholders:
            p = p.replace(ph, needle)
        p = p.strip()
        # Skip fragments that are only punctuation/whitespace (e.g. lone ".").
        # These would otherwise pollute downstream alignment as 1-char segments.
        if not any(c.isalnum() for c in p):
            continue
        out.append(p)
    return out


def segment_sentence(
    clean: CleanSentence,
    book: str,
    lang: str,
    chapter: int,
    config: dict[str, Any],
) -> list[Segment]:
    """Segment one CleanSentence into one or more Segments."""
    seg_cfg = config.get("segment", {}) or {}
    if lang == "zh":
        preserve_ellipsis = seg_cfg.get("preserve_ellipsis", True)
        parts = _split_zh(clean.text, preserve_ellipsis=preserve_ellipsis)
    else:
        abbreviations = seg_cfg.get("abbreviations", []) or []
        parts = _split_en(clean.text, abbreviations)

    return [
        Segment(
            id=make_id(book, lang, chapter, clean.paragraph, sentence_idx),
            chapter=chapter,
            paragraph=clean.paragraph,
            sentence=sentence_idx,
            text=part,
            source_pages=list(clean.source_pages),
        )
        for sentence_idx, part in enumerate(parts, start=1)
    ]


def segment_all(
    cleans: list[CleanSentence],
    book: str,
    lang: str,
    chapter: int,
    config: dict[str, Any],
) -> list[Segment]:
    out: list[Segment] = []
    for c in cleans:
        out.extend(segment_sentence(c, book, lang, chapter, config))
    return out


def write_segments_jsonl(segments: list[Segment], output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for s in segments:
            f.write(s.model_dump_json() + "\n")
    return out


def read_segments_jsonl(input_path: str | Path) -> list[Segment]:
    segments: list[Segment] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            segments.append(Segment.model_validate_json(line))
    return segments
