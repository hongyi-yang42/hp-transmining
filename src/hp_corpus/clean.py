"""Cleaning pipeline: OCR JSONL → plain UTF-8 text + separated footnotes.

Rules (applied in order):
1. Drop pages outside configured chapter range.
2. Drop recurring header_patterns — but keep the first occurrence as chapter title.
3. Drop page-number blocks (short digit-only text at top/bottom of page).
4. Drop decorative glyphs (single-char blocks, asterisks separators).
5. Separate footnotes into _notes.jsonl — ①②③-prefixed blocks (scanned
   sources) and, when ``clean.footnote_spans`` is configured, superscript
   digit markers + small-print note bodies identified via span font sizes
   (born-digital text layers).
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
from .text_split import split_concat

# Sentence terminators used for cross-page rejoin decisions.
_TERMINATORS_ZH = "。！？；”’》）】》\"'"
# German guillemets included: a dialogue line ends »…Satz.«, so paragraph
# assembly (new dialogue line after a terminator-ending line) must see « as a
# terminator-ending text.
_TERMINATORS_EN = ".!?\";'…)]»«"
_TERMINATORS = _TERMINATORS_ZH + _TERMINATORS_EN

# When ``clean.concat_split`` is enabled, a token must appear at least this
# many times in the chapter to enter the validation wordlist. The filter
# keeps one-off concat artifacts out of the wordlist (so the splitter
# actually tries to split them) while still including real one-off words
# that happen to be ≥2 — the splitter's morphology fallback handles the
# genuinely rare ones.
_CONCAT_WORDLIST_MIN_FREQ = 2

# Decorative single-char blocks (asterisks, ornaments, dots).
_DECORATIVE = {"*", "·", "•", "◇", "◆", "■", "□", "★", "☆", "✦", "✧", "~", "〜", "§", "||"}
_FOOTNOTE_RE = re.compile(r"^[①②③④⑤⑥⑦⑧⑨⑩]")
_PURE_DIGITS_RE = re.compile(r"^\d{1,4}$")

# Regexes used by the optional concat-split pass.
# _TOKEN_RE splits a string into alternating (token, whitespace) pairs so we
# can re-join with original whitespace preserved. _WORD_TOKEN_RE extracts
# word-only tokens (incl. German umlauts) for wordlist-building. _WORD_PUNCT_RE
# captures leading punctuation, word core, trailing punctuation — used to
# apply the splitter while keeping surrounding punctuation intact.
_TOKEN_RE = re.compile(r"(\s+)")
_WORD_TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")
_WORD_PUNCT_RE = re.compile(r"^([^\wäöüßÄÖÜ]*)([\wäöüßÄÖÜ]+?)([^\wäöüßÄÖÜ]*)$")


def _collapse_ws(text: str) -> str:
    """Remove all whitespace (incl. U+3000) for typography-insensitive compare."""
    return re.sub(r"\s+", "", text)

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
    # Drop C1 control chars only — PyMuPDF sometimes wraps page numbers in
    # private-use markers like "\x91 17 \x91" as decorative side flourishes.
    # Without this, the page-number regex sees "\x9117\x91" and misses it.
    # (C0 control chars — \n, \t, etc. — are handled by the whitespace
    # normalize below; stripping them would concatenate adjacent words.)
    text = re.sub(r"[\x7f-\x9f]", "", text)
    # Preserve a leading ideographic-space run (CJK first-line indent) as a
    # single U+3000 prefix. Born-digital Chinese text layers mark paragraph
    # starts typographically with U+3000 rather than (or in addition to) a
    # bbox x0 shift; the paragraph assembler consumes the prefix as an
    # indent signal and the final paragraph text strips it again.
    m = re.match(r"　+", text)
    prefix = m.group(0)[0] if m else ""
    # Repair end-of-line hyphenation *inside* a block before the whitespace
    # normalize flattens the newline: letter + "-" + line break + lowercase
    # continuation is a wrapped compound ("Kran-\nkenflügel" →
    # "Krankenflügel"). The block-boundary repair in _assemble_paragraphs
    # cannot see these — it only receives already-flattened text. Same
    # lowercase-continuation guard as there; capitalized continuations stay
    # hyphenated by policy.
    text = re.sub(r"([A-Za-zÄÖÜäöüß])-[\n\r]+([a-zäöüß])", r"\1\2", text)
    # Normalize any whitespace run (incl. \n, \t, \xa0) to a single space.
    # Born-digital PDF text layers frequently separate words within a block
    # with \n; collapsing those to nothing (the old behavior) joined the
    # words into one concat artifact.
    text = re.sub(r"\s+", " ", text).strip()
    return prefix + text


def _build_inline_wordlist(blocks: list[OCRBlock]) -> set[str]:
    """Build a validation wordlist from this chapter\'s own blocks.

    Frequency-filtered: a token must appear at least
    :data:`_CONCAT_WORDLIST_MIN_FREQ` times to enter the wordlist. This
    keeps one-off text-layer concat artifacts (e.g. ``beimersten``,
    ``Endeder``) out of the wordlist so the splitter actually tries to
    split them, while still capturing the chapter\'s common vocabulary.
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    for b in blocks:
        counts.update(_WORD_TOKEN_RE.findall(b.text))
    wl: set[str] = set()
    for tok, n in counts.items():
        if n >= _CONCAT_WORDLIST_MIN_FREQ:
            wl.add(tok)
            wl.add(tok.lower())
    # Function words (articles, prepositions, pronouns, auxiliaries) are
    # intentionally NOT added here — they are already baked into
    # text_split.FUNCTION_WORDS as split indicators, so listing them in
    # the wordlist would be redundant.
    return wl


def _split_concat_in_text(text: str, wordlist: set[str]) -> tuple[str, int]:
    """Apply :func:`split_concat` to every non-whitespace run in ``text``.

    Leading/trailing punctuation on each token is preserved verbatim.
    Returns ``(new_text, n_splits)``. When ``n_splits == 0`` the text is
    unchanged.
    """
    pieces = _TOKEN_RE.split(text)
    n_splits = 0
    for i, piece in enumerate(pieces):
        if i % 2 == 1 or not piece:
            continue  # whitespace separator or empty
        m = _WORD_PUNCT_RE.match(piece)
        if not m:
            continue
        leading, core, trailing = m.groups()
        sub = split_concat(core, wordlist)
        if len(sub) > 1:
            n_splits += len(sub) - 1
            pieces[i] = leading + " ".join(sub) + trailing
    return "".join(pieces), n_splits


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
    # Optional: split text-layer concat artifacts (e.g. ``beimersten``,
    # ``Endeder``) typical of born-digital PDFs. Defaults to off — only
    # needed when the source PDF's text layer has known concat issues.
    concat_split = bool(clean_cfg.get("concat_split", False))

    # 1. Page range filter
    in_range = [b for b in blocks if start_page <= b.page <= end_page]
    # Sort by page, then block_idx for stable reading order
    in_range.sort(key=lambda b: (b.page, b.block_idx))

    # 1a. Span-based footnote separation (text-layer sources with small-print
    # notes). Runs before normalization so it sees raw span texts; the note
    # blocks it emits are already normalized and go straight to the notes
    # output, bypassing the per-block classification below.
    footnotes: list[OCRBlock] = []
    span_note_cfg = clean_cfg.get("footnote_spans") or None
    if span_note_cfg:
        in_range, span_notes = _separate_span_footnotes(in_range, span_note_cfg)
        footnotes.extend(span_notes)

    # 1b. Drop PaddleOCR blocks with implausible bbox (extreme x0 outlier —
    # these are almost always OCR false positives like stray Latin letters or
    # digits detected at the page margin). A block is an outlier if its x0 is
    # more than 3x the page's median PaddleOCR x0. Text-layer (pymupdf)
    # blocks never go through this filter — see _drop_bbox_outliers.
    in_range = _drop_bbox_outliers(in_range)

    # 1c. Whitespace-normalize every block in-range. We do this once here
    # so both the wordlist builder and the classifier see the same text.
    in_range = [
        b.model_copy(update={"text": t})
        for b in in_range
        if (t := _strip_block_text(b.text))
    ]

    # 1d. Optional concat-split pass. Build the wordlist from this chapter's
    # own normalized blocks (frequency-filtered), then apply the splitter
    # token-by-token. The pass is conservative — see text_split.py — so
    # real compounds (Apfelbaum) survive untouched.
    if concat_split:
        wordlist = _build_inline_wordlist(in_range)
        split_blocks: list[OCRBlock] = []
        for b in in_range:
            new_text, _ = _split_concat_in_text(b.text, wordlist)
            split_blocks.append(b.model_copy(update={"text": new_text}))
        in_range = split_blocks

    # 2-5. Per-block classification
    seen_headers: set[str] = set()
    chapter_title: str | None = None
    kept: list[OCRBlock] = []
    for b in in_range:
        text = b.text  # already normalized (and optionally split) above

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
        # running header), "C H A P T E R  O N E" (letter-spaced chapter
        # title), and a pattern written with U+3000 (第７章　分院帽) whose
        # normalized text carries an ASCII space still match.
        matched_pat = next(
            (
                pat
                for pat in header_patterns
                if text == pat or _collapse_ws(text) == _collapse_ws(pat)
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
    """Drop PaddleOCR blocks whose x0 is an extreme outlier within their page.

    PaddleOCR occasionally emits stray single-character detections at the
    page edge or in unexpected positions (e.g. misreading a margin mark as
    "X" or "食"). These have bbox x0 values 3–20× larger than typical body
    text. Drop any PaddleOCR block whose x0 exceeds 3× the page's median
    PaddleOCR x0.

    Text-layer (pymupdf) blocks are passed through untouched: legitimate
    born-digital layout — centered letters/address blocks, chapter titles,
    dedication lines — can sit far right of the body margin, and a
    born-digital text layer has no detection noise to filter.
    """
    by_page: dict[int, list[float]] = defaultdict(list)
    for b in blocks:
        if (
            b.engine == "paddleocr"
            and len(b.bbox) >= 4
            and b.bbox[2] > b.bbox[0] > 0
        ):
            by_page[b.page].append(float(b.bbox[0]))
    medians: dict[int, float] = {}
    for page, xs in by_page.items():
        if xs:
            medians[page] = sorted(xs)[len(xs) // 2]
    return [
        b
        for b in blocks
        if b.engine != "paddleocr"
        or b.page not in medians
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
    """True iff this block starts a new paragraph by first-line indent.

    A leading ideographic space (U+3000) is checked first — born-digital
    Chinese text layers encode the first-line indent typographically, often
    with a bbox x0 shift too small for the adaptive gap analysis to detect.
    Falls back to the bbox threshold, then to leading-whitespace detection
    when bbox is unavailable or unusable (e.g. blocks with [0,0,0,0]).
    """
    if block.text.startswith("　"):
        return True
    if threshold is not None and len(block.bbox) >= 4 and block.bbox[2] > block.bbox[0] > 0:
        return float(block.bbox[0]) > threshold
    # Whitespace fallback — works for English text layer where blocks carry
    # their leading spaces.
    return block.text.startswith("　　") or block.text.startswith("    ")


# A digit-only span below the configured size is a footnote marker: either an
# in-text reference (superscript digit inside a body line) or the leading
# number of a note entry (digit followed by small-print note text).
_MARKER_DIGIT_RE = re.compile(r"\d{1,2}")


def _separate_span_footnotes(
    blocks: list[OCRBlock], cfg: dict[str, Any]
) -> tuple[list[OCRBlock], list[OCRBlock]]:
    """Separate footnote markers and note bodies using span font-size metadata.

    For born-digital text layers whose translator notes are typeset in small
    print (e.g. calibre-converted ebooks): digit-only spans below
    ``marker_max_size`` are footnote markers. A line that contains a marker
    and whose remaining spans are all below ``note_max_size`` is a note
    entry — its text (minus the marker digits) goes to the notes output. On
    body lines the marker digits are simply removed. Small-print lines
    *without* a marker stay in the body: embedded documents (letters, lists,
    songs) share the small font, and deleting real text costs more than a
    missed note (fail-open by design).

    Returns ``(body_blocks, note_blocks)``. Blocks without span metadata
    pass through untouched.
    """
    marker_max = float(cfg.get("marker_max_size", 0.0))
    note_max = float(cfg.get("note_max_size", 0.0))
    out_blocks: list[OCRBlock] = []
    notes: list[OCRBlock] = []
    note_serial = 0
    for b in blocks:
        if not b.lines:
            out_blocks.append(b)
            continue
        body_parts: list[str] = []
        for line in b.lines:
            spans = line.spans
            marker_idx = {
                i
                for i, s in enumerate(spans)
                if s.size < marker_max and _MARKER_DIGIT_RE.fullmatch(s.text.strip())
            }
            if not marker_idx:
                body_parts.append("".join(s.text for s in spans))
                continue
            rest = [s for i, s in enumerate(spans) if i not in marker_idx]
            if rest and all(s.size < note_max for s in rest):
                note_text = _strip_block_text("".join(s.text for s in rest)).strip("　 ").strip()
                if note_text:
                    notes.append(
                        b.model_copy(
                            update={"block_idx": 9000 + note_serial, "text": note_text}
                        )
                    )
                    note_serial += 1
            else:
                body_parts.append("".join(s.text for s in rest))
        new_text = "\n".join(body_parts)
        if new_text.strip():
            out_blocks.append(b.model_copy(update={"text": new_text}))
    return out_blocks, notes


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
