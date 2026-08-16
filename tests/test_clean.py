"""Tests for the cleaning pipeline using synthetic OCR blocks."""

from __future__ import annotations

from hp_corpus.clean import clean_blocks
from hp_corpus.schema import OCRBlock


def _blk(page: int, idx: int, text: str, conf: float = 1.0) -> OCRBlock:
    return OCRBlock(
        page=page,
        block_idx=idx,
        text=text,
        bbox=[0.0, 0.0, 100.0, 20.0],
        confidence=conf,
        engine="pymupdf",
    )


def test_strips_recurring_header_but_keeps_first(en_config) -> None:
    cfg = {**en_config, "chapter": {"number": 1, "start_page": 1, "end_page": 3}}
    blocks = [
        _blk(1, 0, "Harry Potter"),  # header — first occurrence → chapter title
        _blk(1, 1, "THE BOY WHO LIVED"),  # also a header pattern — first occurrence
        _blk(1, 2, "The first paragraph ends here."),
        _blk(2, 0, "Harry Potter"),  # recurring — dropped
        _blk(2, 1, "The second paragraph follows."),
        _blk(3, 0, "Harry Potter"),  # recurring — dropped
        _blk(3, 1, "The third paragraph closes."),
    ]
    result = clean_blocks(blocks, cfg)
    texts = [s.text for s in result.sentences]
    assert "Harry Potter" not in texts
    assert texts[0] == "The first paragraph ends here."
    assert texts[-1] == "The third paragraph closes."
    assert result.chapter_title == "Harry Potter"


def test_drops_page_numbers(en_config) -> None:
    cfg = en_config
    blocks = [
        _blk(1, 0, "13"),
        _blk(1, 1, "The body text."),
        _blk(2, 0, "14"),
        _blk(2, 1, "Continued body."),
    ]
    result = clean_blocks(blocks, cfg)
    texts = " ".join(s.text for s in result.sentences)
    assert "13" not in texts.split()
    assert "14" not in texts.split()


def test_drops_decorative_glyphs(en_config) -> None:
    cfg = en_config
    blocks = [
        _blk(1, 0, "*"),
        _blk(1, 1, "* *"),
        _blk(1, 2, "Real paragraph."),
    ]
    result = clean_blocks(blocks, cfg)
    assert len(result.sentences) == 1
    assert result.sentences[0].text == "Real paragraph."


def test_separates_zh_footnotes(zh_config) -> None:
    cfg = zh_config
    blocks = [
        _blk(1, 0, "正文段落。"),
        _blk(1, 1, "①这是脚注一。"),
        _blk(1, 2, "②这是脚注二。"),
        _blk(1, 3, "正文继续。"),
    ]
    result = clean_blocks(blocks, cfg)
    assert len(result.footnotes) == 2
    assert all("脚注" in n.text for n in result.footnotes)
    body = " ".join(s.text for s in result.sentences)
    assert "脚注" not in body


def test_merges_line_wraps_within_paragraph(en_config) -> None:
    cfg = en_config
    blocks = [
        _blk(1, 0, "This sentence is split across two"),
        _blk(1, 1, "lines but should join."),
    ]
    result = clean_blocks(blocks, cfg)
    # No terminator at end of block 0 → merge
    assert len(result.sentences) == 1
    assert result.sentences[0].text == "This sentence is split across two lines but should join."


def test_repairs_hyphenation(en_config) -> None:
    cfg = en_config
    blocks = [
        _blk(1, 0, "The boy won-"),
        _blk(1, 1, "dered what was happening."),
    ]
    result = clean_blocks(blocks, cfg)
    assert result.sentences[0].text == "The boy wondered what was happening."


def test_preserves_lexical_hyphen(en_config) -> None:
    """Lexical hyphen ('well-known') should not be removed by line-wrap repair."""
    cfg = en_config
    blocks = [
        _blk(1, 0, "He was well-known in town."),
    ]
    result = clean_blocks(blocks, cfg)
    assert "well-known" in result.sentences[0].text


def test_cross_page_paragraph_rejoin(en_config) -> None:
    cfg = en_config
    blocks = [
        _blk(1, 0, "The paragraph continues onto"),
        _blk(2, 0, "the next page without ending."),
    ]
    result = clean_blocks(blocks, cfg)
    assert len(result.sentences) == 1
    assert result.sentences[0].source_pages == [1, 2]


def test_paragraph_boundary_on_terminator(en_config) -> None:
    cfg = en_config
    blocks = [
        _blk(1, 0, "First paragraph."),
        _blk(1, 1, "Second paragraph."),
    ]
    result = clean_blocks(blocks, cfg)
    assert len(result.sentences) == 2


def test_filters_pages_outside_range(en_config) -> None:
    cfg = {**en_config, "chapter": {"number": 1, "start_page": 5, "end_page": 6}}
    blocks = [
        _blk(1, 0, "Page one body."),
        _blk(5, 0, "In-range body."),
        _blk(10, 0, "Out-of-range body."),
    ]
    result = clean_blocks(blocks, cfg)
    texts = [s.text for s in result.sentences]
    assert "In-range body." in texts
    assert "Page one body." not in texts
    assert "Out-of-range body." not in texts


def test_zh_paragraph_rejoin_across_pages(zh_config) -> None:
    cfg = {**zh_config, "chapter": {"number": 1, "start_page": 1, "end_page": 3}}
    blocks = [
        _blk(1, 0, "正文段落在一页开始"),
        _blk(2, 0, "并在下一页继续"),
    ]
    result = clean_blocks(blocks, cfg)
    assert len(result.sentences) == 1
    assert result.sentences[0].source_pages == [1, 2]


def test_strips_header_with_internal_double_spaces(en_config) -> None:
    """Running headers in the Scholastic edition use double-space typography
    ('THE  BOY  WHO  LIVED'). Whitespace-normalized match should catch them."""
    cfg = {**en_config, "chapter": {"number": 1, "start_page": 1, "end_page": 2}}
    blocks = [
        _blk(1, 0, "THE BOY WHO LIVED"),  # establishes chapter_title
        _blk(1, 1, "First body sentence."),
        _blk(2, 0, "THE  BOY  WHO  LIVED"),  # double-spaced variant
        _blk(2, 1, "Second body sentence."),
    ]
    result = clean_blocks(blocks, cfg)
    texts = [s.text for s in result.sentences]
    assert texts == ["First body sentence.", "Second body sentence."]


def test_strips_letter_spaced_chapter_header(en_config) -> None:
    """Chapter title rendered as 'C H A P T E R  O N E' (letter-spaced
    typography) should match the 'CHAPTER ONE' pattern after whitespace
    normalization."""
    cfg = {**en_config, "chapter": {"number": 1, "start_page": 1, "end_page": 1}}
    cfg["clean"] = {**cfg["clean"], "header_patterns": ["CHAPTER ONE"]}
    blocks = [
        _blk(1, 0, "C H A P T E R  O N E"),
        _blk(1, 1, "CHAPTER  ONE"),  # different typography, same header
        _blk(1, 2, "Body paragraph."),
    ]
    result = clean_blocks(blocks, cfg)
    texts = [s.text for s in result.sentences]
    assert texts == ["Body paragraph."]


def test_strips_control_chars_around_page_number(en_config) -> None:
    """PyMuPDF sometimes wraps page numbers in private-use markers
    ('\\x91 17 \\x91'). Control-char stripping lets the page-number regex
    still match."""
    cfg = en_config
    blocks = [
        _blk(1, 0, "\x91 13 \x91"),
        _blk(1, 1, "Body text follows."),
        _blk(2, 0, "\x91 14 \x91"),
        _blk(2, 1, "More body."),
    ]
    result = clean_blocks(blocks, cfg)
    texts = [s.text for s in result.sentences]
    assert all("13" not in s.split() and "14" not in s.split() for s in texts)
    assert all("\x91" not in s for s in texts)


def test_drops_single_letter_block(en_config) -> None:
    """A standalone single-letter block (drop cap extracted as its own block)
    should be dropped — otherwise it would prepend to the wrong paragraph."""
    cfg = en_config
    blocks = [
        _blk(1, 0, "Previous paragraph ends."),
        _blk(1, 1, "M"),  # drop-cap fragment
        _blk(1, 2, "Next paragraph starts."),
    ]
    result = clean_blocks(blocks, cfg)
    texts = [s.text for s in result.sentences]
    # "M" should not appear as a standalone paragraph or be prepended
    assert all(not s.startswith("M ") for s in texts)
    assert all(s != "M" for s in texts)


def test_zh_min_confidence_filter(zh_config) -> None:
    """Low-confidence PaddleOCR blocks (garbage characters) should be dropped
    when min_confidence is set."""
    cfg = {**zh_config, "clean": {**zh_config.get("clean", {}), "min_confidence": 0.5}}
    blocks = [
        _blk(1, 0, "正常中文句子。", conf=0.95),
        _blk(1, 1, "里房机局品济和", conf=0.42),  # garbage below threshold
        _blk(1, 2, "另一段正文。", conf=0.91),
    ]
    result = clean_blocks(blocks, cfg)
    texts = [s.text for s in result.sentences]
    assert all("里房机局" not in s for s in texts)
    assert any("正常中文" in s for s in texts)


def test_de_line_join_adds_space(de_config) -> None:
    """German PaddleOCR is line-by-line; joining two lines without a space
    produces 'warenstolz'. Lines must be joined with a space."""
    cfg = de_config
    blocks = [
        _blk(1, 0, "Die Dursleys waren"),
        _blk(1, 1, "stolz darauf,"),  # next line, no terminator on prev
    ]
    result = clean_blocks(blocks, cfg)
    assert len(result.sentences) == 1
    assert "waren stolz" in result.sentences[0].text
    assert "warenstolz" not in result.sentences[0].text


def test_de_line_join_repairs_hyphenation(de_config) -> None:
    """German end-of-line hyphenation ('Grun-' + 'nings') should be repaired
    to 'Grunnings' (de-hyphenation when next char is lowercase)."""
    cfg = de_config
    blocks = [
        _blk(1, 0, "Eine Firma namens Grun-"),
        _blk(1, 1, "nings"),  # line-wrap hyphenation
    ]
    result = clean_blocks(blocks, cfg)
    assert "Grunnings" in result.sentences[0].text
    assert "Grun-nings" not in result.sentences[0].text


def test_de_preserves_lexical_hyphen(de_config) -> None:
    """Lexical hyphen in middle of a line ('Wohlfühl-Essen') should not be
    removed — only end-of-line hyphen + lowercase next line is repaired."""
    cfg = de_config
    blocks = [
        _blk(1, 0, "Er aß das Wohlfühl-Essen."),
    ]
    result = clean_blocks(blocks, cfg)
    assert "Wohlfühl-Essen" in result.sentences[0].text


def test_zh_unaffected_by_de_line_join(zh_config) -> None:
    """Chinese line joining must remain direct concatenation (no space added)
    after the German fix introduces the lang-in-(en,de) branch."""
    cfg = zh_config
    blocks = [
        _blk(1, 0, "达斯利先生"),
        _blk(1, 1, "非常骄傲"),  # next line — no terminator
    ]
    result = clean_blocks(blocks, cfg)
    assert len(result.sentences) == 1
    # No space inserted between Chinese lines
    assert "达斯利先生非常骄傲" in result.sentences[0].text
    assert " " not in result.sentences[0].text


def test_whitespace_normalization_replaces_control_char_strip(en_config) -> None:
    """A newline inside a block must become a single space, not be dropped.

    The old behavior stripped \\n along with other C0 control chars, which
    concatenated the two words on either side of the newline into one
    token. The new behavior normalizes any whitespace run (incl. \\n,
    \\t, \\xa0) to a single space.
    """
    cfg = en_config
    blocks = [
        _blk(1, 0, "first\nsecond third"),  # \\n between "first" and "second"
        _blk(1, 1, "End of paragraph."),
    ]
    result = clean_blocks(blocks, cfg)
    text = " ".join(s.text for s in result.sentences)
    assert "first second" in text
    assert "firstsecond" not in text


def test_concat_split_disabled_by_default(de_config) -> None:
    """Without ``clean.concat_split``, the splitter must not fire — even on
    an obvious concat artifact — so existing behavior (e.g. PaddleOCR'd
    2013 source) is unchanged."""
    cfg = de_config
    blocks = [
        # "derHaus" is a textbook concat that the splitter would split when
        # enabled. With the flag off, it must survive untouched.
        _blk(1, 0, "derHaus steht"),
        _blk(1, 1, "derHaus steht"),  # repeated so it'd pass freq filter
        _blk(1, 2, "Ende."),
    ]
    result = clean_blocks(blocks, cfg)
    text = " ".join(s.text for s in result.sentences)
    assert "derHaus" in text


def test_concat_split_enabled_splits_artifacts(de_config) -> None:
    """With ``clean.concat_split: true``, the inline wordlist is built from
    this chapter's blocks and obvious concat artifacts are split.

    The frequency filter (>=2) keeps the artifact out of the wordlist
    (it appears once), so the splitter tries to split it.
    """
    cfg = {**de_config, "clean": {**de_config["clean"], "concat_split": True}}
    blocks = [
        # "der" and "Haus" appear often enough to enter the wordlist;
        # "derHaus" appears once as a concat artifact.
        _blk(1, 0, "der Haus steht"),
        _blk(1, 1, "das Haus steht"),
        _blk(1, 2, "derHaus steht"),  # artifact
        _blk(1, 3, "Ende."),
    ]
    result = clean_blocks(blocks, cfg)
    text = " ".join(s.text for s in result.sentences)
    assert "der Haus steht" in text
    assert "derHaus" not in text


def test_concat_split_preserves_real_compounds(de_config) -> None:
    """A real German compound (Apfelbaum) that appears multiple times must
    NOT be split even when concat_split is on — the early `_is_known`
    check in split_concat returns the token unchanged."""
    cfg = {**de_config, "clean": {**de_config["clean"], "concat_split": True}}
    blocks = [
        _blk(1, 0, "Apfelbaum blüht"),
        _blk(1, 1, "Apfelbaum wächst"),
        _blk(1, 2, "Ende."),
    ]
    result = clean_blocks(blocks, cfg)
    text = " ".join(s.text for s in result.sentences)
    assert "Apfelbaum" in text  # not split into "Apfel baum"


# --- Span-based footnote separation (born-digital text layers) -------------


def _tl_blk(page: int, idx: int, lines_spec: list[list[tuple[str, float]]]) -> OCRBlock:
    """Build a text-layer block from (text, size) span specs, mirroring the
    pymupdf engine's output (text == '\\n'.join of line span texts)."""
    from hp_corpus.schema import OCRLine, OCRSpan

    lines = [OCRLine(spans=[OCRSpan(text=t, size=sz) for t, sz in ln]) for ln in lines_spec]
    text = "\n".join("".join(t for t, _ in ln) for ln in lines_spec)
    return OCRBlock(
        page=page,
        block_idx=idx,
        text=text,
        bbox=[0.0, 0.0, 100.0, 20.0],
        confidence=1.0,
        engine="pymupdf",
        lines=lines,
    )


def test_span_footnotes_separate_note_entries(zh_textlayer_config) -> None:
    """A small-print line starting with a marker digit is a note entry: its
    text goes to _notes output and disappears from the body."""
    cfg = zh_textlayer_config
    blocks = [
        _tl_blk(1, 0, [[("　第一段落正文内容甲乙。", 10.8), ("继续第二句。", 10.8)]]),
        _tl_blk(1, 1, [[("1", 6.6), ("译注说明文字丙丁戊。", 7.8)]]),
        _tl_blk(1, 2, [[("2", 6.6), ("第二条译注文字己庚。", 7.8)]]),
    ]
    result = clean_blocks(blocks, cfg)
    assert len(result.footnotes) == 2
    assert all("译注" in n.text for n in result.footnotes)
    assert all(not n.text[0].isdigit() for n in result.footnotes)
    body = " ".join(s.text for s in result.sentences)
    assert "译注" not in body
    assert "第一段落" in body


def test_span_footnotes_strip_inline_reference_markers(zh_textlayer_config) -> None:
    """A superscript digit inside a body-size line is an in-text reference:
    drop the digit, keep everything else, and emit no note."""
    cfg = zh_textlayer_config
    blocks = [
        _tl_blk(1, 0, [[("句子前半内容", 10.8), ("1", 6.6), ("，句子后半内容。", 10.8)]]),
    ]
    result = clean_blocks(blocks, cfg)
    assert len(result.footnotes) == 0
    assert len(result.sentences) == 1
    assert result.sentences[0].text == "句子前半内容，句子后半内容。"


def test_span_small_print_without_marker_stays_in_body(zh_textlayer_config) -> None:
    """Small-print lines with no marker digit are embedded documents
    (letters/lists), not notes — they must stay in the body."""
    cfg = zh_textlayer_config
    blocks = [
        _tl_blk(1, 0, [[("　正文段落内容一。", 10.8)]]),
        _tl_blk(1, 1, [[("嵌入文件的小字内容。", 7.8)]]),
    ]
    result = clean_blocks(blocks, cfg)
    assert len(result.footnotes) == 0
    body = " ".join(s.text for s in result.sentences)
    assert "嵌入文件" in body


def test_span_footnotes_inert_without_config(zh_config) -> None:
    """Without clean.footnote_spans, marker digits pass through untouched
    (backward compatibility for scanned sources)."""
    blocks = [
        _tl_blk(1, 0, [[("句子前半", 10.8), ("1", 6.6), ("，句子后半。", 10.8)]]),
    ]
    result = clean_blocks(blocks, zh_config)
    assert len(result.footnotes) == 0
    assert "1" in result.sentences[0].text


def test_span_footnotes_lone_marker_line_dropped(zh_textlayer_config) -> None:
    """A line that is only a marker digit contributes nothing."""
    cfg = zh_textlayer_config
    blocks = [
        _tl_blk(1, 0, [[("　正文段落内容。", 10.8)]]),
        _tl_blk(1, 1, [[("3", 6.6)]]),
    ]
    result = clean_blocks(blocks, cfg)
    assert len(result.footnotes) == 0
    assert len(result.sentences) == 1
    assert result.sentences[0].text == "正文段落内容。"


def test_span_footnotes_blocks_without_metadata_pass_through(zh_textlayer_config) -> None:
    """PaddleOCR-style blocks (no span metadata) are untouched even when
    footnote_spans is configured — the span rules simply don't apply."""
    cfg = zh_textlayer_config
    blocks = [_blk(1, 0, "正文段落一号内容。"), _blk(1, 1, "正文段落二号内容。")]
    result = clean_blocks(blocks, cfg)
    assert len(result.footnotes) == 0
    # No indent signal → ZH blocks merge into one running paragraph.
    assert [s.text for s in result.sentences] == ["正文段落一号内容。正文段落二号内容。"]


def test_leading_ideographic_space_splits_paragraphs(zh_textlayer_config) -> None:
    """A leading U+3000 (CJK first-line indent) starts a new paragraph even
    when the bbox x0 gives no usable indent signal; the prefix itself must
    not leak into the cleaned text."""
    cfg = zh_textlayer_config
    blocks = [
        _tl_blk(1, 0, [[("　缩进的第一段落。", 10.8)]]),
        _tl_blk(1, 1, [[("同栏续行内容。", 10.8)]]),
        _tl_blk(1, 2, [[("　缩进的第二段落。", 10.8)]]),
    ]
    result = clean_blocks(blocks, cfg)
    texts = [s.text for s in result.sentences]
    assert texts == ["缩进的第一段落。同栏续行内容。", "缩进的第二段落。"]


def test_header_pattern_u3000_matches_normalized_text(zh_textlayer_config) -> None:
    """A header pattern written with an internal U+3000 (第７章　分院帽) must
    match the chapter heading after whitespace normalization has turned the
    U+3000 into an ASCII space — the heading is recorded as chapter title and
    kept out of the body."""
    cfg = zh_textlayer_config
    blocks = [
        _tl_blk(1, 0, [[("第７章", 13.8), ("　", 13.8), ("分院帽", 13.8)]]),
        _tl_blk(1, 1, [[("　正文第一段落内容。", 10.8)]]),
    ]
    result = clean_blocks(blocks, cfg)
    assert result.chapter_title == "第７章 分院帽"
    body = " ".join(s.text for s in result.sentences)
    assert "分院帽" not in body
    assert "正文第一段落" in body
