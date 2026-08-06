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
