"""Tests for text-layer extraction (pymupdf engine) — synthetic PDFs only."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from hp_corpus.clean import clean_blocks
from hp_corpus.render import extract_text_layer_blocks


def _make_pdf(path: Path) -> None:
    """Build a one-page PDF mimicking the born-digital ZH ebook layout:
    body text at 10.8pt (first paragraph indented with U+3000), a superscript
    reference marker at 6.6pt mid-line, and a small-print note entry
    (6.6pt digit + 7.8pt text) at the page end."""
    doc = fitz.open()
    page = doc.new_page(width=500, height=709)
    page.insert_text((72, 100), "　测试段落甲乙丙丁。", fontname="china-s", fontsize=10.8)
    page.insert_text((72, 130), "测试句子戊己庚辛", fontname="china-s", fontsize=10.8)
    w = fitz.get_text_length("测试句子戊己庚辛", fontname="china-s", fontsize=10.8)
    page.insert_text((72 + w, 130), "1", fontname="china-s", fontsize=6.6)
    page.insert_text((72, 160), "，测试句子继续壬癸。", fontname="china-s", fontsize=10.8)
    page.insert_text((72, 500), "1", fontname="china-s", fontsize=6.6)
    w2 = fitz.get_text_length("1", fontname="china-s", fontsize=6.6)
    page.insert_text((72 + w2 + 2, 500), "测试译注文字子丑寅卯。", fontname="china-s", fontsize=7.8)
    doc.save(path)
    doc.close()


@pytest.fixture
def synthetic_pdf(tmp_path: Path) -> Path:
    p = tmp_path / "synthetic_zh.pdf"
    _make_pdf(p)
    return p


def test_extract_blocks_preserves_span_metadata(synthetic_pdf: Path) -> None:
    blocks = extract_text_layer_blocks(synthetic_pdf, 1, 1)
    assert blocks, "expected at least one text block"
    sizes = [round(s.size, 1) for b in blocks for line in b.lines for s in line.spans]
    assert 10.8 in sizes
    assert 6.6 in sizes
    assert 7.8 in sizes
    assert all(b.engine == "pymupdf" for b in blocks)
    assert all(b.page == 1 for b in blocks)
    assert all(b.confidence == 1.0 for b in blocks)


def test_extract_blocks_text_join_invariant(synthetic_pdf: Path) -> None:
    """OCRBlock.text must equal the '\\n'-join of its line span texts, so the
    clean stage can reconstruct exact character offsets from metadata."""
    blocks = extract_text_layer_blocks(synthetic_pdf, 1, 1)
    for b in blocks:
        joined = "\n".join("".join(s.text for s in line.spans) for line in b.lines)
        assert b.text == joined


def test_extract_blocks_page_range_window(synthetic_pdf: Path) -> None:
    """A page range that excludes the only page yields no blocks."""
    assert extract_text_layer_blocks(synthetic_pdf, 2, 3) == []


def test_span_footnote_pipeline_end_to_end(
    synthetic_pdf: Path, zh_textlayer_config: dict
) -> None:
    """Text-layer extraction → cleaning separates the small-print note entry
    into _notes output, strips the inline reference digit from the body, and
    keeps the U+3000-indent paragraph structure."""
    blocks = extract_text_layer_blocks(synthetic_pdf, 1, 1)
    result = clean_blocks(blocks, zh_textlayer_config)
    assert len(result.footnotes) == 1
    assert "测试译注" in result.footnotes[0].text
    assert not result.footnotes[0].text[0].isdigit()
    body = " ".join(s.text for s in result.sentences)
    assert "测试译注" not in body
    assert "测试句子戊己庚辛，测试句子继续壬癸。" in body
    assert "　" not in body
