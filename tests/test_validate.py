"""Tests for source validation: SHA-256, page count, text-layer detection.

Uses synthetic PDFs generated in tmp_path via PyMuPDF — no source PDFs required.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import fitz
import pytest

from hp_corpus.render import ValidationError, has_text_layer, sha256_file, validate_source


def _make_text_pdf(path: Path, n_pages: int = 5) -> None:
    """Create a synthetic PDF with a usable text layer."""
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page()
        # Insert enough text to exceed the has_text_layer threshold (50 chars median).
        page.insert_text(
            (72, 72),
            f"Page {i + 1}: synthetic content for testing purposes. " * 5,
            fontsize=12,
        )
    doc.save(str(path))
    doc.close()


def _make_image_pdf(path: Path, n_pages: int = 5) -> None:
    """Create a synthetic PDF with NO text layer (image-only)."""
    doc = fitz.open()
    for _ in range(n_pages):
        doc.new_page()
    doc.save(str(path))
    doc.close()


def test_sha256_matches_known_value(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_bytes(b"hello world")
    assert sha256_file(p) == hashlib.sha256(b"hello world").hexdigest()


def test_has_text_layer_true_for_text_pdf(tmp_path: Path) -> None:
    p = tmp_path / "text.pdf"
    _make_text_pdf(p, n_pages=5)
    with fitz.open(p) as doc:
        assert has_text_layer(doc) is True


def test_has_text_layer_false_for_image_pdf(tmp_path: Path) -> None:
    p = tmp_path / "image.pdf"
    _make_image_pdf(p, n_pages=5)
    with fitz.open(p) as doc:
        assert has_text_layer(doc) is False


def test_validate_source_passes_on_matching_text_pdf(tmp_path: Path) -> None:
    p = tmp_path / "text.pdf"
    _make_text_pdf(p, n_pages=5)
    expected_sha = sha256_file(p)
    cfg = {
        "book": "test",
        "lang": "en",
        "pdf_path": str(p),
        "expected_sha256": expected_sha,
        "total_pages": 5,
        "has_text_layer": True,
    }
    report = validate_source(cfg)
    assert report.sha256_ok
    assert report.page_count_ok
    assert report.text_layer_ok


def test_validate_source_detects_sha_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "text.pdf"
    _make_text_pdf(p, n_pages=5)
    cfg = {
        "book": "test",
        "lang": "en",
        "pdf_path": str(p),
        "expected_sha256": "0" * 64,
        "total_pages": 5,
        "has_text_layer": True,
    }
    with pytest.raises(ValidationError, match="sha256"):
        validate_source(cfg)


def test_validate_source_detects_page_count_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "text.pdf"
    _make_text_pdf(p, n_pages=5)
    cfg = {
        "book": "test",
        "lang": "en",
        "pdf_path": str(p),
        "expected_sha256": sha256_file(p),
        "total_pages": 999,
        "has_text_layer": True,
    }
    with pytest.raises(ValidationError, match="page count"):
        validate_source(cfg)


def test_validate_source_detects_text_layer_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "image.pdf"
    _make_image_pdf(p, n_pages=5)
    cfg = {
        "book": "test",
        "lang": "zh",
        "pdf_path": str(p),
        "expected_sha256": sha256_file(p),
        "total_pages": 5,
        "has_text_layer": True,  # claim it has a layer; it doesn't
    }
    with pytest.raises(ValidationError, match="text-layer"):
        validate_source(cfg)


def test_validate_source_raises_when_file_missing(tmp_path: Path) -> None:
    cfg = {
        "book": "test",
        "lang": "en",
        "pdf_path": str(tmp_path / "nonexistent.pdf"),
        "expected_sha256": "0" * 64,
        "total_pages": 5,
        "has_text_layer": True,
    }
    with pytest.raises(ValidationError, match="not found"):
        validate_source(cfg)


def test_validate_source_skips_hash_when_expected_sha_absent(tmp_path: Path) -> None:
    """Public configs ship without expected_sha256; validation must still
    succeed (file-existence, page-count, and text-layer checks run; hash
    check is skipped)."""
    p = tmp_path / "text.pdf"
    _make_text_pdf(p, n_pages=5)
    cfg = {
        "book": "test",
        "lang": "en",
        "pdf_path": str(p),
        # NOTE: no expected_sha256 field — mimics the public configs.
        "total_pages": 5,
        "has_text_layer": True,
    }
    report = validate_source(cfg)
    assert report.expected_sha256 is None
    assert report.actual_sha256_prefix == ""  # not computed when expected is absent
    assert report.sha256_ok is True  # skipped counts as ok
    assert report.page_count_ok
    assert report.text_layer_ok


def test_validate_source_without_sha_still_detects_page_count_mismatch(
    tmp_path: Path,
) -> None:
    """Hash-skip must not also skip page-count / text-layer checks."""
    p = tmp_path / "text.pdf"
    _make_text_pdf(p, n_pages=5)
    cfg = {
        "book": "test",
        "lang": "en",
        "pdf_path": str(p),
        "total_pages": 999,  # mismatch — must still be caught
        "has_text_layer": True,
    }
    with pytest.raises(ValidationError, match="page count"):
        validate_source(cfg)
