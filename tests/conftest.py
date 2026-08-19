"""Shared test fixtures. All fixtures use synthetic, non-novel text."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Provide a clean working directory under tmp_path."""
    return tmp_path


@pytest.fixture
def fake_encoder(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``sentence_transformers.SentenceTransformer`` so tests
    exercise the cache logic without loading a real embedding model.

    The fake produces a deterministic 4-dim vector per input string
    (hash-based), so identical inputs yield identical vectors across calls."""

    class FakeModel:
        def __init__(self, name: str) -> None:
            self.name = name

        def encode(self, inputs, **kwargs):  # type: ignore[no-untyped-def]
            return np.array(
                [[float((hash(s + f"|{i}") % 1000) / 1000.0) for i in range(4)] for s in inputs],
                dtype=np.float32,
            )

    fake_module = type(sys)("sentence_transformers")
    fake_module.SentenceTransformer = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return FakeModel


@pytest.fixture
def en_config() -> dict:
    return {
        "book": "hp1",
        "lang": "en",
        "chapter": {"number": 1, "start_page": 1, "end_page": 2},
        "ocr": {"engine": "pymupdf", "lang": "en"},
        "clean": {
            "header_patterns": ["Harry Potter", "THE BOY WHO LIVED"],
            "remove_page_numbers": True,
            "merge_line_breaks": True,
            "paragraph_detection": {"indent_threshold": 2, "dialogue_markers": ['"', '"']},
        },
        "segment": {
            "lang": "en",
            "split_on": [".", "!", "?"],
            "abbreviations": ["Mr.", "Mrs.", "Dr.", "St.", "etc.", "Prof."],
        },
    }


@pytest.fixture
def zh_config() -> dict:
    return {
        "book": "hp1",
        "lang": "zh",
        "chapter": {"number": 1, "start_page": 1, "end_page": 2},
        "ocr": {"engine": "paddleocr", "lang": "ch"},
        "clean": {
            "header_patterns": ["哈利·波特与魔法石", "第一章 大难不死的男孩"],
            "remove_page_numbers": True,
            "footnote_markers": ["①", "②", "③", "④", "⑤"],
            "merge_line_breaks": True,
            "paragraph_detection": {"indent_threshold": 2, "dialogue_markers": ['"', '"']},
        },
        "segment": {
            "lang": "zh",
            "split_on": ["。", "！", "？", "；"],
            "preserve_ellipsis": True,
        },
    }


@pytest.fixture
def zh_textlayer_config() -> dict:
    """Config mirroring the born-digital ZH ebook source (pymupdf engine,
    span-based footnote separation, U+3000 paragraph indents)."""
    return {
        "book": "hp1",
        "lang": "zh",
        "chapter": {"number": 7, "start_page": 1, "end_page": 2},
        "ocr": {"engine": "pymupdf", "lang": "zh"},
        "clean": {
            "header_patterns": ["第７章　分院帽"],
            "remove_page_numbers": True,
            "footnote_spans": {"marker_max_size": 7.0, "note_max_size": 9.0},
            "merge_line_breaks": True,
            "paragraph_detection": {"indent_threshold": 2, "dialogue_markers": ["“", "”"]},
        },
        "segment": {
            "lang": "zh",
            "split_on": ["。", "！", "？", "；"],
            "preserve_ellipsis": True,
        },
    }


@pytest.fixture
def de_config() -> dict:
    return {
        "book": "hp1",
        "lang": "de",
        "chapter": {"number": 1, "start_page": 1, "end_page": 2},
        "ocr": {"engine": "paddleocr", "lang": "german"},
        "clean": {
            "header_patterns": ["Harry Potter", "Ein Junge überlebt"],
            "remove_page_numbers": True,
            "merge_line_breaks": True,
            "paragraph_detection": {"indent_threshold": 2, "dialogue_markers": ["»", "«"]},
        },
        "segment": {
            "lang": "de",
            "split_on": [".", "!", "?"],
            "abbreviations": ["Mr.", "Mrs.", "Dr.", "St.", "etc.", "Prof.", "Nr.", "bzw."],
        },
    }


@pytest.fixture
def write_yaml(tmp_path: Path):
    """Helper: write a config dict to a YAML file and return its path."""

    def _write(name: str, data: dict) -> Path:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        return p

    return _write
