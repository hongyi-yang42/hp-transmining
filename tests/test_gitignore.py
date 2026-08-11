"""Tests that ensure no copyrighted corpus files end up tracked in git.

These are a guardrail against accidentally committing PDFs, page images,
OCR output, cleaned text, embeddings, or model weights.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _git_check_ignore(path: str) -> bool:
    """Return True if `path` is gitignored at the repo root."""
    result = subprocess.run(
        ["git", "check-ignore", path],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_data_raw_is_ignored(repo_root: Path) -> None:
    assert _git_check_ignore("data/raw/hp1_en.pdf")


def test_data_pages_is_ignored(repo_root: Path) -> None:
    assert _git_check_ignore("data/pages/hp1_en/hp1_en_p0001.png")


def test_data_ocr_raw_is_ignored(repo_root: Path) -> None:
    assert _git_check_ignore("data/ocr_raw/hp1_en_ch01.jsonl")


def test_data_text_clean_is_ignored(repo_root: Path) -> None:
    assert _git_check_ignore("data/text_clean/hp1_en_ch01.txt")


def test_data_segmented_is_ignored(repo_root: Path) -> None:
    assert _git_check_ignore("data/segmented/hp1_en_ch01.jsonl")


def test_data_aligned_is_ignored(repo_root: Path) -> None:
    assert _git_check_ignore("data/aligned/hp1_en_zh_ch01.jsonl")


def test_data_extracted_is_ignored(repo_root: Path) -> None:
    """Step 3 output: per-occurrence TSVs with German text + lemma."""
    assert _git_check_ignore("data/extracted/hp1_de_ch01_contracted.tsv")


def test_data_derived_step4_is_ignored(repo_root: Path) -> None:
    """Step 4 output: candidate JSONL and pilot TSV with DE/EN/ZH novel text."""
    assert _git_check_ignore("data/derived/step4/ch1_3_pilot_20.tsv")
    assert _git_check_ignore("data/derived/step4/ch1_3_all_candidates.jsonl")


def test_data_embeddings_is_ignored(repo_root: Path) -> None:
    assert _git_check_ignore("data/embeddings/en.npy")


def test_models_is_ignored(repo_root: Path) -> None:
    assert _git_check_ignore("models/paddleocr/some.bin")


def test_vendor_vecalign_is_ignored(repo_root: Path) -> None:
    assert _git_check_ignore("vendor/vecalign/vecalign.py")


def test_no_tracked_pdf_or_png_or_jsonl(repo_root: Path) -> None:
    """`git ls-files` should return nothing matching corpus file extensions.
    Test fixtures are synthetic, but we still don't want any binary or text
    data files in git."""
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr
    files = result.stdout.splitlines()
    forbidden_extensions = (".pdf", ".png", ".jpg", ".jpeg", ".epub", ".npy", ".pkl")
    bad = [f for f in files if f.endswith(forbidden_extensions)]
    assert not bad, f"Tracked files with forbidden extensions: {bad}"


def test_no_tracked_data_path(repo_root: Path) -> None:
    """Nothing under data/, models/, vendor/, artifacts/, tmp/ should be tracked.

    The one exception is ``vendor/conll-extractor.commit`` — a single-line
    text file pinning the vendored extractor commit. The vendor checkout
    itself stays gitignored.
    """
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    files = result.stdout.splitlines()
    forbidden_prefixes = ("data/", "models/", "vendor/", "artifacts/", "tmp/")
    allowed = {"vendor/conll-extractor.commit"}
    bad = [
        f
        for f in files
        if any(f.startswith(p) for p in forbidden_prefixes) and f not in allowed
    ]
    assert not bad, f"Tracked files in gitignored directories: {bad}"
