"""Focused tests for ``scripts/run_alignments_v2.py`` chapter and pair selection.

The alignment run itself needs the embedding model and real segmented
inputs; here we only cover the pure selection surface: chapter
normalisation, range validation, the full-novel default, the production /
diagnostics pair split, and fail-fast on bad input (before any output
directory is touched).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_alignments_v2.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_alignments_v2", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_is_full_novel():
    mod = _load_script()
    assert mod.DEFAULT_CHAPTERS == tuple(range(1, 18))
    chapters, err = mod.validate_chapters(list(mod.DEFAULT_CHAPTERS))
    assert err is None
    assert chapters == tuple(range(1, 18))


def test_production_pairs_exclude_en_zh():
    """Downstream reads DE-EN and DE-ZH only; EN-ZH must not run (or gate
    anything) unless --diagnostics is passed."""
    mod = _load_script()
    assert mod.effective_pairs(False) == (("de", "en"), ("de", "zh"))
    assert mod.effective_pairs(True) == (("de", "en"), ("de", "zh"), ("en", "zh"))


def test_validate_chapters_sorts_and_dedupes():
    mod = _load_script()
    chapters, err = mod.validate_chapters([17, 4, 4, 5, 1])
    assert err is None
    assert chapters == (1, 4, 5, 17)


def test_validate_chapters_accepts_full_novel_range():
    mod = _load_script()
    chapters, err = mod.validate_chapters(list(range(1, 18)))
    assert err is None
    assert chapters == tuple(range(1, 18))


def test_validate_chapters_rejects_out_of_range():
    mod = _load_script()
    for bad in ([0], [18], [3, 99]):
        chapters, err = mod.validate_chapters(bad)
        assert chapters == ()
        assert err is not None and "1..17" in err


def test_validate_chapters_rejects_empty():
    mod = _load_script()
    chapters, err = mod.validate_chapters([])
    assert chapters == ()
    assert err is not None


def test_main_fails_fast_on_bad_chapters(capsys):
    mod = _load_script()
    rc = mod.main(["--chapters", "18"])
    assert rc == 2
    err_out = capsys.readouterr().err
    assert "1..17" in err_out
