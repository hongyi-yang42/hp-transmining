"""Focused tests for ``scripts/run_alignments_v2.py`` chapter selection.

The alignment run itself needs the e5-base model and real segmented inputs;
here we only cover the new backward-compatible ``--chapters`` surface:
normalisation, range validation, the Ch.1-3 default, and fail-fast on bad
input (before any output directory is touched).
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


def test_default_is_ch1_3():
    mod = _load_script()
    assert mod.DEFAULT_CHAPTERS == (1, 2, 3)
    chapters, err = mod.validate_chapters(list(mod.DEFAULT_CHAPTERS))
    assert err is None
    assert chapters == (1, 2, 3)


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
