"""DEPRECATED thin wrapper — use ``run_full_novel_german_extraction.py``.

This script was the original Ch.1–3 paper-extractor CLI. Its standalone
implementation is gone: it emitted only ``sentence_id`` (ambiguous once
one segment is parsed into several blocks, incompatible with the
current extraction schema), mis-padded chapter paths for Ch.10–17
(``ch010``), and silently skipped missing parsed inputs.

The paper-faithful algorithm lives on, unchanged, in
``hp_corpus.german_extraction.extract_chapter`` behind
``scripts/run_full_novel_german_extraction.py``, which emits
``parse_block_id`` + ``source_segment_id`` per row, zero-pads chapter
paths, and fails closed on missing/empty/corrupt input. There is exactly
one extraction implementation.

This wrapper keeps the old invocation working: it forwards ``--chapters``
(default ``1 2 3``), ``--parsed-dir`` and ``--output-dir`` to the new
CLI and prints a deprecation notice to stderr. Output TSVs use the new
schema and filenames (``hp1_de_ch{NN}_{form}.tsv`` plus a
``manifest.json`` in the output directory).

Usage:
    uv run python scripts/run_paper_extractor.py [--chapters 1 2 3] \
        [--parsed-dir data/parsed] [--output-dir data/extracted]
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "run_full_novel_german_extraction",
    _SCRIPTS_DIR / "run_full_novel_german_extraction.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_full_novel = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_full_novel)

DEPRECATION_NOTICE = (
    "DEPRECATED: scripts/run_paper_extractor.py is a thin wrapper. "
    "Use scripts/run_full_novel_german_extraction.py instead "
    "(same algorithm, block-level provenance columns, fixed chapter paths)."
)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    # The legacy CLI defaulted to chapters 1-3; the new one requires an
    # explicit list. Handle both "--chapters 1 2 3" and "--chapters=1".
    has_chapters = any(
        a == "--chapters" or a.startswith("--chapters=") for a in args
    )
    if not has_chapters:
        args = ["--chapters", "1", "2", "3"] + args
    print(DEPRECATION_NOTICE, file=sys.stderr)
    return _full_novel.main(args)


if __name__ == "__main__":
    sys.exit(main())
