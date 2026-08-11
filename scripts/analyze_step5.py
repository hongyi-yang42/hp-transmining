"""Aggregate Step 5 analysis on a derived-category TSV.

Reads the detailed derived-category TSV produced by
``derive_paper_categories.py`` and writes:

  * ``<out-dir>/summary.json`` — top-level aggregate counts, distribution
    objects, cross-tab objects with explicit denominators, and the
    ``source_labels_rolled_up`` block enumerating every fine label that
    contributed to each ``other`` bucket.
  * ``<out-dir>/de_distribution.json``
  * ``<out-dir>/en_distribution.json``
  * ``<out-dir>/zh_distribution.json``
  * ``<out-dir>/de_x_zh_table.json``
  * ``<out-dir>/de_x_en_table.json``
  * ``<out-dir>/uncontracted_mandarin_bare_review.tsv`` — one row per
    datapoint in the ``uncontracted + Mandarin bare`` cell, columns
    ``datapoint_id`` and ``chapter`` only (no sentence text).

All aggregate outputs carry counts and row percentages with explicit
denominators. Stdout carries only summary counts and the output paths.

Usage::

    uv run python scripts/analyze_step5.py \\
        --derived-tsv data/derived/step5/derived_categories.tsv \\
        --out-dir data/derived/step5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from hp_corpus.step5 import analyze

EXIT_OK = 0
EXIT_INPUT_ERROR = 2


def _exit(rule: str, message: str) -> int:
    print(f"FAIL: rule={rule} {message}", file=sys.stderr)
    return EXIT_INPUT_ERROR


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def _write_review_tsv(path: Path, ids: list[str], chapters: dict[str, str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["datapoint_id", "chapter"])
        for dp in ids:
            w.writerow([dp, chapters.get(dp, "")])
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--derived-tsv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--force-output",
        action="store_true",
        help="Required if any output file already exists.",
    )
    args = ap.parse_args(argv)

    if not args.derived_tsv.exists():
        return _exit("DERIVED_TSV_NOT_FOUND", str(args.derived_tsv))

    outputs = [
        args.out_dir / "summary.json",
        args.out_dir / "de_distribution.json",
        args.out_dir / "en_distribution.json",
        args.out_dir / "zh_distribution.json",
        args.out_dir / "de_x_zh_table.json",
        args.out_dir / "de_x_en_table.json",
        args.out_dir / "uncontracted_mandarin_bare_review.tsv",
    ]
    if any(p.exists() for p in outputs) and not args.force_output:
        existing = [str(p) for p in outputs if p.exists()]
        return _exit("OUTPUT_EXISTS", f"{', '.join(existing)}; pass --force-output")

    with open(args.derived_tsv, encoding="utf-8") as f:
        derived = list(csv.DictReader(f, delimiter="\t"))

    summary = analyze(derived)

    _write_json(args.out_dir / "summary.json", summary)
    _write_json(args.out_dir / "de_distribution.json", summary["de_distribution"])
    _write_json(args.out_dir / "en_distribution.json", summary["en_distribution"])
    _write_json(args.out_dir / "zh_distribution.json", summary["zh_distribution"])
    _write_json(args.out_dir / "de_x_zh_table.json", summary["de_x_zh"])
    _write_json(args.out_dir / "de_x_en_table.json", summary["de_x_en"])

    # Review list: IDs only, no novel text. The user opens the gold TSV
    # to read each row's full content.
    from hp_corpus.step5 import uncontracted_mandarin_bare_ids

    ids = uncontracted_mandarin_bare_ids(derived)
    chapters = {r["datapoint_id"]: r.get("chapter", "") for r in derived}
    _write_review_tsv(args.out_dir / "uncontracted_mandarin_bare_review.tsv", ids, chapters)

    n = summary["analysis_ready_total"]
    print(f"analysis_ready: {n}")
    print(f"uncontracted_mandarin_bare: {summary['uncontracted_mandarin_bare_count']}")
    print(f"out_dir: {args.out_dir}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
