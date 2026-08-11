"""Derive coarse paper-facing categories from a gold annotation TSV.

Reads a gold TSV (must have passed ``validate_step4_annotations.py
--require-complete``), filters to analysis-ready rows, derives coarse
DE / EN / ZH categories while preserving the fine annotator labels, and
writes:

  * ``<output>`` — detailed derived-category TSV (one row per
    analysis-ready datapoint).

Stdout carries aggregate counts only — no lemmas, surface forms, or
datapoint IDs.

Usage::

    uv run python scripts/derive_paper_categories.py \\
        --gold-tsv data/derived/step4/gold.tsv \\
        --output data/derived/step5/derived_categories.tsv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from hp_corpus.step5 import DERIVED_COLUMNS, derive_rows

EXIT_OK = 0
EXIT_INPUT_ERROR = 2


def _exit(rule: str, message: str) -> int:
    print(f"FAIL: rule={rule} {message}", file=sys.stderr)
    return EXIT_INPUT_ERROR


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold-tsv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--force-output",
        action="store_true",
        help="Required if --output already exists.",
    )
    args = ap.parse_args(argv)

    if not args.gold_tsv.exists():
        return _exit("GOLD_TSV_NOT_FOUND", str(args.gold_tsv))
    if args.output.exists() and not args.force_output:
        return _exit("OUTPUT_EXISTS", f"{args.output}; pass --force-output to overwrite")

    with open(args.gold_tsv, encoding="utf-8") as f:
        gold_rows = list(csv.DictReader(f, delimiter="\t"))

    derived = derive_rows(gold_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(DERIVED_COLUMNS),
            delimiter="\t",
            lineterminator="\n",
        )
        w.writeheader()
        for r in derived:
            w.writerow(r)

    print(f"gold_rows: {len(gold_rows)}")
    print(f"analysis_ready: {len(derived)}")
    print(f"output: {args.output}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
