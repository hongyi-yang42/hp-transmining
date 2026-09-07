"""Deterministic rejoin of the reviewed split pair into one complete CSV.

The annotation deliverable is split by machine reliability
(``scripts/build_annotation_csv.py --min-confidence``): the annotator file
plus a low-confidence companion for deferred rows. Both come back filled,
but ``scripts/build_eligible_pool.py --review-csv`` reads exactly one
complete review covering every master ``datapoint_id`` — this script is the
bridge between the two contracts.

It assembles, it does not judge. Human-entered annotation cells are copied
**verbatim** from whichever returned file owns the row; interpreting them
(vocabularies, coupling, review completeness) stays the job of the ordinary
full run of ``scripts/validate_annotation_csv.py`` on the merged file, which
is the required next step of the runbook — only a validator-clean complete
merged file may be consumed by ``build_eligible_pool.py``.

Fail-closed checks, all before anything is written (exit 2, aggregate
stdout only):

  * master absent, duplicate master ``datapoint_id``, or header missing
    required columns;
  * the split is re-derived from the master via
    ``hp_corpus.annotation_csv.split_low_confidence`` (the same single
    definition the builder and the validator use) — the file pair is never
    trusted to define its own row sets;
  * main header exactly ``CSV_COLUMNS``; companion header exactly
    ``LOW_CONF_COLUMNS``;
  * main id set equals the kept rows exactly; companion id set equals the
    deferred rows exactly — so a row moved between files, an extra id, an
    overlap, or a missing row all fail closed, and the disjoint union is
    exactly the master id set;
  * every returned ``row_hash`` equals the master's
    ``source_row_sha256`` for the same id;
  * every machine column is re-derived from the master row and compared
    cell by cell (after ``.strip()``, so the writer's Excel-safe leading
    space round-trips);
  * the companion's three diagnostic machine columns
    (``machine_*_alignment_confidence``, ``machine_low_conf_sides``) are
    re-derived from the master like any machine cell.

The merged file's machine columns are the canonical master projection
(``excel_safe`` applied as in ``build_annotation_csv.py``), so the output is
a pure function of the master plus the returned human cells — independent
of the row order or Excel quirks of either returned file. Rows are emitted
in master ``datapoint_id`` order (the canonical order the builder used).
The companion's diagnostic columns are dropped: they restate master cells
and exist only so deferred rows can be eyeballed standalone.

Usage::

    uv run python scripts/merge_companion_review.py \\
        --review-csv data/derived/annotation/annotation_pairs.csv \\
        --companion-csv data/derived/annotation/annotation_pairs_low_confidence.csv \\
        --master-tsv data/derived/step4/full_novel_annotation_master.tsv \\
        --min-confidence 0.40 \\
        --output data/derived/annotation/annotation_pairs_merged.csv

Stdout carries aggregate counts only — never corpus text, lemmas, or ids.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from hp_corpus.annotation_csv import (
    _EXCEL_SAFE_COLUMNS,
    ANNOTATOR_COLUMNS,
    CSV_COLUMNS,
    LOW_CONF_COLUMNS,
    LOW_CONF_EXTRA_COLUMNS,
    MASTER_TO_CSV_COLUMNS,
    excel_safe,
    split_low_confidence,
)

EXIT_OK = 0
EXIT_INPUT_ERROR = 2

# Legitimately blank machine columns (anchorless side → empty context).
_BLANKABLE = ("english_context", "chinese_context")


def _exit(rule: str, message: str) -> int:
    print(f"FAIL: rule={rule} {message}", file=sys.stderr)
    return EXIT_INPUT_ERROR


def read_master_rows(path: Path) -> list[dict[str, str]]:
    """Read the machine master and fail closed on structural problems."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        header = list(reader.fieldnames or [])
    required = list(MASTER_TO_CSV_COLUMNS)
    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(f"master header is missing required columns: {missing}")
    seen: set[str] = set()
    for r in rows:
        dp = r.get("datapoint_id", "")
        if not dp:
            raise ValueError("master row with empty datapoint_id")
        if dp in seen:
            raise ValueError(f"duplicate datapoint_id in master: {dp}")
        seen.add(dp)
    return rows


def read_returned(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    # utf-8-sig tolerates files returned with or without the BOM (Excel
    # round-trips sometimes drop it).
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    return rows, header


def _binding_errors(
    label: str,
    rows: list[dict[str, str]],
    header: list[str],
    expected_header: tuple[str, ...],
    master_index: dict[str, dict[str, str]],
    expected_ids: set[str],
    low_diag: dict[str, dict[str, str]] | None,
) -> tuple[list[str], set[str]]:
    """Header, row-set binding, and machine-cell integrity for one returned
    file. ``low_diag`` is non-None for the companion (adds the three
    diagnostic columns to the re-derivation). Returns (errors, id_set) —
    rule-level errors only, no row content."""
    errors: list[str] = []
    if header != list(expected_header):
        errors.append(f"{label}_HEADER_MISMATCH")
        return errors, set()

    ids: list[str] = []
    for r in rows:
        dp = (r.get("id", "") or "").strip()
        if not dp:
            errors.append(f"{label}_BLANK_ID")
        else:
            ids.append(dp)
    if len(ids) != len(set(ids)):
        errors.append(f"{label}_DUPLICATE_ID")
    id_set = set(ids)
    if id_set - expected_ids:
        errors.append(f"{label}_ID_NOT_EXPECTED_SET")
    if expected_ids - id_set:
        errors.append(f"{label}_EXPECTED_ROW_MISSING")

    hash_bad = False
    edited_columns: set[str] = set()
    for r in rows:
        dp = (r.get("id", "") or "").strip()
        master = master_index.get(dp)
        if master is None or dp not in expected_ids:
            continue  # already reported above
        if (r.get("row_hash", "") or "").strip() != (
            master.get("source_row_sha256", "") or ""
        ).strip():
            hash_bad = True
        for master_col, csv_col in MASTER_TO_CSV_COLUMNS.items():
            expected = (master.get(master_col, "") or "").strip()
            actual = (r.get(csv_col, "") or "").strip()
            if actual != expected:
                edited_columns.add(csv_col)
        if low_diag is not None:
            d = low_diag[dp]
            for csv_col, expected in (
                ("machine_en_alignment_confidence", d["en"]),
                ("machine_zh_alignment_confidence", d["zh"]),
                ("machine_low_conf_sides", d["sides"]),
            ):
                if (r.get(csv_col, "") or "").strip() != expected:
                    edited_columns.add(csv_col)
    if hash_bad:
        errors.append(f"{label}_ROW_HASH_MISMATCH")
    if edited_columns:
        errors.append(
            f"{label}_MACHINE_COLUMN_EDITED (" + ", ".join(sorted(edited_columns)) + ")"
        )
    return errors, id_set


def _canonical_machine_row(master_row: dict[str, str]) -> dict[str, str]:
    """Project one master row into its canonical CSV machine cells (the
    same projection ``build_annotation_csv.build_csv_rows`` applies)."""
    row: dict[str, str] = {}
    for master_col, csv_col in MASTER_TO_CSV_COLUMNS.items():
        value = (master_row.get(master_col) or "").strip()
        if not value and csv_col not in _BLANKABLE:
            raise ValueError(
                f"blank machine cell: master column {master_col!r} "
                f"(CSV column {csv_col!r})"
            )
        if csv_col in _EXCEL_SAFE_COLUMNS:
            value = excel_safe(value)
        row[csv_col] = value
    return row


def write_merged_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    """Write with the builder's conventions (utf-8-sig, minimal quoting,
    CRLF) so the merged file is a drop-in replacement for a built file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS), lineterminator="\r\n")
        w.writeheader()
        w.writerows(rows)
    return path


def merge(
    master_rows: list[dict[str, str]],
    main_rows: list[dict[str, str]],
    main_header: list[str],
    companion_rows: list[dict[str, str]],
    companion_header: list[str],
    min_confidence: float,
) -> tuple[list[dict[str, str]], dict[str, int], list[str]]:
    """Assemble the merged review. Returns (merged_rows, counts, errors).

    On any error the merged rows are meaningless — callers must fail on a
    non-empty error list before writing anything."""
    master_index = {r["datapoint_id"]: r for r in master_rows}
    low_diag = split_low_confidence(master_rows, min_confidence)
    kept_ids = set(master_index) - set(low_diag)
    deferred_ids = set(low_diag)

    errors: list[str] = []
    main_errors, main_ids = _binding_errors(
        "MAIN", main_rows, main_header, CSV_COLUMNS, master_index, kept_ids, None
    )
    companion_errors, companion_ids = _binding_errors(
        "COMPANION",
        companion_rows,
        companion_header,
        LOW_CONF_COLUMNS,
        master_index,
        deferred_ids,
        low_diag,
    )
    errors.extend(main_errors)
    errors.extend(companion_errors)

    # Explicit disjoint-exact-union requirement (implied by the two exact
    # set checks above; asserted so the contract reads in code).
    if main_ids & companion_ids:
        errors.append("FILE_OVERLAP")
    if (main_ids | companion_ids) != set(master_index):
        errors.append("UNION_NOT_MASTER")

    if errors:
        return [], {}, errors

    main_by_id = {(r.get("id", "") or "").strip(): r for r in main_rows}
    companion_by_id = {(r.get("id", "") or "").strip(): r for r in companion_rows}

    merged: list[dict[str, str]] = []
    for master_row in master_rows:  # canonical master order
        dp = master_row["datapoint_id"]
        source = main_by_id[dp] if dp in main_ids else companion_by_id[dp]
        row = _canonical_machine_row(master_row)
        for col in ANNOTATOR_COLUMNS:
            # Human cells verbatim — never stripped, never repaired.
            row[col] = source.get(col, "")
        merged.append({col: row[col] for col in CSV_COLUMNS})

    counts = {
        "rows": len(merged),
        "from_main": len(main_ids),
        "from_companion": len(companion_ids),
    }
    return merged, counts, errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review-csv", type=Path, required=True,
                    help="returned annotator CSV (annotation_pairs.csv, filled in)")
    ap.add_argument("--companion-csv", type=Path, required=True,
                    help="returned low-confidence companion CSV (adjudicated)")
    ap.add_argument("--master-tsv", type=Path, required=True)
    ap.add_argument("--min-confidence", type=float, required=True,
                    help="the split threshold the builder used (re-derived here; "
                         "must match the pair's build, e.g. 0.40)")
    ap.add_argument("--output", type=Path, required=True,
                    help="merged complete review CSV (CSV_COLUMNS only)")
    ap.add_argument("--force-output", action="store_true",
                    help="Required if the output file already exists.")
    args = ap.parse_args(argv)

    if not 0.0 <= args.min_confidence <= 1.0:
        return _exit("MIN_CONFIDENCE_RANGE", str(args.min_confidence))
    for label, p in (
        ("REVIEW_CSV_ABSENT", args.review_csv),
        ("COMPANION_CSV_ABSENT", args.companion_csv),
        ("MASTER_TSV_ABSENT", args.master_tsv),
    ):
        if not p.exists():
            return _exit(label, str(p))
    if args.output.exists() and not args.force_output:
        return _exit("OUTPUT_EXISTS", f"{args.output}; pass --force-output to overwrite")

    try:
        master_rows = read_master_rows(args.master_tsv)
        main_rows, main_header = read_returned(args.review_csv)
        companion_rows, companion_header = read_returned(args.companion_csv)
    except ValueError as exc:
        return _exit("INPUT_INVALID", str(exc))

    try:
        merged, counts, errors = merge(
            master_rows, main_rows, main_header,
            companion_rows, companion_header, args.min_confidence,
        )
    except ValueError as exc:
        return _exit("MASTER_INVALID", str(exc))

    if errors:
        print(f"violations: {len(errors)}")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  … and {len(errors) - 20} more")
        return EXIT_INPUT_ERROR

    write_merged_csv(args.output, merged)

    # Aggregate stdout only — no text, lemmas, or ids.
    print(f"rows: {counts['rows']}")
    print(f"from_main: {counts['from_main']}")
    print(f"from_companion: {counts['from_companion']}")
    print(f"columns: {len(CSV_COLUMNS)} ({len(MASTER_TO_CSV_COLUMNS)} machine + "
          f"{len(ANNOTATOR_COLUMNS)} annotator; "
          f"{len(LOW_CONF_EXTRA_COLUMNS)} companion diagnostic column(s) dropped)")
    print(f"output: {args.output}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
