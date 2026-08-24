"""Build the annotator-facing trilingual pair CSV from the machine master.

Primary deliverable: every German PP occurrence (Ch.1–17) whose EN and ZH
machine alignment both clear ``--min-confidence``, with its aligned English
and Chinese retrieval context and blank annotation columns for the
annotator to fill — German validity (include/exclude + corrected lemma) and
the EN/ZH counterpart marking (span text, form, alignment confidence,
notes). See ``docs/ANNOTATION_CSV.md`` for the annotator instructions and
vocabularies.

The returned file is the single review source for
``scripts/build_eligible_pool.py --review-csv``; validate it first with
``scripts/validate_annotation_csv.py``.

Encoding: UTF-8 with BOM (``utf-8-sig``) so the Chinese text opens
correctly in Excel. Quoting is minimal — fields containing commas or
quotes are quoted per standard CSV. Free-text machine cells that start
with ``=``/``+``/``-``/``@`` (e.g. German dialogue lines opening with a
dash) are prefixed with one space so spreadsheets parse them as text
instead of a formula; the return-gate validator compares after
``.strip()``, so the prefix round-trips.

Usage (canonical deliverable — machine-reliability split, threshold 0.40)::

    uv run python scripts/build_annotation_csv.py \\
        --master-tsv data/derived/step4/full_novel_annotation_master.tsv \\
        --output data/derived/annotation/annotation_pairs.csv \\
        --min-confidence 0.40 \\
        --low-confidence-output data/derived/annotation/annotation_pairs_low_confidence.csv

The split defers a row to the companion CSV (same columns plus three
diagnostic machine columns) when either side fails the machine-reliability
check: EN or ZH alignment confidence below the threshold, or EN/ZH context
provenance ``manual_review`` / ``neighbor_fallback`` (no normal reliable
anchor/context; see ``hp_corpus.annotation_csv.DEFER_PROVENANCES``).
Deferred rows are separated at the current machine-alignment stage for
later inspection and adjudication — not excluded from the study; the
master keeps every row, so the eligible-pool join is unaffected.
Omit both flags to emit the unsplit full-pool file (legacy behaviour).

Fail-closed rules: master file absent; duplicate master datapoint_ids;
master header missing required columns; any blank machine cell;
existing output without --force-output; only one of the two split flags.
Stdout carries aggregate counts only — never corpus text, lemmas, or ids.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from hp_corpus.annotation_csv import (
    _EXCEL_SAFE_COLUMNS,
    CSV_COLUMNS,
    LOW_CONF_COLUMNS,
    MASTER_TO_CSV_COLUMNS,
    excel_safe,
    split_low_confidence,
)

EXIT_OK = 0
EXIT_INPUT_ERROR = 2


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


def build_csv_rows(master_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Project master rows into CSV rows (machine columns filled,
    annotator columns blank). Any blank machine cell fails closed — except
    the two context columns, which are legitimately empty when the DE
    sentence has no aligned anchor on that side (the annotator then marks
    that side ``not_aligned``)."""
    blankable = ("english_context", "chinese_context")
    out: list[dict[str, str]] = []
    for r in master_rows:
        row: dict[str, str] = {}
        for master_col, csv_col in MASTER_TO_CSV_COLUMNS.items():
            value = (r.get(master_col) or "").strip()
            if not value and csv_col not in blankable:
                raise ValueError(
                    f"blank machine cell: master column {master_col!r} "
                    f"(CSV column {csv_col!r})"
                )
            if csv_col in _EXCEL_SAFE_COLUMNS:
                value = excel_safe(value)
            row[csv_col] = value
        for col in CSV_COLUMNS:
            row.setdefault(col, "")
        out.append({col: row[col] for col in CSV_COLUMNS})
    return out


def write_csv(
    path: Path, rows: list[dict[str, str]], columns: tuple[str, ...] = CSV_COLUMNS
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), lineterminator="\r\n")
        w.writeheader()
        w.writerows(rows)
    return path


def _count_by_form(rows: list[dict[str, str]]) -> dict[str, int]:
    n_by_form = {"contracted": 0, "uncontracted": 0}
    for r in rows:
        if r["form"] in n_by_form:
            n_by_form[r["form"]] += 1
    return n_by_form


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master-tsv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Defer rows whose EN or ZH machine alignment confidence is below "
        "this threshold to the low-confidence companion CSV.",
    )
    ap.add_argument(
        "--low-confidence-output",
        type=Path,
        default=None,
        help="Companion CSV path for deferred rows (required with --min-confidence).",
    )
    ap.add_argument(
        "--force-output",
        action="store_true",
        help="Required if any output file already exists.",
    )
    args = ap.parse_args(argv)

    if (args.min_confidence is None) != (args.low_confidence_output is None):
        return _exit(
            "SPLIT_FLAGS_PAIRED",
            "--min-confidence and --low-confidence-output must be passed together",
        )
    if args.min_confidence is not None and not 0.0 <= args.min_confidence <= 1.0:
        return _exit("MIN_CONFIDENCE_RANGE", str(args.min_confidence))

    if not args.master_tsv.exists():
        return _exit("MASTER_TSV_ABSENT", str(args.master_tsv))
    outputs = [args.output] + ([args.low_confidence_output] if args.low_confidence_output else [])
    existing = [str(p) for p in outputs if p.exists()]
    if existing and not args.force_output:
        return _exit("OUTPUT_EXISTS", f"{'; '.join(existing)}; pass --force-output to overwrite")

    try:
        master_rows = read_master_rows(args.master_tsv)
        low_diag: dict[str, dict[str, str]] | None = None
        if args.min_confidence is not None:
            low_diag = split_low_confidence(master_rows, args.min_confidence)
        kept_master = [r for r in master_rows if r["datapoint_id"] not in (low_diag or {})]
        low_master = [r for r in master_rows if r["datapoint_id"] in (low_diag or {})]
        csv_rows = build_csv_rows(kept_master)
        low_rows = build_csv_rows(low_master) if low_diag is not None else None
    except ValueError as exc:
        return _exit("MASTER_INVALID", str(exc))
    if not csv_rows:
        return _exit("NO_ROWS", "master has no rows")

    write_csv(args.output, csv_rows)
    if low_rows is not None and low_diag is not None:
        for r, m in zip(low_rows, low_master, strict=True):
            d = low_diag[m["datapoint_id"]]
            r["machine_en_alignment_confidence"] = d["en"]
            r["machine_zh_alignment_confidence"] = d["zh"]
            r["machine_low_conf_sides"] = d["sides"]
        write_csv(args.low_confidence_output, low_rows, columns=LOW_CONF_COLUMNS)

    # Aggregate stdout only — no text, lemmas, or ids.
    print(f"rows: {len(csv_rows)}")
    print(f"by_form: {_count_by_form(csv_rows)}")
    if low_rows is not None:
        print(f"low_confidence_rows: {len(low_rows)}")
        print(f"low_confidence_by_form: {_count_by_form(low_rows)}")
    n_cols = len(LOW_CONF_COLUMNS) if low_rows is not None else len(CSV_COLUMNS)
    print(f"columns: {n_cols} ({len(MASTER_TO_CSV_COLUMNS)} machine + "
          f"{n_cols - len(MASTER_TO_CSV_COLUMNS)} annotator/diagnostic)")
    print(f"output: {args.output}")
    if args.low_confidence_output:
        print(f"low_confidence_output: {args.low_confidence_output}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
