"""Build the human-fillable adjudication ledger from a comparison TSV.

Reads ``<out-stem>.disagreements.tsv`` produced by ``compare_annotations.py``
and emits one row per disagreement with blank adjudication fields for the
human adjudicator to fill in.

Ledger columns:

  * ``datapoint_id``           — which row the disagreement is on
  * ``field``                  — which RESEARCH_FIELD disagrees
  * ``value_a``                — annotator A's value
  * ``value_b``                — annotator B's value
  * ``adjudicated_value``      — blank (the adjudicator fills this in)
  * ``resolution_status``      — ``pending`` (adjudicator sets to ``adjudicated``
                                 or ``rejected``)
  * ``adjudication_note``      — blank (free-text justification)

Plus a sidecar JSON ``<output>.provenance.json`` records source-file
basenames and the source-row SHA-256 so the ledger is self-tracing
without writing novel text into the TSV.

Stdout discipline: aggregate counts only. Never prints annotation values,
``datapoint_id`` values, or source text.

Usage::

    uv run python scripts/build_adjudication_ledger.py \\
        --comparison data/derived/step4/comparison.disagreements.tsv \\
        --a data/derived/step4/annotator_a.tsv \\
        --b data/derived/step4/annotator_b.tsv \\
        --output data/derived/step4/adjudication_ledger.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

EXIT_OK = 0
EXIT_INPUT_ERROR = 2

LEDGER_COLUMNS = [
    "datapoint_id",
    "field",
    "value_a",
    "value_b",
    "adjudicated_value",
    "resolution_status",
    "adjudication_note",
]


def _exit_input(rule: str, message: str) -> int:
    print(f"FAIL: rule={rule} {message}", file=sys.stderr)
    return EXIT_INPUT_ERROR


def _read_a_b_hashes(path_a: Path, path_b: Path) -> dict[str, dict[str, str]]:
    """Build datapoint_id → source_row_sha256 maps from each annotator TSV.

    Returned as ``{"a": {dp: hash, ...}, "b": {dp: hash, ...}}``.
    """
    out: dict[str, dict[str, str]] = {"a": {}, "b": {}}
    for label, path in (("a", path_a), ("b", path_b)):
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                dp = row.get("datapoint_id", "")
                h = row.get("source_row_sha256", "")
                if dp:
                    out[label][dp] = h
    return out


def build_ledger_rows(
    comparison_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Convert each disagreement row into a ledger row with blank
    adjudication fields and ``resolution_status=pending``."""
    out: list[dict[str, str]] = []
    for c in comparison_rows:
        out.append(
            {
                "datapoint_id": c.get("datapoint_id", ""),
                "field": c.get("field", ""),
                "value_a": c.get("value_a", ""),
                "value_b": c.get("value_b", ""),
                "adjudicated_value": "",
                "resolution_status": "pending",
                "adjudication_note": "",
            }
        )
    return out


def write_ledger(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=LEDGER_COLUMNS, delimiter="\t", lineterminator="\n"
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def write_provenance(
    path: Path,
    *,
    annotator_a_path: Path,
    annotator_b_path: Path,
    comparison_path: Path,
    ledger_path: Path,
    hashes_a: dict[str, str],
    hashes_b: dict[str, str],
) -> Path:
    """Write a sidecar JSON with file basenames and per-datapoint hashes.

    Datapoint IDs and SHA-256 hashes are not novel text — they are
    operational identifiers — so they are acceptable in this sidecar.
    Sentence text and annotation values are NEVER written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "annotator_a_file": annotator_a_path.name,
        "annotator_b_file": annotator_b_path.name,
        "comparison_file": comparison_path.name,
        "ledger_file": ledger_path.name,
        "source_hashes_a": hashes_a,
        "source_hashes_b": hashes_b,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comparison", type=Path, required=True)
    ap.add_argument("--a", type=Path, required=True, help="Annotator A TSV (for provenance)")
    ap.add_argument("--b", type=Path, required=True, help="Annotator B TSV (for provenance)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--force-output",
        action="store_true",
        help="Required if --output or the provenance sidecar already exists",
    )
    args = ap.parse_args(argv)

    if not args.comparison.exists():
        return _exit_input("COMPARISON_NOT_FOUND", str(args.comparison))

    provenance_path = args.output.with_suffix(args.output.suffix + ".provenance.json")
    for p in (args.output, provenance_path):
        if p.exists() and not args.force_output:
            return _exit_input("OUTPUT_EXISTS", f"{p}; pass --force-output to overwrite")

    with open(args.comparison, encoding="utf-8") as f:
        comparison_rows = list(csv.DictReader(f, delimiter="\t"))

    # Sanity: comparison TSV must have the expected columns.
    expected = {"datapoint_id", "field", "value_a", "value_b"}
    if comparison_rows:
        actual = set(comparison_rows[0].keys())
        if not expected.issubset(actual):
            return _exit_input(
                "COMPARISON_BAD_COLUMNS",
                f"expected at least {sorted(expected)}, got {sorted(actual)}",
            )

    try:
        hashes = _read_a_b_hashes(args.a, args.b)
    except FileNotFoundError as e:
        return _exit_input("ANNOTATOR_TSV_NOT_FOUND", str(e))

    ledger_rows = build_ledger_rows(comparison_rows)
    write_ledger(args.output, ledger_rows)
    write_provenance(
        provenance_path,
        annotator_a_path=args.a,
        annotator_b_path=args.b,
        comparison_path=args.comparison,
        ledger_path=args.output,
        hashes_a=hashes["a"],
        hashes_b=hashes["b"],
    )

    by_field: dict[str, int] = {}
    for r in ledger_rows:
        by_field[r["field"]] = by_field.get(r["field"], 0) + 1

    print(f"disagreements: {len(ledger_rows)}")
    print(f"unique_datapoints: {len({r['datapoint_id'] for r in ledger_rows})}")
    if by_field:
        print("by_field:")
        for f in sorted(by_field):
            print(f"  {f}: {by_field[f]}")
    print(f"ledger: {args.output}")
    print(f"provenance: {provenance_path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
