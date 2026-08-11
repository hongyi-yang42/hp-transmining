"""Merge two annotator TSVs plus an adjudication ledger into a gold TSV.

Algorithm:

  1. Load master, annotator-A, annotator-B, and the adjudication ledger.
  2. Group ledger disagreements by ``datapoint_id``. Each disagreement
     must have ``resolution_status=adjudicated`` and a non-blank
     ``adjudicated_value`` — otherwise REFUSE the merge.
  3. For each row in the master:
       * Start from the master's source columns.
       * Copy every editable column from annotator-A (the editable base).
       * Override with annotator-B's values for fields where A and B
         agree on a non-blank value (prefer B's encoding only when both
         are non-blank and equal — otherwise the field is either agreed
         from A or disputed and goes through the ledger).
       * For each field with an adjudicated disagreement, OVERWRITE with
         ``adjudicated_value`` from the ledger.
       * Set ``adjudication_status = "adjudicated"`` if any disagreement
         existed for this row, else copy A's value.
  4. Write the gold TSV (use ``ALL_TSV_COLUMNS`` column order).
  5. Call the Step 4 validator in ``--require-complete`` mode on the gold
     TSV. Exit non-zero if it fails. Print only its aggregate output.

Safety:

  * Source-hash mismatches across master/A/B for the same datapoint →
    REFUSE (``SOURCE_HASH_MISMATCH``).
  * Unresolved or blank adjudication → REFUSE
    (``UNRESOLVED_DISAGREEMENT``).
  * Never guess a winning annotation. If the ledger's
    ``adjudicated_value`` is blank for a disagreement, fail.
  * Refuses to overwrite an existing output unless ``--force-output``.
  * Refuses if --output collides with any input path.

Stdout discipline: aggregate counts only.

Usage::

    uv run python scripts/merge_adjudicated.py \\
        --master data/derived/step4/master.tsv \\
        --annotator-a data/derived/step4/annotator_a.tsv \\
        --annotator-b data/derived/step4/annotator_b.tsv \\
        --ledger data/derived/step4/adjudication_ledger.tsv \\
        --output data/derived/step4/gold.tsv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

from hp_corpus.step4 import (
    ALL_TSV_COLUMNS,
    EDITABLE_COLUMNS,
)

EXIT_OK = 0
EXIT_INPUT_ERROR = 2
EXIT_VALIDATION_ERROR = 1

VALIDATOR_PATH = (
    Path(__file__).resolve().parent / "validate_step4_annotations.py"
)


def _exit(rule: str, message: str, *, code: int = EXIT_INPUT_ERROR) -> int:
    print(f"FAIL: rule={rule} {message}", file=sys.stderr)
    return code


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_step4_annotations", VALIDATOR_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_indexed(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _index_by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        dp = r.get("datapoint_id", "")
        if dp:
            out[dp] = r
    return out


def _validate_ledger(
    ledger_rows: list[dict[str, str]],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Group ledger rows by datapoint_id → {field: row}.

    Returns ``(by_datapoint, blocking_ids)`` where ``blocking_ids`` is the
    list of datapoint IDs whose ledger rows are not all
    ``resolution_status=adjudicated`` with non-blank ``adjudicated_value``.
    """
    by_dp: dict[str, dict[str, str]] = {}
    blocking: list[str] = []
    for r in ledger_rows:
        dp = r.get("datapoint_id", "")
        field = r.get("field", "")
        if not dp or not field:
            continue
        by_dp.setdefault(dp, {})[field] = r
    for dp, fields in by_dp.items():
        for row in fields.values():
            status = row.get("resolution_status", "").strip()
            value = row.get("adjudicated_value", "").strip()
            if status != "adjudicated" or not value:
                blocking.append(dp)
                break
    return by_dp, blocking


def merge(
    master_rows: list[dict[str, str]],
    rows_a: list[dict[str, str]],
    rows_b: list[dict[str, str]],
    ledger_by_dp: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Build the gold row list. Returns ``(gold_rows, counts)``."""
    a_by_id = _index_by_id(rows_a)
    b_by_id = _index_by_id(rows_b)

    # Source-hash consistency check across master / A / B.
    hash_mismatches: list[str] = []
    for m in master_rows:
        dp = m.get("datapoint_id", "")
        h_m = m.get("source_row_sha256", "")
        h_a = a_by_id.get(dp, {}).get("source_row_sha256", "")
        h_b = b_by_id.get(dp, {}).get("source_row_sha256", "")
        if h_m and (h_a != h_m or h_b != h_m):
            hash_mismatches.append(dp)
    if hash_mismatches:
        # Caller will refuse; surface the count via counts dict.
        return [], {
            "rows": 0,
            "merged": 0,
            "adjudicated_rows": 0,
            "hash_mismatches": len(hash_mismatches),
            "blocking_ledger_ids": 0,
        }

    gold: list[dict[str, str]] = []
    merged_count = 0
    adjudicated_count = 0
    for m in master_rows:
        dp = m.get("datapoint_id", "")
        a_row = a_by_id.get(dp, {})

        new_row: dict[str, str] = {col: m.get(col, "") for col in ALL_TSV_COLUMNS}
        # Copy editable base from annotator A.
        for col in EDITABLE_COLUMNS:
            new_row[col] = a_row.get(col, "")
        # If A and B agree on a non-blank editable value, prefer it (no
        # change needed since A's value already matches). If they disagree
        # and the ledger has an adjudicated value, apply it below.

        dp_ledger = ledger_by_dp.get(dp, {})
        if dp_ledger:
            adjudicated_count += 1
            new_row["adjudication_status"] = "adjudicated"
            for field, ledger_row in dp_ledger.items():
                if field not in EDITABLE_COLUMNS:
                    # Skip unknown fields rather than silently inventing a
                    # column — surface as a no-op rather than corrupting
                    # the schema.
                    continue
                value = ledger_row.get("adjudicated_value", "")
                if not value:
                    # _validate_ledger already caught this; defensive.
                    return [], {
                        "rows": 0,
                        "merged": 0,
                        "adjudicated_rows": 0,
                        "hash_mismatches": 0,
                        "blocking_ledger_ids": 1,
                    }
                new_row[field] = value
        gold.append(new_row)
        merged_count += 1

    return gold, {
        "rows": len(gold),
        "merged": merged_count,
        "adjudicated_rows": adjudicated_count,
        "hash_mismatches": 0,
        "blocking_ledger_ids": 0,
    }


def write_gold(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(ALL_TSV_COLUMNS),
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--annotator-a", type=Path, required=True)
    ap.add_argument("--annotator-b", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--force-output",
        action="store_true",
        help="Required if --output already exists",
    )
    ap.add_argument(
        "--annotation-pool",
        action="store_true",
        help="Passed through to the validator (skip pilot-balance check).",
    )
    ap.add_argument(
        "--skip-validation",
        action="store_true",
        help="Do not run the validator after merging (NOT recommended).",
    )
    args = ap.parse_args(argv)

    # Path collision checks.
    inputs = {args.master, args.annotator_a, args.annotator_b, args.ledger}
    if args.output in inputs:
        return _exit("OUTPUT_COLLIDES_WITH_INPUT", "--output must not equal any input")
    if len(inputs) != 4:
        return _exit("DUPLICATE_INPUT_PATHS", "input paths must be distinct")
    if args.output.exists() and not args.force_output:
        return _exit("OUTPUT_EXISTS", f"{args.output}; pass --force-output to overwrite")

    # Read inputs.
    try:
        master_rows = _read_indexed(args.master)
        rows_a = _read_indexed(args.annotator_a)
        rows_b = _read_indexed(args.annotator_b)
    except FileNotFoundError as e:
        return _exit("INPUT_NOT_FOUND", str(e))
    if not args.ledger.exists():
        return _exit("LEDGER_NOT_FOUND", str(args.ledger))
    with open(args.ledger, encoding="utf-8") as f:
        ledger_rows = list(csv.DictReader(f, delimiter="\t"))

    # Validate ledger — every entry must be adjudicated with a value.
    ledger_by_dp, blocking_ids = _validate_ledger(ledger_rows)
    if blocking_ids:
        # Datapoint IDs are operational identifiers; safe to count, not
        # safe to print as a list (could be many). Surface count only.
        return _exit(
            "UNRESOLVED_DISAGREEMENT",
            f"count={len(blocking_ids)} (resolution_status != adjudicated or blank value)",
        )

    # Merge.
    gold_rows, counts = merge(master_rows, rows_a, rows_b, ledger_by_dp)
    if counts["hash_mismatches"]:
        return _exit(
            "SOURCE_HASH_MISMATCH",
            f"count={counts['hash_mismatches']} (master / A / B disagree)",
        )
    if counts["blocking_ledger_ids"]:
        return _exit(
            "UNRESOLVED_DISAGREEMENT",
            "ledger has unresolved or blank-adjudicated-value rows",
        )

    write_gold(args.output, gold_rows)

    # Validator pass (require-complete).
    if not args.skip_validation:
        validator = _load_validator()
        extra_flags: list[str] = []
        if args.annotation_pool:
            extra_flags.append("--annotation-pool")
        rc = validator.main([str(args.output), *extra_flags, "--require-complete"])
        if rc != 0:
            # The validator already printed its aggregate output. Surface
            # a final summary marker and exit non-zero.
            print(f"gold_validation_failed: exit={rc}", file=sys.stderr)
            return EXIT_VALIDATION_ERROR

    # Aggregate stdout.
    print(f"master_rows: {len(master_rows)}")
    print(f"annotator_a_rows: {len(rows_a)}")
    print(f"annotator_b_rows: {len(rows_b)}")
    print(f"ledger_rows: {len(ledger_rows)}")
    print(f"merged_rows: {counts['merged']}")
    print(f"adjudicated_rows: {counts['adjudicated_rows']}")
    print(f"output: {args.output}")
    if not args.skip_validation:
        print("gold_validation: OK (--require-complete)")
    else:
        print("gold_validation: SKIPPED")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
