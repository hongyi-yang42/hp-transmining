"""Safe annotation-template refresh for Step 4 TSVs.

An annotator has filled in editable columns on an OLD TSV. A new
extraction / alignment run has produced a NEW blank TSV. This CLI
merges the two: copy editable fields from OLD to NEW where the row
is unchanged (matched ``datapoint_id`` AND identical
``source_row_sha256``), surface hash mismatches as conflicts (blank +
conflict-ledger record), leave new rows blank, and report removed IDs
in the JSON summary.

Safety guarantees:

  * Inputs (``--old-tsv``, ``--new-tsv``) are opened READ-ONLY. The
    script never opens either input for writing.
  * Refuses to overwrite an existing output or conflict-ledger unless
    ``--force-output`` is supplied.
  * Refuses if the two input paths are identical (``SAME_INPUT_PATHS``)
    or if ``--output`` collides with either input
    (``OUTPUT_COLLIDES_WITH_INPUT``).
  * ``OLD_MISSING_HASH`` (exit 2) if any OLD row lacks
    ``source_row_sha256``.

Stdout discipline: aggregate counts only. Never prints ``datapoint_id``
values, source text, hashes, annotation values, lemmas, or surface
forms. Hashes and IDs go to the conflict-ledger / summary JSON only.

Usage::

    uv run python scripts/refresh_annotation_template.py \\
        --old-tsv data/derived/step4/ch1_3_old.tsv \\
        --new-tsv data/derived/step4/ch1_3_new.tsv \\
        --output  data/derived/step4/ch1_3_refreshed.tsv

Optional:

    --conflict-ledger <path>   default <output>.conflicts.jsonl
    --summary <path>           default <output>.summary.json
    --force-output             required if output or ledger exists
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from hp_corpus.step4 import (
    ALL_TSV_COLUMNS,
    BUILDER_DEFAULT_EDITABLE,
    EDITABLE_COLUMNS,
)

# --------------------------------------------------------------------- constants

EXIT_OK = 0
EXIT_INPUT_ERROR = 2


def _blank_editable_value(col: str) -> str:
    """Return the value to write into an editable column when blanking a row.

    For most editable columns, blank is the empty string. For columns in
    :data:`BUILDER_DEFAULT_EDITABLE`, blank is the builder default
    (e.g. ``assumed_ok`` for the two alignment_qc columns) so the refreshed
    TSV matches the writer's initial state.
    """
    return BUILDER_DEFAULT_EDITABLE.get(col, "")


def _read_indexed_tsv(path: Path) -> list[dict[str, str]]:
    """Read a TSV via DictReader and return rows in file order.

    Raises FileNotFoundError if the path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(r) for r in reader]


def _safe_resolve(path: Path) -> Path:
    """Resolve a path for collision checks. Falls back to absolute if the
    file does not yet exist (so a not-yet-created output still compares
    against existing inputs by their absolute form)."""
    try:
        return path.resolve(strict=True)
    except FileNotFoundError:
        return path.absolute()


def _format_row_for_output(
    new_row: dict[str, str],
    *,
    copy_from: dict[str, str] | None,
) -> dict[str, str]:
    """Build an output row in ``ALL_TSV_COLUMNS`` order.

    * Source columns: always come from the NEW row (we never copy source
      fields from OLD).
    * Editable columns: if ``copy_from`` is provided, copy each editable
      field from it; otherwise write the blank value (empty string, or
      builder default for ``{en,zh}_alignment_qc``).
    """
    out: dict[str, str] = {}
    for col in ALL_TSV_COLUMNS:
        if col in EDITABLE_COLUMNS:
            if copy_from is not None:
                out[col] = copy_from.get(col, "")
            else:
                out[col] = _blank_editable_value(col)
        else:
            out[col] = new_row.get(col, "")
    return out


def _write_output_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write rows to a TSV using ``ALL_TSV_COLUMNS`` column order."""
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
        for row in rows:
            w.writerow({col: row.get(col, "") for col in ALL_TSV_COLUMNS})


def _append_conflict_ledger(
    path: Path, records: list[dict[str, Any]]
) -> None:
    """Write conflict-ledger records as JSONL. One record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def _write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


# --------------------------------------------------------------------- core merge


def refresh_template(
    *,
    old_tsv: Path,
    new_tsv: Path,
    output: Path,
    conflict_ledger: Path,
    summary_path: Path,
) -> dict[str, Any]:
    """Run the merge and return the summary dict.

    The summary is also written to ``summary_path`` and conflicts (if any)
    to ``conflict_ledger``. The output TSV is written to ``output``.

    Caller is responsible for safety checks (existence / collision /
    ``--force-output``); this function assumes they have passed.
    """
    old_rows = _read_indexed_tsv(old_tsv)
    new_rows = _read_indexed_tsv(new_tsv)

    # Verify every OLD row has source_row_sha256. Exit-2 rule:
    # OLD_MISSING_HASH. (NEW rows always come from the builder, which
    # always writes a hash; we still assert defensively.)
    for r in old_rows:
        if not r.get("source_row_sha256"):
            raise OldMissingHashError(
                "OLD row missing source_row_sha256 (cannot verify row identity)"
            )

    # Index OLD by datapoint_id for O(1) lookup.
    old_by_id: dict[str, dict[str, str]] = {}
    for r in old_rows:
        dp = r.get("datapoint_id", "")
        if dp:
            old_by_id[dp] = r

    new_by_id: dict[str, dict[str, str]] = {}
    for r in new_rows:
        dp = r.get("datapoint_id", "")
        if dp:
            new_by_id[dp] = r

    out_rows: list[dict[str, str]] = []
    conflicts: list[dict[str, Any]] = []
    matched = 0
    hash_mismatched = 0
    new_only = 0

    for new_row in new_rows:
        dp = new_row.get("datapoint_id", "")
        old_row = old_by_id.get(dp)
        if old_row is None:
            # ID not in OLD — write blank.
            out_rows.append(_format_row_for_output(new_row, copy_from=None))
            new_only += 1
            continue
        old_hash = old_row.get("source_row_sha256", "")
        new_hash = new_row.get("source_row_sha256", "")
        if old_hash == new_hash:
            # Match — copy editable from OLD onto NEW source.
            out_rows.append(_format_row_for_output(new_row, copy_from=old_row))
            matched += 1
        else:
            # Hash mismatch — write blank, log conflict (no source text).
            out_rows.append(_format_row_for_output(new_row, copy_from=None))
            conflicts.append(
                {
                    "datapoint_id": dp,
                    "old_hash": old_hash,
                    "new_hash": new_hash,
                    "chapter": new_row.get("chapter", ""),
                    "de_form": new_row.get("de_form", ""),
                }
            )
            hash_mismatched += 1

    # Removed IDs: in OLD but not in NEW.
    removed_ids = [dp for dp in old_by_id if dp not in new_by_id]

    _write_output_tsv(output, out_rows)
    if conflicts:
        _append_conflict_ledger(conflict_ledger, conflicts)

    summary: dict[str, Any] = {
        "old_tsv": old_tsv.name,
        "new_tsv": new_tsv.name,
        "output": output.name,
        "matched": matched,
        "hash_mismatched": hash_mismatched,
        "new": new_only,
        "removed": len(removed_ids),
        "total_in_new": len(new_rows),
        "total_in_old": len(old_rows),
        "removed_ids": removed_ids,
    }
    if conflicts:
        summary["conflict_ledger"] = conflict_ledger.name
    _write_summary_json(summary_path, summary)
    return summary


# --------------------------------------------------------------------- errors


class OldMissingHashError(Exception):
    """Raised when an OLD TSV row lacks ``source_row_sha256``."""


# --------------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--old-tsv", type=Path, required=True, help="OLD TSV (annotated).")
    ap.add_argument("--new-tsv", type=Path, required=True, help="NEW TSV (blank template).")
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output TSV path (refreshed template).",
    )
    ap.add_argument(
        "--conflict-ledger",
        type=Path,
        default=None,
        help="Conflict-ledger JSONL path. Defaults to <output>.conflicts.jsonl.",
    )
    ap.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Summary JSON path. Defaults to <output>.summary.json.",
    )
    ap.add_argument(
        "--force-output",
        action="store_true",
        help="Required if --output or --conflict-ledger already exists.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)

    old_tsv: Path = args.old_tsv
    new_tsv: Path = args.new_tsv
    output: Path = args.output
    conflict_ledger: Path = (
        args.conflict_ledger
        or output.with_name(output.name + ".conflicts.jsonl")
    )
    summary_path: Path = args.summary or (output.with_name(output.name + ".summary.json"))

    # --- Safety checks (exit 2 with named rules) ---

    # Same input paths.
    if _safe_resolve(old_tsv) == _safe_resolve(new_tsv):
        print("FAIL: --old-tsv and --new-tsv are the same path.", file=sys.stderr)
        print("rule: SAME_INPUT_PATHS", file=sys.stderr)
        return EXIT_INPUT_ERROR

    # Output collides with either input.
    out_resolved = _safe_resolve(output)
    if out_resolved == _safe_resolve(old_tsv) or out_resolved == _safe_resolve(new_tsv):
        print(
            "FAIL: --output collides with an input path.",
            file=sys.stderr,
        )
        print("rule: OUTPUT_COLLIDES_WITH_INPUT", file=sys.stderr)
        return EXIT_INPUT_ERROR

    # Existence of inputs.
    if not old_tsv.exists():
        print(f"FAIL: --old-tsv not found: {old_tsv}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    if not new_tsv.exists():
        print(f"FAIL: --new-tsv not found: {new_tsv}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    # Output / ledger existence without --force-output.
    if output.exists() and not args.force_output:
        print(
            f"FAIL: --output already exists: {output} (pass --force-output to overwrite).",
            file=sys.stderr,
        )
        print("rule: OUTPUT_EXISTS", file=sys.stderr)
        return EXIT_INPUT_ERROR
    if conflict_ledger.exists() and not args.force_output:
        print(
            f"FAIL: --conflict-ledger already exists: {conflict_ledger} "
            "(pass --force-output to overwrite).",
            file=sys.stderr,
        )
        print("rule: OUTPUT_EXISTS", file=sys.stderr)
        return EXIT_INPUT_ERROR

    # --- Run merge ---

    try:
        summary = refresh_template(
            old_tsv=old_tsv,
            new_tsv=new_tsv,
            output=output,
            conflict_ledger=conflict_ledger,
            summary_path=summary_path,
        )
    except OldMissingHashError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        print("rule: OLD_MISSING_HASH", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except FileNotFoundError as e:
        print(f"FAIL: input not found: {e.filename}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    # --- Stdout: aggregate counts only (privacy) ---
    print(f"matched: {summary['matched']}")
    print(f"hash_mismatched: {summary['hash_mismatched']}")
    print(f"new: {summary['new']}")
    print(f"removed: {summary['removed']}")
    print(f"total_in_new: {summary['total_in_new']}")
    print(f"total_in_old: {summary['total_in_old']}")
    print(f"output: {output}")
    if summary["hash_mismatched"] > 0:
        print(f"conflict_ledger: {conflict_ledger}")
    print(f"summary: {summary_path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
