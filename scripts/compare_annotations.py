"""Compare two annotator TSVs field-by-field.

Joins by ``datapoint_id`` and surfaces linguistic disagreements on the
research fields. The annotator-name and workflow-status fields are NOT
compared as linguistic content.

Outputs:

  * ``<out-stem>.disagreements.tsv`` — one row per disagreement, columns
    ``datapoint_id, field, value_a, value_b``. Gitignored location
    (carries annotation values; the file itself is the work product).
  * stdout — aggregate counts only.

Refuses to compare rows whose ``source_row_sha256`` differs between the
two files (rule ``SOURCE_HASH_MISMATCH``): the annotators must be
operating on the same source rows, otherwise the disagreement is
methodological rather than linguistic.

Stdout discipline: aggregate counts only. Never prints annotation
values, ``datapoint_id`` values, or source text.

Usage::

    uv run python scripts/compare_annotations.py \\
        --a data/derived/step4/annotator_a.tsv \\
        --b data/derived/step4/annotator_b.tsv \\
        --out-stem data/derived/step4/comparison
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

EXIT_OK = 0
EXIT_INPUT_ERROR = 2

# Linguistic annotation fields compared at field level. Excludes:
#   * annotator (the annotator-name field itself)
#   * annotation_status, adjudication_status (workflow metadata)
#   * *_notes / general_notes (free text — surfaced separately if needed)
#   * de_candidate_decision / de_exclusion_reason / de_candidate_notes
#     (German vetting, not translation annotation)
RESEARCH_FIELDS: tuple[str, ...] = (
    "en_alignment_qc",
    "en_alignment_relation",
    "en_span_text",
    "en_char_ranges",
    "en_form",
    "en_confidence",
    "zh_alignment_qc",
    "zh_alignment_relation",
    "zh_span_text",
    "zh_char_ranges",
    "zh_form",
    "zh_confidence",
)

DISAGREEMENT_COLUMNS = ["datapoint_id", "field", "value_a", "value_b"]


def _read_indexed(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _exit_input(rule: str, message: str) -> int:
    print(f"FAIL: rule={rule} {message}", file=sys.stderr)
    return EXIT_INPUT_ERROR


def compare(
    rows_a: list[dict[str, str]],
    rows_b: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, int], list[str]]:
    """Compare two annotator row lists.

    Returns ``(disagreements, counts, hash_mismatch_ids)``.

      * ``disagreements`` — list of dicts with keys ``datapoint_id``,
        ``field``, ``value_a``, ``value_b``.
      * ``counts`` — aggregate counts: ``total_comparisons``,
        ``agreements``, ``disagreements``, ``both_blank``,
        ``a_only_filled``, ``b_only_filled``.
      * ``hash_mismatch_ids`` — list of ``datapoint_id`` values where the
        source hashes differ (caller refuses to proceed in that case).
    """
    a_by_id = {r["datapoint_id"]: r for r in rows_a}
    b_by_id = {r["datapoint_id"]: r for r in rows_b}

    only_a = sorted(set(a_by_id) - set(b_by_id))
    only_b = sorted(set(b_by_id) - set(a_by_id))
    shared = sorted(set(a_by_id) & set(b_by_id))

    hash_mismatches: list[str] = []
    for dp in shared:
        ha = a_by_id[dp].get("source_row_sha256", "")
        hb = b_by_id[dp].get("source_row_sha256", "")
        if ha != hb:
            hash_mismatches.append(dp)

    disagreements: list[dict[str, str]] = []
    counts = {
        "total_comparisons": 0,
        "agreements": 0,
        "disagreements": 0,
        "both_blank": 0,
        "a_only_filled": 0,
        "b_only_filled": 0,
    }
    by_field_disagree: dict[str, int] = {f: 0 for f in RESEARCH_FIELDS}

    for dp in shared:
        ra = a_by_id[dp]
        rb = b_by_id[dp]
        for field in RESEARCH_FIELDS:
            va = ra.get(field, "")
            vb = rb.get(field, "")
            counts["total_comparisons"] += 1
            if va == vb:
                counts["agreements"] += 1
                if not va:
                    counts["both_blank"] += 1
                continue
            # Differ. Classify.
            if not va and vb:
                counts["a_only_filled"] += 1
            elif va and not vb:
                counts["b_only_filled"] += 1
            # Whether blank-mismatch or value-mismatch, it counts as a
            # disagreement (per spec: blank vs nonblank = disagreement).
            counts["disagreements"] += 1
            by_field_disagree[field] += 1
            disagreements.append(
                {"datapoint_id": dp, "field": field, "value_a": va, "value_b": vb}
            )

    counts["only_in_a"] = len(only_a)
    counts["only_in_b"] = len(only_b)
    counts["shared"] = len(shared)
    counts["by_field_disagreements"] = by_field_disagree
    return disagreements, counts, hash_mismatches


def write_disagreements_tsv(path: Path, disagreements: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=DISAGREEMENT_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        w.writeheader()
        for d in disagreements:
            w.writerow(d)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", type=Path, required=True, help="Annotator A TSV")
    ap.add_argument("--b", type=Path, required=True, help="Annotator B TSV")
    ap.add_argument(
        "--out-stem",
        type=Path,
        required=True,
        help="Output stem; <stem>.disagreements.tsv will be written",
    )
    ap.add_argument(
        "--force-output",
        action="store_true",
        help="Required if the output path already exists",
    )
    args = ap.parse_args(argv)

    if args.a == args.b:
        return _exit_input("SAME_INPUT_PATHS", "--a and --b must differ")
    if args.out_stem == args.a or args.out_stem == args.b:
        return _exit_input("OUTPUT_COLLIDES_WITH_INPUT", "--out-stem must not equal --a or --b")

    out_path = args.out_stem.with_suffix(args.out_stem.suffix + ".disagreements.tsv")
    if out_path.exists() and not args.force_output:
        return _exit_input("OUTPUT_EXISTS", f"{out_path}; pass --force-output to overwrite")

    try:
        rows_a = _read_indexed(args.a)
    except FileNotFoundError:
        return _exit_input("A_NOT_FOUND", str(args.a))
    try:
        rows_b = _read_indexed(args.b)
    except FileNotFoundError:
        return _exit_input("B_NOT_FOUND", str(args.b))

    # Header sanity — both files must be Step 4 TSVs.
    for label, rows, path in (("a", rows_a, args.a), ("b", rows_b, args.b)):
        if not rows:
            return _exit_input("EMPTY_TSV", f"{label}={path} has no body rows")
        for required in ("datapoint_id", "source_row_sha256", *RESEARCH_FIELDS):
            if required not in rows[0]:
                return _exit_input(
                    "MISSING_COLUMN",
                    f"{label}={path} missing column {required!r}",
                )

    disagreements, counts, hash_mismatches = compare(rows_a, rows_b)
    if hash_mismatches:
        # Refuse: source-row hashes differ between annotators — they were
        # not operating on the same template. Print count + IDs-only
        # stderr (datapoint IDs are not novel text, but route to stderr
        # for clarity).
        print(
            f"FAIL: rule=SOURCE_HASH_MISMATCH count={len(hash_mismatches)}",
            file=sys.stderr,
        )
        return EXIT_INPUT_ERROR

    write_disagreements_tsv(out_path, disagreements)

    # Aggregate stdout — no annotation values, no datapoint IDs.
    print(f"shared: {counts['shared']}")
    print(f"only_in_a: {counts['only_in_a']}")
    print(f"only_in_b: {counts['only_in_b']}")
    print(f"total_comparisons: {counts['total_comparisons']}")
    print(f"agreements: {counts['agreements']}")
    print(f"  both_blank: {counts['both_blank']}")
    print(f"disagreements: {counts['disagreements']}")
    print(f"  a_only_filled: {counts['a_only_filled']}")
    print(f"  b_only_filled: {counts['b_only_filled']}")
    by_field = counts["by_field_disagreements"]
    non_zero = {f: c for f, c in by_field.items() if c > 0}
    if non_zero:
        print("by_field_disagreements:")
        for f in sorted(non_zero):
            print(f"  {f}: {non_zero[f]}")
    print(f"disagreements_tsv: {out_path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
