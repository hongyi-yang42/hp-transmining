"""Validate a returned trilingual annotation CSV against its master.

The gate between the annotators and everything downstream
(``scripts/build_eligible_pool.py --review-csv`` reads exactly this
file). Checks, all fail-closed (exit 2, aggregate stdout only):

Structure + binding (run on every file, filled or not):

  * header exactly ``CSV_COLUMNS`` in order;
  * no duplicate ``id``;
  * the id set equals the master's datapoint_id set — both directions
    (no extra rows, no missing rows);
  * every ``row_hash`` equals the master's ``source_row_sha256`` for
    the same id (the returned file is bound to this exact master);
  * every machine column is re-derived from the master row
    (MASTER_TO_CSV_COLUMNS) and compared cell by cell — a returned file
    whose ``english_context`` / ``chinese_context`` / … was edited (even
    with an untouched ``row_hash``) fails closed.

Annotation content (skipped while the file is in template state —
every annotator cell blank):

  * ``de_valid`` ∈ {include, exclude}; ``exclude`` requires an
    ``de_exclusion_reason`` from the vocabulary; ``include`` requires the
    reason blank;
  * ``en_form`` / ``zh_form`` from their vocabularies;
  * coupling: a counterpart span requires a form and vice versa —
    except ``omitted`` (form set, span must be blank) and ``uncertain``
    (form set, span optional, notes required);
  * a set form requires a non-blank alignment confidence on the same
    side — ``omitted`` must stay distinguishable from retrieval failure
    via ``omitted + not_aligned`` (see docs/ANNOTATION_CSV.md);
  * ``en_alignment_confidence`` / ``zh_alignment_confidence`` ∈
    {high, medium, low, not_aligned}.

A partially filled file is NOT template state — content rules run.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from hp_corpus.annotation_csv import (
    ALIGNMENT_CONFIDENCES,
    ANNOTATOR_COLUMNS,
    CSV_COLUMNS,
    DE_VALID_VALUES,
    EN_FORMS,
    EXCLUSION_REASONS,
    MASTER_TO_CSV_COLUMNS,
    ZH_FORMS,
)

EXIT_OK = 0
EXIT_INPUT_ERROR = 2


def _exit(rule: str, message: str) -> int:
    print(f"FAIL: rule={rule} {message}", file=sys.stderr)
    return EXIT_INPUT_ERROR


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    # utf-8-sig tolerates files returned with or without the BOM (Excel
    # round-trips sometimes drop it).
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    return rows, header


def read_master_rows(path: Path) -> dict[str, dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        out: dict[str, dict[str, str]] = {}
        for row in csv.DictReader(f, delimiter="\t"):
            dp = row.get("datapoint_id", "")
            if dp:
                out[dp] = row
        return out


def _is_template(rows: list[dict[str, str]]) -> bool:
    return all(
        (r.get(col, "") or "").strip() == "" for r in rows for col in ANNOTATOR_COLUMNS
    )


def _binding_errors(
    csv_rows: list[dict[str, str]], master_rows: dict[str, dict[str, str]]
) -> list[str]:
    """Row-set binding plus machine-column integrity. Rule-level errors
    only — no row content, no ids."""
    errors: list[str] = []

    ids: list[str] = []
    for r in csv_rows:
        dp = (r.get("id", "") or "").strip()
        if not dp:
            errors.append("BLANK_ID")
        else:
            ids.append(dp)
    if len(ids) != len(set(ids)):
        errors.append("DUPLICATE_ID")

    csv_id_set = set(ids)
    master_id_set = set(master_rows)
    if csv_id_set - master_id_set:
        errors.append("ID_NOT_IN_MASTER")
    if master_id_set - csv_id_set:
        errors.append("MASTER_ROW_MISSING")

    hash_mismatch = False
    edited_columns: set[str] = set()
    for r in csv_rows:
        dp = (r.get("id", "") or "").strip()
        master = master_rows.get(dp)
        if master is None:
            continue  # already reported above
        if (r.get("row_hash", "") or "").strip() != (
            master.get("source_row_sha256", "") or ""
        ).strip():
            hash_mismatch = True
        # Re-derive every machine column from the master row; the
        # returned file must carry the master's values verbatim.
        for master_col, csv_col in MASTER_TO_CSV_COLUMNS.items():
            expected = (master.get(master_col, "") or "").strip()
            actual = (r.get(csv_col, "") or "").strip()
            if actual != expected:
                edited_columns.add(csv_col)
    if hash_mismatch:
        errors.append("ROW_HASH_MISMATCH")
    if edited_columns:
        errors.append("MACHINE_COLUMN_EDITED (" + ", ".join(sorted(edited_columns)) + ")")
    return errors


def _coupling_errors(row: dict[str, str], lang: str, forms: frozenset[str]) -> list[str]:
    span = (row.get(f"{lang}_counterpart", "") or "").strip()
    form = (row.get(f"{lang}_form", "") or "").strip()
    notes = (row.get(f"{lang}_notes", "") or "").strip()
    confidence = (row.get(f"{lang}_alignment_confidence", "") or "").strip()
    errors: list[str] = []
    if form and form not in forms:
        errors.append(f"{lang}_form={form!r} not in vocabulary")
    if form == "omitted":
        if span:
            errors.append(f"{lang}_form=omitted requires a blank {lang}_counterpart")
    elif form == "uncertain":
        if not notes:
            errors.append(f"{lang}_form=uncertain requires {lang}_notes")
    elif form:
        if not span:
            errors.append(f"{lang}_form={form!r} requires a {lang}_counterpart span")
    elif span:
        errors.append(f"{lang}_counterpart without {lang}_form")
    if form and not confidence:
        errors.append(f"{lang}_form set but {lang}_alignment_confidence blank")
    if confidence and confidence not in ALIGNMENT_CONFIDENCES:
        errors.append(f"{lang}_ALIGNMENT_CONFIDENCE_INVALID")
    return errors


def validate(
    csv_rows: list[dict[str, str]],
    header: list[str],
    master_rows: dict[str, dict[str, str]],
) -> tuple[list[str], dict[str, int]]:
    """Return (errors, summary_counts). Errors are rule-level strings
    without row content — no text, lemmas, or ids leak."""
    errors: list[str] = []

    if header != list(CSV_COLUMNS):
        errors.append("HEADER_MISMATCH")
        return errors, {"rows": len(csv_rows), "template_state": 0}

    errors.extend(_binding_errors(csv_rows, master_rows))

    counts = {
        "rows": len(csv_rows),
        "template_state": 1 if _is_template(csv_rows) else 0,
        "de_include": 0,
        "de_exclude": 0,
        "en_marked": 0,
        "zh_marked": 0,
    }

    if counts["template_state"]:
        return errors, counts

    for r in csv_rows:
        valid = (r.get("de_valid", "") or "").strip()
        reason = (r.get("de_exclusion_reason", "") or "").strip()
        if valid not in DE_VALID_VALUES:
            errors.append(f"DE_VALID_INVALID ({valid or 'blank'})")
        elif valid == "exclude":
            counts["de_exclude"] += 1
            if reason not in EXCLUSION_REASONS:
                errors.append("EXCLUDE_WITHOUT_VALID_REASON")
        else:
            counts["de_include"] += 1
            if reason:
                errors.append("INCLUDE_WITH_REASON")

        for lang, forms in (("en", EN_FORMS), ("zh", ZH_FORMS)):
            span = (r.get(f"{lang}_counterpart", "") or "").strip()
            form = (r.get(f"{lang}_form", "") or "").strip()
            if span or form:
                counts[f"{lang}_marked"] += 1
            errors.extend(_coupling_errors(r, lang, forms))

    return errors, counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path, help="the returned annotation CSV")
    ap.add_argument("--master-tsv", type=Path, required=True)
    args = ap.parse_args(argv)

    if not args.csv.exists():
        return _exit("CSV_ABSENT", str(args.csv))
    if not args.master_tsv.exists():
        return _exit("MASTER_TSV_ABSENT", str(args.master_tsv))

    csv_rows, header = read_csv_rows(args.csv)
    master_rows = read_master_rows(args.master_tsv)
    errors, counts = validate(csv_rows, header, master_rows)

    # Aggregate stdout only — rule names and counts, never row content.
    print(f"rows: {counts['rows']}")
    print(f"template_state: {bool(counts['template_state'])}")
    if counts.get("de_include") or counts.get("de_exclude"):
        print(f"de_valid: include={counts['de_include']} exclude={counts['de_exclude']}")
    if counts.get("en_marked") or counts.get("zh_marked"):
        print(f"marked: en={counts['en_marked']} zh={counts['zh_marked']}")
    print(f"violations: {len(errors)}")
    for e in errors[:20]:
        print(f"  {e}")
    if len(errors) > 20:
        print(f"  … and {len(errors) - 20} more")

    return EXIT_INPUT_ERROR if errors else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
