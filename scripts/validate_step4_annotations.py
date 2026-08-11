"""Validate a Step 4 pilot annotation TSV.

Runs structural + (if any annotation is filled in) annotation checks.
Prints only aggregate pass/fail counts and per-rule violation counts to
stdout — never source text, segment IDs, lemmas, or surface forms.

Exits non-zero on any structural violation or filled-in annotation
violation. Exits zero when:

  * all structural checks pass, AND
  * either the TSV is in its initial unannotated state (every editable
    cell blank), or every filled-in editable cell passes the
    annotation checks.

Usage:
    uv run python scripts/validate_step4_annotations.py \\
        data/derived/step4/ch1_3_pilot_20.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from hp_corpus.step4 import (
    ADJUDICATION_STATUSES,
    ALIGNMENT_RELATIONS,
    ALL_TSV_COLUMNS,
    ANNOTATION_STATUSES,
    CONFIDENCE_LEVELS,
    EDITABLE_COLUMNS,
    EN_FORMS,
    PILOT_DEFAULT_N_CONTRACTED,
    PILOT_DEFAULT_N_UNCONTRACTED,
    SOURCE_COLUMNS,
    ZH_FORMS,
    compute_source_row_sha256,
)

# --------------------------------------------------------------------- types


class Violation:
    """One validation finding. ``message`` carries only the rule name and
    counts — never source novel text."""

    def __init__(self, rule: str, message: str, *, row: int | None = None):
        self.rule = rule
        self.message = message
        self.row = row

    def __str__(self) -> str:
        if self.row is None:
            return f"[{self.rule}] {self.message}"
        return f"[{self.rule}] row {self.row}: {self.message}"


# --------------------------------------------------------------------- parsing


_LIST_COLS = {"en_sentence_ids", "zh_sentence_ids"}
_BOOL_COLS = {"paper_final_sample", "author_resource_match", "pilot_selected"}
_INT_COLS = {"chapter", "de_token_start", "de_token_end"}
_FLOAT_COLS = {"en_alignment_confidence", "zh_alignment_confidence"}


def _parse_source_value(col: str, raw: str) -> Any:
    """Reverse the writer's serialization for one source column."""
    if col in _LIST_COLS:
        if not raw:
            return []
        try:
            v = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"column {col} is not valid JSON: {e}") from e
        if not isinstance(v, list):
            raise ValueError(f"column {col} expected list, got {type(v).__name__}")
        return v
    if col in _BOOL_COLS:
        return raw == "true"
    if col in _INT_COLS:
        return int(raw) if raw != "" else 0
    if col in _FLOAT_COLS:
        return float(raw) if raw != "" else 0.0
    return raw


def _reconstruct_candidate_from_row(row: dict[str, str]) -> dict[str, Any]:
    """Rebuild a typed candidate dict from a TSV row, using only SOURCE_COLUMNS."""
    out: dict[str, Any] = {}
    for col in SOURCE_COLUMNS:
        if col == "source_row_sha256":
            continue
        out[col] = _parse_source_value(col, row.get(col, ""))
    return out


# --------------------------------------------------------------------- checks


def _check_headers(actual: list[str]) -> list[Violation]:
    expected = list(ALL_TSV_COLUMNS)
    if actual != expected:
        return [
            Violation(
                "HEADER_MISMATCH",
                f"expected {len(expected)} columns in fixed order, got {len(actual)}",
            )
        ]
    return []


def _check_initial_state(rows: list[dict[str, str]]) -> bool:
    """Return True iff every editable column is blank in every row."""
    for r in rows:
        for col in EDITABLE_COLUMNS:
            if r.get(col, ""):
                return False
    return True


def _check_unique_datapoint_ids(rows: list[dict[str, str]]) -> list[Violation]:
    seen: dict[str, int] = {}
    out: list[Violation] = []
    for idx, r in enumerate(rows, start=2):  # row 1 is header
        dp = r.get("datapoint_id", "")
        if dp in seen:
            out.append(
                Violation(
                    "DUPLICATE_DATAPOINT_ID",
                    f"datapoint_id appears at rows {seen[dp]} and {idx}",
                )
            )
        else:
            seen[dp] = idx
    return out


def _check_pilot_balance(rows: list[dict[str, str]]) -> list[Violation]:
    by_form = Counter(r.get("de_form", "") for r in rows)
    out: list[Violation] = []
    n_c = by_form.get("contracted", 0)
    n_u = by_form.get("uncontracted", 0)
    if n_c != PILOT_DEFAULT_N_CONTRACTED:
        out.append(
            Violation(
                "PILOT_IMBALANCE",
                f"expected {PILOT_DEFAULT_N_CONTRACTED} contracted, got {n_c}",
            )
        )
    if n_u != PILOT_DEFAULT_N_UNCONTRACTED:
        out.append(
            Violation(
                "PILOT_IMBALANCE",
                f"expected {PILOT_DEFAULT_N_UNCONTRACTED} uncontracted, got {n_u}",
            )
        )
    return out


def _check_occurrence_coords(rows: list[dict[str, str]]) -> list[Violation]:
    out: list[Violation] = []
    for idx, r in enumerate(rows, start=2):
        for col in ("de_token_start", "de_token_end"):
            v = r.get(col, "")
            if v == "":
                out.append(Violation("MISSING_COORDS", f"{col} empty", row=idx))
                continue
            try:
                n = int(v)
                if n <= 0:
                    out.append(Violation("MISSING_COORDS", f"{col}={v} not positive", row=idx))
            except ValueError:
                out.append(Violation("MISSING_COORDS", f"{col}={v!r} not an int", row=idx))
        try:
            s = int(r.get("de_token_start", "0"))
            e = int(r.get("de_token_end", "0"))
            if s > e:
                out.append(Violation("MISSING_COORDS", f"start {s} > end {e}", row=idx))
        except ValueError:
            pass
    return out


def _check_source_row_hash(rows: list[dict[str, str]]) -> list[Violation]:
    out: list[Violation] = []
    for idx, r in enumerate(rows, start=2):
        try:
            cand = _reconstruct_candidate_from_row(r)
        except ValueError as e:
            out.append(Violation("SOURCE_ROW_HASH_MISMATCH", str(e), row=idx))
            continue
        recomputed = compute_source_row_sha256(cand)
        actual = r.get("source_row_sha256", "")
        if recomputed != actual:
            out.append(
                Violation(
                    "SOURCE_ROW_HASH_MISMATCH",
                    "source columns appear to have been edited "
                    "(recomputed hash does not match stored hash)",
                    row=idx,
                )
            )
    return out


def _parse_char_ranges(raw: str) -> tuple[list[list[int]], str | None]:
    """Return (ranges, error). ranges is a list of [start, end] pairs."""
    if raw == "":
        return [], None
    try:
        v = json.loads(raw)
    except json.JSONDecodeError as e:
        return [], f"invalid JSON: {e}"
    if not isinstance(v, list):
        return [], "JSON value is not a list"
    for pair in v:
        if not isinstance(pair, list) or len(pair) != 2:
            return [], "each range must be a 2-element list"
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in pair):
            return [], "range endpoints must be integers"
        if pair[0] < 0 or pair[1] < pair[0]:
            return [], "range must have 0 <= start <= end"
    return v, None


def _ranges_to_text(text: str, ranges: list[list[int]]) -> str:
    """Join span substrings with single space (discontinuous → space-joined)."""
    parts = []
    for start, end in ranges:
        parts.append(text[start:end])
    return " ".join(parts)


def _check_filled_annotations(rows: list[dict[str, str]]) -> list[Violation]:
    """Run only on rows where any editable cell is filled."""
    out: list[Violation] = []
    for idx, r in enumerate(rows, start=2):
        any_filled = any(r.get(col, "") for col in EDITABLE_COLUMNS)
        if not any_filled:
            continue
        # Per-language checks (EN, ZH). Same logic, different vocab.
        for lang, form_vocab in (("en", EN_FORMS), ("zh", ZH_FORMS)):
            relation = r.get(f"{lang}_alignment_relation", "")
            span_text = r.get(f"{lang}_span_text", "")
            ranges_raw = r.get(f"{lang}_char_ranges", "")
            form = r.get(f"{lang}_form", "")
            confidence = r.get(f"{lang}_confidence", "")
            notes = r.get(f"{lang}_notes", "")

            # relation: empty allowed (cell untouched) OR in vocab
            if relation and relation not in ALIGNMENT_RELATIONS:
                allowed = sorted(ALIGNMENT_RELATIONS)
                out.append(
                    Violation(
                        "BAD_VOCAB",
                        f"{lang}_alignment_relation={relation!r} not in {allowed}",
                        row=idx,
                    )
                )
            if form and form not in form_vocab:
                out.append(
                    Violation(
                        "BAD_VOCAB",
                        f"{lang}_form={form!r} not in {sorted(form_vocab)}",
                        row=idx,
                    )
                )
            if confidence and confidence not in CONFIDENCE_LEVELS:
                out.append(
                    Violation(
                        "BAD_VOCAB",
                        f"{lang}_confidence={confidence!r} not in {sorted(CONFIDENCE_LEVELS)}",
                        row=idx,
                    )
                )

            # omitted rule
            if relation == "omitted":
                if span_text or ranges_raw:
                    out.append(
                        Violation(
                            "OMITTED_HAS_SPAN",
                            f"{lang}_alignment_relation=omitted but span/ranges present",
                            row=idx,
                        )
                    )
                if form != "omitted":
                    out.append(
                        Violation(
                            "OMITTED_WRONG_FORM",
                            f"{lang}_alignment_relation=omitted but {lang}_form={form!r}",
                            row=idx,
                        )
                    )

            # uncertain rule
            if relation == "uncertain" and not notes:
                out.append(
                    Violation(
                        "UNCERTAIN_NO_NOTE",
                        f"{lang}_alignment_relation=uncertain requires {lang}_notes",
                        row=idx,
                    )
                )

            # char ranges JSON + bounds + reconstruction
            aligned_text = r.get(f"{lang}_aligned_text", "")
            ranges, err = _parse_char_ranges(ranges_raw)
            if err:
                out.append(Violation("BAD_RANGES_JSON", f"{lang}_char_ranges: {err}", row=idx))
            else:
                text_len = len(aligned_text)
                for start, end in ranges:
                    if end > text_len:
                        out.append(
                            Violation(
                                "RANGE_OUT_OF_BOUNDS",
                                f"{lang}_char_ranges [{start},{end}] exceeds "
                                f"{lang}_aligned_text length {text_len}",
                                row=idx,
                            )
                        )
                if ranges and not err:
                    reconstructed = _ranges_to_text(aligned_text, ranges)
                    if span_text and reconstructed != span_text:
                        out.append(
                            Violation(
                                "SPAN_MISMATCH",
                                f"{lang}_span_text does not equal "
                                f"{lang}_char_ranges reconstruction",
                                row=idx,
                            )
                        )

        # annotation_status / adjudication_status
        a_status = r.get("annotation_status", "")
        if a_status and a_status not in ANNOTATION_STATUSES:
            out.append(
                Violation(
                    "BAD_VOCAB",
                    f"annotation_status={a_status!r} not in {sorted(ANNOTATION_STATUSES)}",
                    row=idx,
                )
            )
        ad_status = r.get("adjudication_status", "")
        if ad_status and ad_status not in ADJUDICATION_STATUSES:
            out.append(
                Violation(
                    "BAD_VOCAB",
                    f"adjudication_status={ad_status!r} not in {sorted(ADJUDICATION_STATUSES)}",
                    row=idx,
                )
            )

        # complete-rule: when annotation_status=complete, both languages must
        # carry relation + form + confidence.
        if a_status == "complete":
            for lang in ("en", "zh"):
                for col in (
                    f"{lang}_alignment_relation",
                    f"{lang}_form",
                    f"{lang}_confidence",
                ):
                    if not r.get(col, ""):
                        out.append(
                            Violation(
                                "INCOMPLETE",
                                f"annotation_status=complete but {col} empty",
                                row=idx,
                            )
                        )
    return out


def _check_alignment_completeness(rows: list[dict[str, str]]) -> list[Violation]:
    """For each row, if the cardinality on the tgt side is 0 but the source
    side has data, flag it. Catches lost 1:n target sentences."""
    out: list[Violation] = []
    for idx, r in enumerate(rows, start=2):
        for lang in ("en", "zh"):
            card = r.get(f"{lang}_alignment_cardinality", "")
            status = r.get(f"{lang}_alignment_status", "")
            ids = r.get(f"{lang}_sentence_ids", "")
            # If cardinality says 1:0 but status says 'aligned' (or vice versa)
            # we have an internal inconsistency that the builder would not
            # produce. Don't trust the row.
            n_tgt = 0
            try:
                n_tgt = int(card.split(":")[1]) if ":" in card else 0
            except (ValueError, IndexError):
                pass
            if status == "aligned" and n_tgt == 0:
                out.append(
                    Violation(
                        "LOST_TGT_SENTENCE",
                        f"{lang}_alignment_status=aligned but cardinality={card!r} "
                        f"indicates no target sentence",
                        row=idx,
                    )
                )
            if status == "aligned" and not ids:
                out.append(
                    Violation(
                        "LOST_TGT_SENTENCE",
                        f"{lang}_alignment_status=aligned but {lang}_sentence_ids empty",
                        row=idx,
                    )
                )
    return out


# --------------------------------------------------------------------- driver


def _build_summary(
    rows_count: int,
    initial_state: bool,
    violations: list[Violation],
    full_sample: bool,
) -> dict[str, Any]:
    return {
        "rows": rows_count,
        "initial_state": initial_state,
        "violation_count": len(violations),
        "by_rule": dict(Counter(v.rule for v in violations)),
        "full_sample": full_sample,
    }


def _check_nonempty_body(body: list[dict[str, str]]) -> list[Violation]:
    """A header-only TSV (zero body rows) is almost certainly an upstream
    failure (missing extraction TSVs, Stanza parse error). Flag it rather
    than letting the file slip through as a valid empty annotation target.
    """
    if not body:
        return [Violation("EMPTY_BODY", "TSV has a header but zero body rows")]
    return []


def validate_tsv(
    path: Path, *, full_sample: bool = False
) -> tuple[list[Violation], dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        rows_raw = list(reader)
    if not rows_raw:
        v = [Violation("EMPTY_FILE", "TSV has no rows")]
        return v, _build_summary(0, False, v, full_sample)
    header = rows_raw[0]
    body = [dict(zip(header, row, strict=False)) for row in rows_raw[1:] if any(row)]

    violations: list[Violation] = []
    violations.extend(_check_headers(header))
    # If headers are wrong, downstream parsing may explode; bail.
    if any(v.rule == "HEADER_MISMATCH" for v in violations):
        return violations, _build_summary(0, False, violations, full_sample)

    violations.extend(_check_nonempty_body(body))
    violations.extend(_check_unique_datapoint_ids(body))
    # The pilot-balance check (exactly 10+10) only applies to the method
    # pilot. Skip it when validating a full Bremmers-sample TSV, whose
    # contracted/uncontracted counts are determined by the source text.
    if not full_sample:
        violations.extend(_check_pilot_balance(body))
    violations.extend(_check_occurrence_coords(body))
    violations.extend(_check_source_row_hash(body))
    violations.extend(_check_alignment_completeness(body))

    initial_state = _check_initial_state(body)
    if not initial_state:
        violations.extend(_check_filled_annotations(body))

    return violations, _build_summary(
        len(body), initial_state, violations, full_sample
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tsv", type=Path)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on any violation (default behavior). "
        "Without --strict, exit 0 if all failures are annotation-level only.",
    )
    ap.add_argument(
        "--full-sample",
        action="store_true",
        help="Skip pilot-only checks (PILOT_IMBALANCE). Use when validating "
        "a full Bremmers-sample TSV whose row count is not the 10+10 pilot.",
    )
    args = ap.parse_args(argv)

    if not args.tsv.exists():
        print(f"FAIL: file not found: {args.tsv}", file=sys.stderr)
        return 2

    violations, summary = validate_tsv(args.tsv, full_sample=args.full_sample)

    # Aggregate-only stdout. Never print row text.
    print(f"file: {args.tsv}")
    print(f"rows: {summary.get('rows', 0)}")
    print(f"initial_state: {summary.get('initial_state')}")
    print(f"violations: {summary.get('violation_count', 0)}")
    if summary.get("by_rule"):
        for rule, count in sorted(summary["by_rule"].items()):
            print(f"  {rule}: {count}")
    # Print which ROW had which RULE — no row content, just row index.
    # Group violations by rule for readability.
    by_rule: dict[str, list[int]] = {}
    for v in violations:
        by_rule.setdefault(v.rule, []).append(v.row if v.row is not None else -1)
    if by_rule:
        print("detail (rule → row indices, no row content):")
        for rule, rows in sorted(by_rule.items()):
            rows_str = ", ".join(str(r) for r in rows if r > 0) or "—"
            print(f"  {rule}: {rows_str}")

    if violations:
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
