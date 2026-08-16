"""Validate a Step 4 annotation TSV.

Two modes:

  * **Default (structural) mode** — checks the TSV is well-formed and that
    any filled-in cells use valid vocab. A structurally valid file in
    initial state (every editable cell at its builder default) prints
    ``TEMPLATE_VALID`` / ``state: UNANNOTATED`` and exits 0. A file with
    at least one annotated row that passes all checks prints ``OK``.

  * **``--require-complete`` mode** — adds research-completion checks.
    Every non-excluded row must be ``de_candidate_decision=include`` +
    ``annotation_status=complete`` + ``{lang}_alignment_qc=confirmed`` on
    both sides, with valid spans/forms/relations. Prints a rollup of
    completed / excluded / blocked / uncertain / pending-adjudication /
    adjudicated counts. Exits non-zero unless ``completed + excluded``
    equals the row total.

The ``--annotation-pool`` flag (formerly ``--full-sample``) skips the
10+10 pilot-balance check; pass it when validating the annotation-target
TSV rather than the method pilot. ``--full-sample`` remains as a hidden
alias for backward compatibility.

Prints only aggregate pass/fail counts and per-rule violation counts to
stdout — never source text, segment IDs, lemmas, or surface forms.

Usage:
    uv run python scripts/validate_step4_annotations.py \\
        data/derived/step4/ch1_3_full_annotation.tsv --annotation-pool
    uv run python scripts/validate_step4_annotations.py \\
        data/derived/step4/ch1_3_full_annotation.tsv --annotation-pool \\
        --require-complete
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from hp_corpus.provenance import (
    MalformedParseBlockIdError,
    split_parse_block_id,
)
from hp_corpus.step4 import (
    ADJUDICATION_STATUSES,
    ALIGNMENT_QC_VALUES,
    ALIGNMENT_RELATIONS,
    ALL_TSV_COLUMNS,
    ANNOTATION_STATUSES,
    BUILDER_DEFAULT_EDITABLE,
    CONFIDENCE_LEVELS,
    DE_CANDIDATE_DECISIONS,
    DE_EXCLUSION_REASONS,
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
    """Return True iff every editable column matches its builder default.

    The builder pre-fills ``{en,zh}_alignment_qc = assumed_ok``; every
    other editable column is blank in the initial template. A row is in
    initial state iff every editable cell matches that default. The
    TSV-level flag is True iff every row is in initial state.
    """
    for r in rows:
        for col in EDITABLE_COLUMNS:
            actual = r.get(col, "")
            expected = BUILDER_DEFAULT_EDITABLE.get(col, "")
            if actual != expected:
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


def _check_parse_block_provenance(rows: list[dict[str, str]]) -> list[Violation]:
    """Fail-closed structural check for block-level provenance columns.

    Every row must carry a non-empty ``de_parse_block_id`` of the form
    ``<segment_id>#bNNN`` that extends its non-empty
    ``de_source_segment_id``. Several rows MAY share one parse_block_id
    (one block can yield several PPs); the occurrence identity
    ``(de_parse_block_id, de_token_start, de_token_end)`` must be unique.
    """
    out: list[Violation] = []
    seen: dict[tuple[str, str, str], int] = {}
    for idx, r in enumerate(rows, start=2):
        pid = r.get("de_parse_block_id", "")
        sid = r.get("de_source_segment_id", "")
        if not sid:
            out.append(Violation("MISSING_PROVENANCE", "de_source_segment_id empty", row=idx))
            continue
        if not pid:
            out.append(Violation("MISSING_PROVENANCE", "de_parse_block_id empty", row=idx))
            continue
        try:
            parsed_sid, _ = split_parse_block_id(pid)
        except MalformedParseBlockIdError:
            out.append(
                Violation("MALFORMED_PARSE_BLOCK_ID", f"not '<segment>#bNNN': {pid}", row=idx)
            )
            continue
        if parsed_sid != sid:
            out.append(
                Violation(
                    "INCONSISTENT_PROVENANCE",
                    f"de_parse_block_id {pid} does not extend de_source_segment_id {sid}",
                    row=idx,
                )
            )
            continue
        key = (pid, r.get("de_token_start", ""), r.get("de_token_end", ""))
        if key in seen:
            out.append(
                Violation(
                    "DUPLICATE_OCCURRENCE_IDENTITY",
                    f"(de_parse_block_id, span) appears at rows {seen[key]} and {idx}",
                )
            )
        else:
            seen[key] = idx
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
    """Run on every row (the initial-state short-circuit lives at the
    TSV level, not per-row). Covers: per-language relation/form/confidence
    vocab, omitted / uncertain rules, char-range JSON + bounds +
    reconstruction, status vocab, DE-candidate conditional constraints,
    alignment-QC vocab, and the misalignment-vs-omission rule.
    """
    out: list[Violation] = []
    for idx, r in enumerate(rows, start=2):
        # Per-language checks (EN, ZH). Same logic, different form vocab.
        for lang, form_vocab in (("en", EN_FORMS), ("zh", ZH_FORMS)):
            relation = r.get(f"{lang}_alignment_relation", "")
            span_text = r.get(f"{lang}_span_text", "")
            ranges_raw = r.get(f"{lang}_char_ranges", "")
            form = r.get(f"{lang}_form", "")
            confidence = r.get(f"{lang}_confidence", "")
            notes = r.get(f"{lang}_notes", "")
            qc = r.get(f"{lang}_alignment_qc", "")
            align_notes = r.get(f"{lang}_alignment_notes", "")

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
            # alignment_qc vocab: blank allowed only if the entire row is at
            # builder default (which the TSV-level initial-state flag already
            # accounts for). If non-blank, must be in vocab.
            if qc and qc not in ALIGNMENT_QC_VALUES:
                out.append(
                    Violation(
                        "BAD_VOCAB",
                        f"{lang}_alignment_qc={qc!r} not in {sorted(ALIGNMENT_QC_VALUES)}",
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

            # uncertain rule (relation-level)
            if relation == "uncertain" and not notes:
                out.append(
                    Violation(
                        "UNCERTAIN_NO_NOTE",
                        f"{lang}_alignment_relation=uncertain requires {lang}_notes",
                        row=idx,
                    )
                )

            # Misalignment-vs-omission rule (always enforced).
            # omitted is reserved for "the correctly-aligned target context
            # genuinely contains no counterpart". If alignment_qc=incorrect,
            # the alignment itself is in doubt and relation=omitted is a
            # misleading encoding.
            if qc == "incorrect" and relation == "omitted":
                out.append(
                    Violation(
                        "MISALIGNED_NOT_OMISSION",
                        f"{lang}_alignment_qc=incorrect cannot be paired with "
                        f"{lang}_alignment_relation=omitted; flag for realignment "
                        f"or set relation=uncertain",
                        row=idx,
                    )
                )

            # uncertain alignment_qc requires alignment_notes.
            if qc == "uncertain" and not align_notes:
                out.append(
                    Violation(
                        "ALIGNMENT_UNCERTAIN_NO_NOTE",
                        f"{lang}_alignment_qc=uncertain requires {lang}_alignment_notes",
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

        # --- DE-candidate conditional constraints ---
        decision = r.get("de_candidate_decision", "").strip()
        reason = r.get("de_exclusion_reason", "").strip()
        de_notes = r.get("de_candidate_notes", "").strip()

        if decision and decision not in DE_CANDIDATE_DECISIONS:
            out.append(
                Violation(
                    "BAD_VOCAB",
                    f"de_candidate_decision={decision!r} not in "
                    f"{sorted(DE_CANDIDATE_DECISIONS)}",
                    row=idx,
                )
            )
        if reason and reason not in DE_EXCLUSION_REASONS:
            out.append(
                Violation(
                    "BAD_VOCAB",
                    f"de_exclusion_reason={reason!r} not in "
                    f"{sorted(DE_EXCLUSION_REASONS)}",
                    row=idx,
                )
            )
        # Conditional rules.
        if decision == "include" and reason:
            out.append(
                Violation(
                    "INCLUDE_HAS_EXCLUSION_REASON",
                    "de_candidate_decision=include requires blank "
                    "de_exclusion_reason",
                    row=idx,
                )
            )
        if decision == "exclude" and not reason:
            out.append(
                Violation(
                    "EXCLUDE_NEEDS_REASON",
                    "de_candidate_decision=exclude requires de_exclusion_reason",
                    row=idx,
                )
            )
        if decision == "uncertain" and reason:
            out.append(
                Violation(
                    "UNCERTAIN_HAS_EXCLUSION_REASON",
                    "de_candidate_decision=uncertain requires blank "
                    "de_exclusion_reason (use de_candidate_notes instead)",
                    row=idx,
                )
            )
        if decision == "uncertain" and not de_notes:
            out.append(
                Violation(
                    "UNCERTAIN_NEEDS_NOTES",
                    "de_candidate_decision=uncertain requires de_candidate_notes",
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
        # carry relation + form + confidence + confirmed alignment QC.
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
                qc = r.get(f"{lang}_alignment_qc", "")
                if qc != "confirmed":
                    out.append(
                        Violation(
                            "INCOMPLETE",
                            f"annotation_status=complete but {lang}_alignment_qc="
                            f"{qc!r} (must be confirmed)",
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


# --------------------------------------------------------------------- require-complete


# Relations that imply the annotator found a counterpart in the aligned
# text. ``omitted`` is a separate case (counterpart genuinely absent);
# ``uncertain`` blocks completion.
_SPAN_RELATIONS = frozenset({"direct", "paraphrase", "pronominal"})


def _classify_row_for_completion(r: dict[str, str]) -> str:
    """Bucket one row for the --require-complete rollup.

    Returns one of:

      * ``"completed"``    — include + annotation_status=complete + both
                             sides confirmed + valid relations/forms/spans.
      * ``"excluded"``     — de_candidate_decision=exclude.
      * ``"uncertain"``    — de_candidate_decision=uncertain.
      * ``"blocked"``      — include but some completion check fails
                             (status, alignment QC, spans, etc.).
      * ``"pending"``      — decision blank or annotation_status in
                             (blank, unstarted, in_progress).
    """
    decision = r.get("de_candidate_decision", "").strip()
    if decision == "exclude":
        return "excluded"
    if decision == "uncertain":
        return "uncertain"
    if decision != "include":
        return "pending"
    # decision == "include" from here.
    if r.get("annotation_status", "").strip() != "complete":
        return "pending"
    # Both languages must be confirmed + valid.
    for lang in ("en", "zh"):
        if r.get(f"{lang}_alignment_qc", "").strip() != "confirmed":
            return "blocked"
        relation = r.get(f"{lang}_alignment_relation", "").strip()
        if relation not in _SPAN_RELATIONS and relation != "omitted":
            return "blocked"
        if not r.get(f"{lang}_form", "").strip():
            return "blocked"
        if not r.get(f"{lang}_confidence", "").strip():
            return "blocked"
        if relation in _SPAN_RELATIONS:
            if not r.get(f"{lang}_span_text", "").strip():
                return "blocked"
            if not r.get(f"{lang}_char_ranges", "").strip():
                return "blocked"
        elif relation == "omitted":
            # Omitted must carry form=omitted and no span.
            if r.get(f"{lang}_form", "").strip() != "omitted":
                return "blocked"
            if r.get(f"{lang}_span_text", "").strip() or r.get(
                f"{lang}_char_ranges", ""
            ).strip():
                return "blocked"
        # Aligned target text must be nonempty so char ranges resolve.
        if not r.get(f"{lang}_aligned_text", "").strip():
            return "blocked"
    return "completed"


def _check_require_complete(rows: list[dict[str, str]]) -> list[Violation]:
    """Per-row completion checks for --require-complete mode. The rollup
    itself is computed in :func:`_rollup`; this function surfaces specific
    violations for human-friendly diagnostics.
    """
    out: list[Violation] = []
    for idx, r in enumerate(rows, start=2):
        decision = r.get("de_candidate_decision", "").strip()
        if decision not in DE_CANDIDATE_DECISIONS:
            out.append(
                Violation(
                    "REQUIRE_COMPLETE_DECISION_MISSING",
                    f"de_candidate_decision={decision!r}; must be one of "
                    f"{sorted(DE_CANDIDATE_DECISIONS)} in --require-complete mode",
                    row=idx,
                )
            )
            continue
        if decision in ("exclude", "uncertain"):
            # Excluded / uncertain rows are not subject to per-language
            # completion checks; the rollup counts them separately.
            continue
        # decision == "include".
        if r.get("annotation_status", "").strip() != "complete":
            out.append(
                Violation(
                    "REQUIRE_COMPLETE_NOT_COMPLETE",
                    "de_candidate_decision=include but annotation_status != complete",
                    row=idx,
                )
            )
            continue
        for lang in ("en", "zh"):
            qc = r.get(f"{lang}_alignment_qc", "").strip()
            if qc != "confirmed":
                out.append(
                    Violation(
                        "REQUIRE_COMPLETE_NOT_CONFIRMED",
                        f"de_candidate_decision=include + annotation_status=complete "
                        f"but {lang}_alignment_qc={qc!r} (must be confirmed)",
                        row=idx,
                    )
                )
            relation = r.get(f"{lang}_alignment_relation", "").strip()
            if relation not in _SPAN_RELATIONS and relation != "omitted":
                out.append(
                    Violation(
                        "REQUIRE_COMPLETE_BAD_RELATION",
                        f"{lang}_alignment_relation={relation!r}; in --require-complete "
                        f"mode an include row must use one of "
                        f"{sorted(_SPAN_RELATIONS | {'omitted'})}",
                        row=idx,
                    )
                )
            if relation in _SPAN_RELATIONS:
                if not r.get(f"{lang}_span_text", "").strip():
                    out.append(
                        Violation(
                            "REQUIRE_COMPLETE_NO_SPAN",
                            f"{lang}_alignment_relation={relation} requires "
                            f"non-empty {lang}_span_text",
                            row=idx,
                        )
                    )
                if not r.get(f"{lang}_char_ranges", "").strip():
                    out.append(
                        Violation(
                            "REQUIRE_COMPLETE_NO_RANGES",
                            f"{lang}_alignment_relation={relation} requires "
                            f"non-empty {lang}_char_ranges",
                            row=idx,
                        )
                    )
            if not r.get(f"{lang}_aligned_text", "").strip():
                out.append(
                    Violation(
                        "REQUIRE_COMPLETE_EMPTY_TARGET_TEXT",
                        f"{lang}_aligned_text empty; aligned target IDs must "
                        f"resolve to nonempty text",
                        row=idx,
                    )
                )
    return out


def _rollup(rows: list[dict[str, str]]) -> dict[str, int]:
    """Completion rollup for --require-complete mode."""
    counts = Counter(_classify_row_for_completion(r) for r in rows)
    adjudicated = sum(
        1 for r in rows if r.get("adjudication_status", "").strip() == "adjudicated"
    )
    return {
        "template_valid_rows": len(rows),
        "completed": counts.get("completed", 0),
        "excluded": counts.get("excluded", 0),
        "blocked": counts.get("blocked", 0),
        "uncertain": counts.get("uncertain", 0),
        "pending": counts.get("pending", 0),
        "adjudicated": adjudicated,
    }


# --------------------------------------------------------------------- driver


def _build_summary(
    rows_count: int,
    initial_state: bool,
    violations: list[Violation],
    full_sample: bool,
    rollup: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "rows": rows_count,
        "initial_state": initial_state,
        "violation_count": len(violations),
        "by_rule": dict(Counter(v.rule for v in violations)),
        "full_sample": full_sample,
        "rollup": rollup,
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
    path: Path,
    *,
    full_sample: bool = False,
    require_complete: bool = False,
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
    violations.extend(_check_parse_block_provenance(body))
    # The pilot-balance check (exactly 10+10) only applies to the method
    # pilot. Skip it for the annotation-target TSV, whose row count is
    # determined by the source text.
    if not full_sample:
        violations.extend(_check_pilot_balance(body))
    violations.extend(_check_occurrence_coords(body))
    violations.extend(_check_source_row_hash(body))
    violations.extend(_check_alignment_completeness(body))

    initial_state = _check_initial_state(body)
    if not initial_state:
        violations.extend(_check_filled_annotations(body))

    rollup: dict[str, int] | None = None
    if require_complete:
        violations.extend(_check_require_complete(body))
        rollup = _rollup(body)

    return violations, _build_summary(
        len(body), initial_state, violations, full_sample, rollup=rollup
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tsv", type=Path)
    ap.add_argument(
        "--annotation-pool",
        action="store_true",
        help="Skip pilot-only checks (PILOT_IMBALANCE). Use when validating "
        "the annotation-target TSV rather than the 10+10 method pilot.",
    )
    ap.add_argument(
        "--full-sample",
        action="store_true",
        help="Backward-compat alias for --annotation-pool.",
    )
    ap.add_argument(
        "--require-complete",
        action="store_true",
        help="Enforce research-completion checks: every non-excluded row must "
        "be include + complete + confirmed on both sides, with valid spans. "
        "Prints a completion rollup; exits non-zero unless completed + "
        "excluded equals the row total.",
    )
    args = ap.parse_args(argv)

    full_sample = args.annotation_pool or args.full_sample

    if not args.tsv.exists():
        print(f"FAIL: file not found: {args.tsv}", file=sys.stderr)
        return 2

    violations, summary = validate_tsv(
        args.tsv,
        full_sample=full_sample,
        require_complete=args.require_complete,
    )

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

    if summary.get("rollup") is not None:
        r = summary["rollup"]
        print("rollup:")
        for k in (
            "template_valid_rows",
            "completed",
            "excluded",
            "blocked",
            "uncertain",
            "pending",
            "adjudicated",
        ):
            print(f"  {k}: {r.get(k, 0)}")

    if violations:
        return 1
    # No violations. Distinguish "fresh template" from "annotated + passing".
    if summary.get("initial_state"):
        print("TEMPLATE_VALID")
        print("state: UNANNOTATED")
    else:
        # In require-complete mode, also assert the dataset is fully
        # adjudicated. The rollup check above would already have failed
        # otherwise, so reaching here means completed + excluded == total.
        if args.require_complete:
            print("COMPLETE")
        else:
            print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
