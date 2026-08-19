"""The annotator-facing trilingual pair CSV: single source of schema.

One CSV, one pass: each row is a German PP occurrence with its English and
Chinese **retrieval context** (the aligned anchor sentence group expanded
by a bounded window in corpus order — the master's ``*_context_text``
columns, not the bare anchor), and the annotator fills the German
validity decision plus the EN/ZH counterpart marking in the same file.
The returned file is the single review source for the eligible-pool
CLI (``scripts/build_eligible_pool.py --review-csv``).

Columns are split into machine columns (filled by the builder from the
machine master; annotators must not edit them) and annotator columns
(blank on delivery). The machine ``row_hash`` column carries the
master's ``source_row_sha256`` so the returned file can be bound
row-by-row to the exact master state.

Controlled vocabularies are enforced by
``scripts/validate_annotation_csv.py`` on return.
"""

from __future__ import annotations

from hp_corpus.sampling import REVIEW_DECISIONS

# Machine-filled columns (order = column order in the CSV).
MACHINE_COLUMNS: tuple[str, ...] = (
    "id",  # = master datapoint_id; the only join key
    "chapter",
    "form",  # contracted | uncontracted
    "german_pp",
    "german_head_lemma",  # machine lemma; reference for de_corrected_lemma
    "german_sentence",
    "english_context",
    "chinese_context",
    "row_hash",  # = master source_row_sha256; annotators ignore
)

# Annotator-filled columns (blank on delivery).
ANNOTATOR_COLUMNS: tuple[str, ...] = (
    # German validity review
    "de_valid",  # include | exclude (REVIEW_DECISIONS)
    "de_corrected_lemma",  # blank = machine lemma stands
    "de_exclusion_reason",  # required when de_valid = exclude
    "de_notes",
    # English counterpart marking
    "en_counterpart",  # span text in english_context
    "en_form",
    "en_alignment_confidence",  # HUMAN confidence (not the machine score)
    "en_notes",
    # Chinese counterpart marking (same shape)
    "zh_counterpart",
    "zh_form",
    "zh_alignment_confidence",
    "zh_notes",
)

CSV_COLUMNS: tuple[str, ...] = MACHINE_COLUMNS + ANNOTATOR_COLUMNS

# --- controlled vocabularies -------------------------------------------------

DE_VALID_VALUES = REVIEW_DECISIONS  # {include, exclude} — single definition

EXCLUSION_REASONS: frozenset[str] = frozenset(
    {
        "not_target_pp",
        "not_definite",
        "extraction_error",
        "duplicate",
        "other",
    }
)

EN_FORMS: frozenset[str] = frozenset(
    {
        "definite",
        "bare_singular",
        "demonstrative",
        "indefinite",
        "possessive",
        "pronoun",
        "proper_name",
        "other",
        "omitted",
        "uncertain",
    }
)

ZH_FORMS: frozenset[str] = frozenset(
    {
        "bare",
        "demonstrative",
        "numeral_classifier",
        "possessive",
        "pronoun",
        "proper_name",
        "other",
        "omitted",
        "uncertain",
    }
)

# Human confidence that the provided aligned context is the right
# counterpart context for this PP.
ALIGNMENT_CONFIDENCES: frozenset[str] = frozenset(
    {"high", "medium", "low", "not_aligned"}
)

# master column -> CSV machine column (used by the builder and the
# returned-file integrity check). The context columns are the retrieval
# view (anchor ± window), not the bare aligned anchor — annotators need
# the surrounding sentences to locate counterparts that the strict DP
# left outside the anchor group.
MASTER_TO_CSV_COLUMNS: dict[str, str] = {
    "datapoint_id": "id",
    "chapter": "chapter",
    "de_form": "form",
    "de_pp_surface": "german_pp",
    "de_head_lemma": "german_head_lemma",
    "de_sentence_text": "german_sentence",
    "en_context_text": "english_context",
    "zh_context_text": "chinese_context",
    "source_row_sha256": "row_hash",
}

__all__ = [
    "MACHINE_COLUMNS",
    "ANNOTATOR_COLUMNS",
    "CSV_COLUMNS",
    "DE_VALID_VALUES",
    "EXCLUSION_REASONS",
    "EN_FORMS",
    "ZH_FORMS",
    "ALIGNMENT_CONFIDENCES",
    "MASTER_TO_CSV_COLUMNS",
]
