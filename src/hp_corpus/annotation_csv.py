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

from hp_corpus.candidates import CONTEXT_PROVENANCES
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
    "english_context_provenance",  # how the EN context was retrieved
    "chinese_context",
    "chinese_context_provenance",  # how the ZH context was retrieved
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

# Cells starting with these characters make spreadsheet apps parse the
# cell as a formula (Excel renders "- text" as #NAME? and destroys the
# text on save). The writer prefixes such cells with a single space.
_EXCEL_UNSAFE_PREFIXES = ("=", "+", "-", "@")


def excel_safe(cell: str) -> str:
    """Prefix one space when a cell would be parsed as a formula.

    Applied to machine text columns at CSV write time. The returned-file
    validator compares cells after ``.strip()``, so a round trip through
    the prefix stays bound to the master value.
    """
    if cell.startswith(_EXCEL_UNSAFE_PREFIXES):
        return " " + cell
    return cell


# master column -> CSV machine column (used by the builder and the
# returned-file integrity check). The context columns are the retrieval
# view (anchor ± window, or a neighbour fallback), not the bare aligned
# anchor — annotators need the surrounding sentences to locate
# counterparts that the strict DP left outside the anchor group.
MASTER_TO_CSV_COLUMNS: dict[str, str] = {
    "datapoint_id": "id",
    "chapter": "chapter",
    "de_form": "form",
    "de_pp_surface": "german_pp",
    "de_head_lemma": "german_head_lemma",
    "de_sentence_text": "german_sentence",
    "en_context_text": "english_context",
    "en_context_provenance": "english_context_provenance",
    "zh_context_text": "chinese_context",
    "zh_context_provenance": "chinese_context_provenance",
    "source_row_sha256": "row_hash",
}

# Machine columns whose values pass through :func:`excel_safe` at write
# time — every mapped column that carries free corpus text (ids, hashes,
# vocabularies, and numbers never start with a formula character).
_EXCEL_SAFE_COLUMNS = frozenset(
    {
        "german_pp",
        "german_head_lemma",
        "german_sentence",
        "english_context",
        "chinese_context",
    }
)

# Diagnostic columns appended (after CSV_COLUMNS) to the low-confidence
# companion CSV. They restate master machine cells so the excluded rows can
# be eyeballed and adjudicated without joining back to the master; the
# return-gate validator re-derives them cell by cell like any machine column.
LOW_CONF_EXTRA_COLUMNS: tuple[str, ...] = (
    "machine_en_alignment_confidence",
    "machine_zh_alignment_confidence",
    "machine_low_conf_sides",  # "en" | "zh" | "en,zh"
)
LOW_CONF_COLUMNS: tuple[str, ...] = CSV_COLUMNS + LOW_CONF_EXTRA_COLUMNS


def split_low_confidence(
    master_rows: list[dict[str, str]], min_confidence: float
) -> dict[str, dict[str, str]]:
    """Partition key for the confidence split of the annotator deliverable.

    A row is low-confidence when the EN **or** ZH machine alignment
    confidence is below ``min_confidence`` — the annotator needs both
    retrieval contexts reliable, so either side failing is enough to defer
    the row. Returns ``{datapoint_id: {en, zh, sides}}`` with the master's
    confidence strings verbatim (no float round-trip, so the builder and
    the validator compare identical cell text). Raises ValueError on a
    missing or unparseable confidence cell — the split must fail closed,
    never silently keep a row whose confidence is unknown."""
    diag: dict[str, dict[str, str]] = {}
    for r in master_rows:
        dp = r.get("datapoint_id", "")
        sides: list[str] = []
        confs: dict[str, str] = {}
        for side in ("en", "zh"):
            raw = (r.get(f"{side}_alignment_confidence") or "").strip()
            try:
                value = float(raw)
            except ValueError:
                raise ValueError(
                    f"unparseable machine {side}_alignment_confidence: {raw!r}"
                ) from None
            confs[side] = raw
            if value < min_confidence:
                sides.append(side)
        if sides:
            diag[dp] = {"en": confs["en"], "zh": confs["zh"], "sides": ",".join(sides)}
    return diag


__all__ = [
    "MACHINE_COLUMNS",
    "ANNOTATOR_COLUMNS",
    "CSV_COLUMNS",
    "DE_VALID_VALUES",
    "EXCLUSION_REASONS",
    "EN_FORMS",
    "ZH_FORMS",
    "ALIGNMENT_CONFIDENCES",
    "CONTEXT_PROVENANCES",
    "excel_safe",
    "MASTER_TO_CSV_COLUMNS",
    "LOW_CONF_EXTRA_COLUMNS",
    "LOW_CONF_COLUMNS",
    "split_low_confidence",
]
