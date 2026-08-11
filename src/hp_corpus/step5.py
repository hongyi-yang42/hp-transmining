"""Step 5 paper-category derivation and aggregate analysis.

Eligibility — a row enters the analysis-ready denominator iff:

  * ``de_candidate_decision == "include"``
  * ``annotation_status == "complete"``
  * ``adjudication_status`` is blank or ``"adjudicated"``
  * both ``en_aligned_text`` and ``zh_aligned_text`` are non-empty

The caller is responsible for ensuring the input TSV passes the Step 4
validator's ``--require-complete`` mode; this module's eligibility check
is the in-process mirror of that rule.

Category derivation produces coarse paper-facing labels while preserving
the fine annotator labels:

  * DE: ``contracted`` / ``uncontracted`` (mirrors ``de_form``)
  * EN: ``definite`` / ``bare_singular`` / ``demonstrative`` / ``other``
  * ZH: ``bare`` / ``demonstrative`` / ``other``

The fine labels that roll up to ``other`` (``indefinite``, ``possessive``,
``pronoun``, ``proper_name``, ``other``, ``omitted``, ``uncertain`` for
EN; ``numeral_classifier``, ``possessive``, ``pronoun``, ``proper_name``,
``other``, ``omitted``, ``uncertain`` for ZH) are preserved in the
detailed TSV and enumerated in the aggregate summary's
``source_labels_rolled_up`` block, so nothing disappears silently from
the denominator.

Not implemented (deliberately):

  * weak/strong definiteness labels
  * MDS or any category-distance definition
"""

from __future__ import annotations

from collections import Counter
from typing import Any

# --------------------------------------------------------------------- constants

ELIGIBLE_DECISION = "include"
ELIGIBLE_ANNOTATION_STATUS = "complete"
ELIGIBLE_ADJUDICATION_STATUSES: frozenset[str] = frozenset({"", "adjudicated"})

# Coarse paper-facing category mappings. Keys are the annotator-facing
# fine labels; unmapped values roll up to "other".
EN_COARSE_MAP: dict[str, str] = {
    "definite": "definite",
    "bare_singular": "bare_singular",
    "demonstrative": "demonstrative",
}
ZH_COARSE_MAP: dict[str, str] = {
    "bare": "bare",
    "demonstrative": "demonstrative",
}

# Coarse category value sets (exported for callers / tests).
EN_PAPER_CATEGORIES: frozenset[str] = frozenset(
    {"definite", "bare_singular", "demonstrative", "other"}
)
ZH_PAPER_CATEGORIES: frozenset[str] = frozenset({"bare", "demonstrative", "other"})
DE_PAPER_CATEGORIES: frozenset[str] = frozenset({"contracted", "uncontracted"})

# Detailed TSV columns produced by :func:`derive_rows`.
DERIVED_COLUMNS: tuple[str, ...] = (
    "datapoint_id",
    "chapter",
    "de_form",
    "de_paper_category",
    "en_form_fine",
    "en_paper_category",
    "zh_form_fine",
    "zh_paper_category",
)


# --------------------------------------------------------------------- predicates


def is_analysis_ready(row: dict[str, str]) -> bool:
    """Return True iff the row enters the analysis-ready denominator."""
    decision = row.get("de_candidate_decision", "").strip()
    status = row.get("annotation_status", "").strip()
    adjud = row.get("adjudication_status", "").strip()
    en_text = row.get("en_aligned_text", "").strip()
    zh_text = row.get("zh_aligned_text", "").strip()
    return (
        decision == ELIGIBLE_DECISION
        and status == ELIGIBLE_ANNOTATION_STATUS
        and adjud in ELIGIBLE_ADJUDICATION_STATUSES
        and bool(en_text)
        and bool(zh_text)
    )


# --------------------------------------------------------------------- derivation


def de_paper_category(row: dict[str, str]) -> str:
    """Coarse DE category — mirrors ``de_form``."""
    return row.get("de_form", "").strip()


def en_paper_category(row: dict[str, str]) -> str:
    """Coarse EN category. Unmapped fine labels roll up to ``other``."""
    fine = row.get("en_form", "").strip()
    return EN_COARSE_MAP.get(fine, "other")


def zh_paper_category(row: dict[str, str]) -> str:
    """Coarse ZH category. Unmapped fine labels roll up to ``other``."""
    fine = row.get("zh_form", "").strip()
    return ZH_COARSE_MAP.get(fine, "other")


def derive_row(row: dict[str, str]) -> dict[str, str]:
    """Project one master row into the derived-category shape."""
    return {
        "datapoint_id": row.get("datapoint_id", ""),
        "chapter": row.get("chapter", ""),
        "de_form": row.get("de_form", ""),
        "de_paper_category": de_paper_category(row),
        "en_form_fine": row.get("en_form", ""),
        "en_paper_category": en_paper_category(row),
        "zh_form_fine": row.get("zh_form", ""),
        "zh_paper_category": zh_paper_category(row),
    }


def derive_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Filter to analysis-ready rows and derive categories for each."""
    return [derive_row(r) for r in rows if is_analysis_ready(r)]


# --------------------------------------------------------------------- analysis


def distribution(derived: list[dict[str, str]], key: str) -> dict[str, dict[str, float]]:
    """Count + row-percentage for one category column.

    Denominator is the number of analysis-ready rows (= ``len(derived)``).
    Percentages are rounded to two decimals.
    """
    n = len(derived)
    counts: Counter[str] = Counter(r[key] for r in derived)
    return {
        k: {"count": c, "percent": round(100.0 * c / n, 2) if n else 0.0}
        for k, c in sorted(counts.items())
    }


def cross_tab(
    derived: list[dict[str, str]], row_key: str, col_key: str
) -> dict[str, dict[str, dict[str, float]]]:
    """Two-way count + row-percentage table.

    Returns ``{row_category: {col_category: {"count": int, "percent": float}}}``
    where the percentage denominator is the row-category subtotal (so each
    inner row sums to ~100%). Also includes a ``"_denominators"`` block at
    the top level mapping each row_category to its subtotal count.
    """
    n = len(derived)
    sub: Counter[str] = Counter(r[row_key] for r in derived)
    cells: Counter[tuple[str, str]] = Counter(
        (r[row_key], r[col_key]) for r in derived
    )
    out: dict[str, dict[str, dict[str, float]]] = {}
    for row_cat, row_n in sorted(sub.items()):
        out[row_cat] = {}
        for (rc, col_cat), c in sorted(cells.items()):
            if rc != row_cat:
                continue
            denom = row_n if row_n else 1
            out[row_cat][col_cat] = {
                "count": c,
                "percent": round(100.0 * c / denom, 2),
            }
    out["_denominators"] = {  # type: ignore[assignment]
        k: {"row_total": v, "percent_of_total": round(100.0 * v / n, 2) if n else 0.0}
        for k, v in sorted(sub.items())
    }
    return out


def source_labels_rolled_up(derived: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    """Count every fine label that contributed to each coarse ``other``
    bucket. Ensures nothing disappears silently from the denominator."""
    en_other: Counter[str] = Counter(
        r["en_form_fine"]
        for r in derived
        if r["en_paper_category"] == "other"
    )
    zh_other: Counter[str] = Counter(
        r["zh_form_fine"]
        for r in derived
        if r["zh_paper_category"] == "other"
    )
    return {
        "en_other": dict(sorted(en_other.items())),
        "zh_other": dict(sorted(zh_other.items())),
    }


def uncontracted_mandarin_bare_ids(derived: list[dict[str, str]]) -> list[str]:
    """The ``uncontracted + Mandarin bare`` review list — datapoint IDs
    only. The full row text lives in the gold TSV."""
    return [
        r["datapoint_id"]
        for r in derived
        if r["de_paper_category"] == "uncontracted"
        and r["zh_paper_category"] == "bare"
    ]


def analyze(derived: list[dict[str, str]]) -> dict[str, Any]:
    """Build the aggregate Step 5 summary from derived rows."""
    n = len(derived)
    return {
        "analysis_ready_total": n,
        "de_distribution": distribution(derived, "de_paper_category"),
        "en_distribution": distribution(derived, "en_paper_category"),
        "zh_distribution": distribution(derived, "zh_paper_category"),
        "de_x_zh": cross_tab(derived, "de_paper_category", "zh_paper_category"),
        "de_x_en": cross_tab(derived, "de_paper_category", "en_paper_category"),
        "source_labels_rolled_up": source_labels_rolled_up(derived),
        "uncontracted_mandarin_bare_count": len(uncontracted_mandarin_bare_ids(derived)),
    }


__all__ = [
    "ELIGIBLE_DECISION",
    "ELIGIBLE_ANNOTATION_STATUS",
    "ELIGIBLE_ADJUDICATION_STATUSES",
    "EN_PAPER_CATEGORIES",
    "ZH_PAPER_CATEGORIES",
    "DE_PAPER_CATEGORIES",
    "EN_COARSE_MAP",
    "ZH_COARSE_MAP",
    "DERIVED_COLUMNS",
    "is_analysis_ready",
    "de_paper_category",
    "en_paper_category",
    "zh_paper_category",
    "derive_row",
    "derive_rows",
    "distribution",
    "cross_tab",
    "source_labels_rolled_up",
    "uncontracted_mandarin_bare_ids",
    "analyze",
]
