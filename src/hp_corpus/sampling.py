"""Full-novel sampling ledger for the Bremmers reproduction.

Implements the paper's Ch.1-17 sampling rule as a pure, deterministic
function over German occurrence records.

Sampling policy (see docs/FULL_NOVEL_SAMPLING.md):

  * ``U`` — every inventory-eligible, German-reviewed ``include``
    uncontracted occurrence from Ch.1-17.
  * ``C_early`` — every inventory-eligible, German-reviewed ``include``
    contracted occurrence from Ch.1-3.
  * ``C_late`` — every inventory-eligible, German-reviewed ``include``
    contracted occurrence from Ch.4-17 whose **reviewed head-noun lemma**
    occurs in ``U``.

The C_late expansion is keyed on the head-noun lemma ALONE, not on
``canonical_preposition + lemma``. The paper's §3.2 describes the
expansion in terms of occurrences involving the same noun; canonical
preposition is retained in the ledger for audit / minimal-pair analysis
but does not gate C_late membership. There is no same-preposition
restriction. There is no fuzzy or embedding-based lemma matching — only
exact equality of the effective lemma, with an explicit
``manual_lemma_override`` hook for parser-lemma errors or genuine
orthographic variants.

An ``uncertain`` or blank German decision never enters an analysis-ready
sample (rule ``blocked_german_review``). A selected row whose effective
lemma is required but missing (U rows without a lemma; C_late rows
without a lemma to match against U) is blocked under
``blocked_lemma_review``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from hp_corpus.step4 import (
    CONTRACTED_PREP_NORMALIZATION,
    DE_CANDIDATE_DECISIONS,
    PAPER_SHARED_PREPOSITIONS,
    normalize_contracted_prep,
)

# --------------------------------------------------------------------- constants

# Per the work package spec — every value the sampling_reason column may take.
SAMPLING_REASONS: frozenset[str] = frozenset(
    {
        "uncontracted_full_novel",
        "contracted_ch1_3",
        "contracted_ch4_17_noun_match",
        "contracted_ch4_17_no_noun_match",
        "outside_author_inventory",
        "excluded_by_german_review",
        "blocked_german_review",
        "blocked_lemma_review",
    }
)

SAMPLING_STATUSES: frozenset[str] = frozenset({"selected", "not_selected", "blocked"})

# Form vocabulary — kept locally to avoid importing the full step4 form
# vocab into this module's public surface.
_FORM_VALUES: frozenset[str] = frozenset({"contracted", "uncontracted"})

CHAPTER_RANGE_FULL_NOVEL = range(1, 18)  # Ch.1..17
C_EARLY_CHAPTERS = range(1, 4)  # Ch.1..3
C_LATE_CHAPTERS = range(4, 18)  # Ch.4..17


# --------------------------------------------------------------------- types


@dataclass(frozen=True)
class Occurrence:
    """One German PP occurrence as extracted + reviewed.

    ``manual_lemma_override`` may carry a hand-corrected lemma to handle
    parser-lemma errors or genuine orthographic variants; when non-blank,
    it is preferred over both ``machine_head_lemma`` and
    ``reviewed_head_lemma``.
    """

    datapoint_id: str
    chapter: int
    form: str  # "contracted" | "uncontracted"
    canonical_prep: str
    machine_head_lemma: str
    reviewed_head_lemma: str = ""
    german_candidate_decision: str = ""  # "include" | "exclude" | "uncertain" | ""
    inventory_eligible: bool = False
    source_hash: str = ""
    manual_lemma_override: str = ""


@dataclass(frozen=True)
class SamplingRow:
    """One row of the sampling ledger. Every extracted occurrence becomes
    a row, including rejected / blocked ones."""

    occurrence: Occurrence
    effective_matching_lemma: str
    sampling_selected: bool
    sampling_reason: str
    sampling_status: str  # "selected" | "not_selected" | "blocked"
    supports_late_contracted_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SampleResult:
    """Output of :func:`select_sample`."""

    ledger: tuple[SamplingRow, ...]  # every occurrence, in input order
    selected_ids: frozenset[str]
    summary: dict[str, Any]


# --------------------------------------------------------------------- helpers


def effective_lemma(o: Occurrence, *, use_reviewed_lemma: bool = True) -> str:
    """Return the lemma to use for matching.

    Priority: ``manual_lemma_override`` > reviewed (if requested and
    non-blank) > machine > blank.
    """
    if o.manual_lemma_override:
        return o.manual_lemma_override
    if use_reviewed_lemma and o.reviewed_head_lemma:
        return o.reviewed_head_lemma
    return o.machine_head_lemma or ""


def canonical_preposition(prep_surface: str) -> str:
    """Map a contracted surface form to its canonical preposition.

    ``im`` → ``in``; ``in`` → ``in``. Unknown contractions pass through.
    """
    return normalize_contracted_prep(prep_surface)


def is_inventory_eligible(canonical_prep: str) -> bool:
    """True iff the canonical preposition is in the paper's 13-item
    paired inventory (contracted + uncontracted)."""
    return canonical_prep in PAPER_SHARED_PREPOSITIONS


# --------------------------------------------------------------------- core


def _reason_for_blocked_or_excluded(o: Occurrence) -> str | None:
    """Return the reason for a row that fails pre-selection gates, or None."""
    decision = (o.german_candidate_decision or "").strip()
    if decision == "exclude":
        return "excluded_by_german_review"
    if decision == "uncertain" or decision == "":
        return "blocked_german_review"
    if decision not in DE_CANDIDATE_DECISIONS:
        # Unknown decision value — treat as blocked rather than guess.
        return "blocked_german_review"
    return None


def select_sample(
    occurrences,
    *,
    use_reviewed_lemma: bool = True,
) -> SampleResult:
    """Apply the U / C_early / C_late rule to a collection of occurrences.

    Returns a :class:`SampleResult` whose ``ledger`` contains every input
    occurrence (in input order) annotated with its sampling reason /
    status, plus a summary dict of aggregate counts.

    The function is pure and deterministic: same inputs → same outputs.
    It performs no I/O.
    """
    occ_list = list(occurrences)

    # Pass 1: pre-selection gates (German review + inventory eligibility).
    # Determine each row's reason/status, except C_late which needs the
    # U lemma set built first.
    preliminary: list[tuple[Occurrence, str, str, str]] = []
    #           (occ, effective_lemma, reason_if_pre_decided, status_if_pre_decided)
    #           reason/status are "" if the row is still in contention.
    for o in occ_list:
        eff = effective_lemma(o, use_reviewed_lemma=use_reviewed_lemma)

        pre_reason = _reason_for_blocked_or_excluded(o)
        if pre_reason is None and not o.inventory_eligible:
            pre_reason = "outside_author_inventory"
        if pre_reason is not None:
            status = "blocked" if pre_reason.startswith("blocked") else "not_selected"
            preliminary.append((o, eff, pre_reason, status))
            continue

        # Past pre-gates. Form + chapter decides between U / C_early / C_late.
        if o.form not in _FORM_VALUES:
            preliminary.append((o, eff, "blocked_german_review", "blocked"))
            continue
        if o.chapter not in CHAPTER_RANGE_FULL_NOVEL:
            # Outside the supported range — treat as not_selected.
            preliminary.append((o, eff, "outside_author_inventory", "not_selected"))
            continue
        if o.form == "uncontracted":
            # U — lemma required.
            if not eff:
                preliminary.append((o, eff, "blocked_lemma_review", "blocked"))
                continue
            preliminary.append((o, eff, "uncontracted_full_novel", "selected"))
            continue
        # form == "contracted"
        if o.chapter in C_EARLY_CHAPTERS:
            preliminary.append((o, eff, "contracted_ch1_3", "selected"))
            continue
        # C_late — needs U lemma set. Defer.
        preliminary.append((o, eff, "", ""))

    # Build U's effective lemma set.
    u_lemmas: set[str] = set()
    for _, eff, reason, _ in preliminary:
        if reason == "uncontracted_full_novel":
            u_lemmas.add(eff)

    # Pass 2: resolve C_late rows.
    rows: list[SamplingRow] = []
    for o, eff, reason, status in preliminary:
        if reason == "":
            # C_late
            if not eff:
                reason, status = "blocked_lemma_review", "blocked"
            elif eff in u_lemmas:
                reason, status = "contracted_ch4_17_noun_match", "selected"
            else:
                reason, status = "contracted_ch4_17_no_noun_match", "not_selected"
        rows.append(
            SamplingRow(
                occurrence=o,
                effective_matching_lemma=eff,
                sampling_selected=(status == "selected"),
                sampling_reason=reason,
                sampling_status=status,
            )
        )

    # Back-fill supports_late_contracted_ids for selected U rows.
    # A U row "supports" each C_late row whose effective lemma equals the
    # U row's effective lemma.
    for i, row in enumerate(rows):
        if row.sampling_reason != "uncontracted_full_novel":
            continue
        supported = [
            r.occurrence.datapoint_id
            for r in rows
            if r.sampling_reason == "contracted_ch4_17_noun_match"
            and r.effective_matching_lemma == row.effective_matching_lemma
        ]
        rows[i] = SamplingRow(
            occurrence=row.occurrence,
            effective_matching_lemma=row.effective_matching_lemma,
            sampling_selected=row.sampling_selected,
            sampling_reason=row.sampling_reason,
            sampling_status=row.sampling_status,
            supports_late_contracted_ids=tuple(sorted(supported)),
        )

    selected_ids = frozenset(r.occurrence.datapoint_id for r in rows if r.sampling_selected)

    # Aggregate summary — counts only, no lemmas or IDs in the top-level
    # keys (callers may add their own drill-downs).
    by_reason: dict[str, int] = defaultdict(int)
    by_status: dict[str, int] = defaultdict(int)
    by_form_chapter: dict[str, int] = defaultdict(int)
    for r in rows:
        by_reason[r.sampling_reason] += 1
        by_status[r.sampling_status] += 1
        if r.sampling_selected:
            by_form_chapter[f"ch{r.occurrence.chapter:02d}_{r.occurrence.form}"] += 1

    summary: dict[str, Any] = {
        "occurrence_total": len(rows),
        "selected_total": len(selected_ids),
        "by_reason": dict(by_reason),
        "by_status": dict(by_status),
        "by_form_chapter_selected": dict(by_form_chapter),
        "u_lemma_count": len(u_lemmas),
    }

    return SampleResult(
        ledger=tuple(rows),
        selected_ids=selected_ids,
        summary=summary,
    )


__all__ = [
    "Occurrence",
    "SamplingRow",
    "SampleResult",
    "SAMPLING_REASONS",
    "SAMPLING_STATUSES",
    "CHAPTER_RANGE_FULL_NOVEL",
    "C_EARLY_CHAPTERS",
    "C_LATE_CHAPTERS",
    "select_sample",
    "effective_lemma",
    "canonical_preposition",
    "is_inventory_eligible",
    # Re-exported for caller convenience; the canonical contract is in step4.
    "CONTRACTED_PREP_NORMALIZATION",
    "PAPER_SHARED_PREPOSITIONS",
]
