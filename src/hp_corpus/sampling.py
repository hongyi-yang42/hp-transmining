"""German eligible-pool rule for the Bremmers reproduction.

The formal post-review selection path, in plain terms
(``docs/FULL_NOVEL_SAMPLING.md`` is the user-facing description):

1. Read every extracted German PP occurrence (Ch.1–17, contracted and
   uncontracted).
2. **Preposition inventory** — the canonical preposition must be in the
   paper's 13-item paired inventory (contracted + uncontracted).
   ``um``/``vor`` contractions are out: they have no uncontracted
   counterpart in the paper's lists.
3. **Structural gate** (uncontracted only) — the determiner token must
   carry ``xpos=ART`` and ``deprel=det`` as propagated from the
   extractor. Wrong tags exclude the row; missing metadata is a hard
   failure (the gate never passes by default). Contracted rows have no
   determiner and are exempt.
4. **German review** — every row must carry a final human decision
   (``include`` / ``exclude``). The run that builds the pool only ever
   sees a *complete* review; the CLI enforces that and fails closed on
   any blank, ``uncertain``, or missing decision before writing
   anything.
5. **Eligible pool** (paper-literal, the paper's §2.2.1 "the same
   preposition and noun"):
   - every reviewed-``include`` **uncontracted** occurrence, Ch.1–17;
   - every reviewed-``include`` **contracted** occurrence, Ch.1–3;
   - a reviewed-``include`` **contracted** occurrence, Ch.4–17, only
     when its ``(canonical_preposition, head_lemma)`` pair also occurs
     in the reviewed-include uncontracted set. The head lemma is the
     reviewer's ``corrected_head_lemma`` when non-blank, else the
     machine lemma.

Occurrence identity is ``(chapter, source_segment_id, parse_block_id,
pp_token_start, pp_token_end)`` (block-level provenance per PR #4).
Exact duplicate rows collapse deterministically to the first; the same
identity with conflicting core fields is a hard failure.

There is exactly one selector and one run: after the German review is
complete, the pool is built once. No sampling modes, no projections,
no counterfactual selectors.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from hp_corpus.step4 import (
    CONTRACTED_PREP_NORMALIZATION,
    PAPER_SHARED_PREPOSITIONS,
    normalize_contracted_prep,
)

# --------------------------------------------------------------------- constants

# Every value the pool_reason column of the eligible-pool TSV may take.
POOL_REASONS: frozenset[str] = frozenset(
    {
        "uncontracted_all_chapters",
        "contracted_ch1_3",
        "contracted_ch4_17_pair_matched",
        "contracted_ch4_17_no_uncontracted_counterpart",
        "outside_paired_inventory",
        "failed_structural_gate",
        "excluded_by_german_review",
    }
)

# Form vocabulary — kept locally to avoid importing the full step4 form
# vocab into this module's public surface.
_FORM_VALUES: frozenset[str] = frozenset({"contracted", "uncontracted"})

# The only decisions a completed German review may carry. Anything else
# — blank, ``uncertain``, a typo — means the review is not finished and
# fails the build. Single source of truth; the CLI imports this.
REVIEW_DECISIONS: frozenset[str] = frozenset({"include", "exclude"})

FULL_NOVEL_CHAPTERS = range(1, 18)  # Ch.1..17
EARLY_CHAPTERS = range(1, 4)  # Ch.1..3
LATE_CHAPTERS = range(4, 18)  # Ch.4..17

# Structural gate constants (STTS xpos + UD deprel of the uncontracted
# determiner token).
REQUIRED_DET_XPOS = "ART"
REQUIRED_DET_DEPREL = "det"

# The extraction TSV writes "-" for "no determiner token" (contracted
# rows). On an uncontracted row a "-" or blank means the structural
# metadata did not arrive — a hard failure, never a silent pass.
_MISSING_VALUES = ("", "-")

# Fields that must agree for two rows sharing one identity to count as
# exact duplicates (anything else is a conflict and fails closed).
_IDENTITY_CORE_FIELDS = (
    "form",
    "canonical_prep",
    "machine_head_lemma",
    "corrected_head_lemma",
    "decision",
    "inventory_eligible",
    "det_xpos",
    "det_deprel",
)


# --------------------------------------------------------------------- errors


class EligiblePoolError(Exception):
    """Base class for eligible-pool failures that must fail closed."""


class OccurrenceIdentityConflictError(EligiblePoolError):
    """Two occurrences share one identity but disagree on core fields.

    Identity is (chapter, source_segment_id, parse_block_id,
    pp_token_start, pp_token_end). Exact duplicates collapse; a row with
    the same coordinates but a different form / lemma / decision /
    structural metadata is a data corruption the caller must resolve
    before any pool is trusted.
    """


class StructuralMetadataMissingError(EligiblePoolError):
    """An uncontracted row lacks the determiner's xpos/deprel metadata.

    The structural gate fails closed: the row is never passed by
    default, and because this indicates an upstream schema/provenance
    break rather than a legitimate data condition, the whole run stops.
    """


class IncompleteReviewError(EligiblePoolError):
    """An occurrence carries a decision that is not include/exclude.

    The eligible pool may only be built from a completed review. A
    blank, ``uncertain``, or misspelled decision fails the whole build
    — it never counts as included, here or anywhere else.
    """


# --------------------------------------------------------------------- types


@dataclass(frozen=True)
class Occurrence:
    """One German PP occurrence as extracted + reviewed.

    ``decision`` is the final German-review decision
    (``include`` / ``exclude``); the CLI guarantees completeness before
    this structure is ever built. ``corrected_head_lemma`` is the
    reviewer's correction — blank means the machine lemma stands.

    ``source_segment_id`` / ``parse_block_id`` / ``pp_token_start`` /
    ``pp_token_end`` carry the identity components explicitly (not
    string-parsed out of ``datapoint_id``). ``det_xpos`` / ``det_deprel``
    carry the structural-gate metadata propagated from the extractor
    (``""`` = absent; ``"-"`` = no det token).
    """

    datapoint_id: str
    chapter: int
    form: str  # "contracted" | "uncontracted"
    canonical_prep: str
    machine_head_lemma: str
    decision: str = ""  # "include" | "exclude"
    corrected_head_lemma: str = ""
    inventory_eligible: bool = False
    source_hash: str = ""
    source_segment_id: str = ""
    parse_block_id: str = ""
    pp_token_start: int = -1
    pp_token_end: int = -1
    det_xpos: str = ""
    det_deprel: str = ""


@dataclass(frozen=True)
class PoolRow:
    """One occurrence with its pool reason (eligible rows carry one of
    the three eligible reasons; the rest record why not)."""

    occurrence: Occurrence
    head_lemma: str  # corrected when non-blank, else machine
    eligible: bool
    pool_reason: str


@dataclass(frozen=True)
class EligiblePoolResult:
    """Output of :func:`build_eligible_pool`."""

    rows: tuple[PoolRow, ...]  # every unique occurrence, in input order
    eligible_ids: frozenset[str]
    summary: dict[str, Any]


# --------------------------------------------------------------------- helpers


def occurrence_identity(o: Occurrence) -> tuple[Any, ...]:
    """Identity: (chapter, source_segment_id, parse_block_id,
    pp_token_start, pp_token_end)."""
    return (
        o.chapter,
        o.source_segment_id,
        o.parse_block_id,
        o.pp_token_start,
        o.pp_token_end,
    )


def collapse_occurrences(occurrences) -> tuple[list[Occurrence], dict[str, int]]:
    """Collapse exact duplicates; fail closed on identity conflicts.

    Rows sharing one :func:`occurrence_identity` are exact duplicates
    iff every :data:`_IDENTITY_CORE_FIELDS` value agrees; later copies
    collapse into the first occurrence (deterministic). The same
    identity with any disagreeing core field raises
    :class:`OccurrenceIdentityConflictError`.
    """
    seen: dict[tuple[Any, ...], Occurrence] = {}
    out: list[Occurrence] = []
    collapsed = 0
    for o in occurrences:
        key = occurrence_identity(o)
        prev = seen.get(key)
        if prev is not None:
            diffs = [f for f in _IDENTITY_CORE_FIELDS if getattr(prev, f) != getattr(o, f)]
            if diffs:
                raise OccurrenceIdentityConflictError(
                    f"occurrence identity {key!r} appears with conflicting "
                    f"core fields: {', '.join(diffs)}; refusing to collapse"
                )
            collapsed += 1
            continue
        seen[key] = o
        out.append(o)
    return out, {
        "duplicate_rows_collapsed": collapsed,
        "unique_occurrences": len(out),
    }


def head_lemma(o: Occurrence) -> str:
    """The lemma used for matching: the reviewer's correction when
    non-blank, else the machine lemma."""
    return o.corrected_head_lemma or o.machine_head_lemma or ""


def canonical_preposition(prep_surface: str) -> str:
    """Map a contracted surface form to its canonical preposition.

    ``im`` → ``in``; ``in`` → ``in``. Unknown contractions pass through.
    """
    return normalize_contracted_prep(prep_surface)


def in_paired_inventory(canonical_prep: str) -> bool:
    """True iff the canonical preposition is in the paper's 13-item
    paired inventory (contracted + uncontracted)."""
    return canonical_prep in PAPER_SHARED_PREPOSITIONS


def structural_gate_status(o: Occurrence) -> str:
    """Structural-gate verdict: ``"pass"`` / ``"excluded"`` / ``"missing"``.

    Uncontracted rows require ``det_xpos == "ART"`` and
    ``det_deprel == "det"``. Missing metadata (blank or the ``"-"``
    no-det placeholder) is ``"missing"`` — the caller must treat it as
    a hard failure. Contracted rows have no determiner and pass.
    """
    if o.form != "uncontracted":
        return "pass"
    if o.det_xpos in _MISSING_VALUES or o.det_deprel in _MISSING_VALUES:
        return "missing"
    if o.det_xpos != REQUIRED_DET_XPOS or o.det_deprel != REQUIRED_DET_DEPREL:
        return "excluded"
    return "pass"


# --------------------------------------------------------------------- core


def build_eligible_pool(occurrences) -> EligiblePoolResult:
    """Apply the formal eligible-pool rule to reviewed occurrences.

    Input: :class:`Occurrence` records whose ``decision`` is the final
    German-review decision. The function itself fails closed on an
    incomplete review: any decision outside
    :data:`REVIEW_DECISIONS` — blank, ``uncertain``, or misspelled —
    raises :class:`IncompleteReviewError` before any row is classified,
    so an unreviewed row can never enter the pool even if a caller
    skipped the upstream validation.

    Returns an :class:`EligiblePoolResult` with one row per unique
    occurrence plus an aggregate summary. Pure and deterministic; no
    I/O.
    """
    occ_list, collapse_stats = collapse_occurrences(occurrences)

    auto_excluded_inventory = 0
    auto_excluded_structural = 0
    human_included = 0
    human_excluded = 0
    not_pair_matched = 0

    # Pass 1: automatic gates, then the review decision. Rows still in
    # contention after pass 1 are contracted Ch.4-17 (pair-matched
    # against the reviewed-include uncontracted set, built below).
    preliminary: list[tuple[Occurrence, str, str, bool]] = []
    #          (occ, head_lemma, reason, eligible)
    for o in occ_list:
        lemma = head_lemma(o)

        # Preposition inventory.
        if not o.inventory_eligible:
            auto_excluded_inventory += 1
            preliminary.append((o, lemma, "outside_paired_inventory", False))
            continue

        # Structural gate (uncontracted only; missing metadata is fatal).
        gate = structural_gate_status(o)
        if gate == "missing":
            raise StructuralMetadataMissingError(
                f"uncontracted occurrence {o.datapoint_id!r} lacks determiner "
                "xpos/deprel metadata; the structural gate fails closed"
            )
        if gate == "excluded":
            auto_excluded_structural += 1
            preliminary.append((o, lemma, "failed_structural_gate", False))
            continue

        # German review decision — fail closed on anything that is not a
        # final include/exclude (blank, uncertain, or a typo means the
        # review is not finished; it must never count as included).
        if o.decision not in REVIEW_DECISIONS:
            raise IncompleteReviewError(
                f"review not complete: decision {o.decision!r} for occurrence {o.datapoint_id!r}"
            )
        if o.decision == "exclude":
            human_excluded += 1
            preliminary.append((o, lemma, "excluded_by_german_review", False))
            continue
        human_included += 1

        if o.form == "uncontracted":
            preliminary.append((o, lemma, "uncontracted_all_chapters", True))
            continue
        if o.chapter in EARLY_CHAPTERS:
            preliminary.append((o, lemma, "contracted_ch1_3", True))
            continue
        # Contracted Ch.4-17: deferred to pass 2.
        preliminary.append((o, lemma, "", False))

    # The reviewed-include uncontracted (preposition, lemma) pair set.
    reviewed_uncontracted_pairs: set[tuple[str, str]] = set()
    for o, lemma, reason, _ in preliminary:
        if reason == "uncontracted_all_chapters":
            reviewed_uncontracted_pairs.add((o.canonical_prep, lemma))

    # Pass 2: pair-match the contracted Ch.4-17 rows.
    rows: list[PoolRow] = []
    for o, lemma, reason, eligible in preliminary:
        if reason == "":
            if (o.canonical_prep, lemma) in reviewed_uncontracted_pairs:
                reason, eligible = "contracted_ch4_17_pair_matched", True
            else:
                reason, eligible = "contracted_ch4_17_no_uncontracted_counterpart", False
                not_pair_matched += 1
        rows.append(PoolRow(occurrence=o, head_lemma=lemma, eligible=eligible, pool_reason=reason))

    eligible_ids = frozenset(r.occurrence.datapoint_id for r in rows if r.eligible)

    by_reason: dict[str, int] = defaultdict(int)
    eligible_by_reason: dict[str, int] = defaultdict(int)
    for r in rows:
        by_reason[r.pool_reason] += 1
        if r.eligible:
            eligible_by_reason[r.pool_reason] += 1

    summary: dict[str, Any] = {
        "extracted_total": (
            collapse_stats["unique_occurrences"] + collapse_stats["duplicate_rows_collapsed"]
        ),
        "duplicate_rows_collapsed": collapse_stats["duplicate_rows_collapsed"],
        "automatically_excluded": {
            "outside_paired_inventory": auto_excluded_inventory,
            "failed_structural_gate": auto_excluded_structural,
        },
        "human_review": {
            "included": human_included,
            "excluded": human_excluded,
        },
        "eligible_pool": {
            "uncontracted_all_chapters": eligible_by_reason.get("uncontracted_all_chapters", 0),
            "contracted_ch1_3": eligible_by_reason.get("contracted_ch1_3", 0),
            "contracted_ch4_17_pair_matched": eligible_by_reason.get(
                "contracted_ch4_17_pair_matched", 0
            ),
            "eligible_total": len(eligible_ids),
        },
        "contracted_ch4_17_no_uncontracted_counterpart": not_pair_matched,
        "by_reason": dict(by_reason),
    }

    return EligiblePoolResult(
        rows=tuple(rows),
        eligible_ids=eligible_ids,
        summary=summary,
    )


__all__ = [
    "Occurrence",
    "PoolRow",
    "EligiblePoolResult",
    "POOL_REASONS",
    "REVIEW_DECISIONS",
    "FULL_NOVEL_CHAPTERS",
    "EARLY_CHAPTERS",
    "LATE_CHAPTERS",
    "REQUIRED_DET_XPOS",
    "REQUIRED_DET_DEPREL",
    "EligiblePoolError",
    "OccurrenceIdentityConflictError",
    "StructuralMetadataMissingError",
    "IncompleteReviewError",
    "build_eligible_pool",
    "collapse_occurrences",
    "occurrence_identity",
    "head_lemma",
    "canonical_preposition",
    "in_paired_inventory",
    "structural_gate_status",
    # Re-exported for caller convenience; the canonical contract is in step4.
    "CONTRACTED_PREP_NORMALIZATION",
    "PAPER_SHARED_PREPOSITIONS",
]
