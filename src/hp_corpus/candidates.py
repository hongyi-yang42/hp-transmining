"""Retrieval-view candidate windows over the alignment partition.

The DP alignment is a strict partition — every sentence is consumed exactly
once — which is the right shape for the anchor but the wrong shape for the
downstream lookup: a DE PP sentence needs "the Chinese sentences that
contain my translation", and the LLM-judge audit showed the true partner
sits within ±1 sentence of the aligned group for 92% of DE sentences
(±2: 97%) while the strict group itself only covers 63%.

The window is a *view*: alignment records keep their partition semantics
(rank-1 anchor); consumers get the anchor group expanded by ``w`` sentences
on each side in corpus order, with each member flagged ``in_anchor`` so
anchor and context stay distinguishable.
"""

from __future__ import annotations

from dataclasses import dataclass

# How a side's retrieval context was obtained. ``anchor_window`` is the
# default anchor ± 1 view; ``merge_widened`` marks N:M-merged anchors whose
# window was widened; ``heuristic_widened`` marks sides widened on a
# suspicion signal (under-segmented German segment or low-confidence
# anchor); ``neighbor_fallback`` marks anchorless sides bracketed between
# neighbour anchors; ``manual_review`` marks sides the machine could not
# retrieve reliably.
CONTEXT_PROVENANCES = frozenset(
    {
        "anchor_window",
        "merge_widened",
        "heuristic_widened",
        "neighbor_fallback",
        "manual_review",
    }
)


@dataclass(frozen=True)
class WindowCandidate:
    """One sentence in a candidate window, in corpus order."""

    segment_id: str
    in_anchor: bool


def candidate_window(
    anchor_ids: list[str],
    chapter_order: list[str],
    w: int = 1,
) -> list[WindowCandidate]:
    """Expand an alignment anchor group by ``w`` sentences on each side.

    ``anchor_ids`` are the alignment record's sentence ids for one side
    (possibly several — N:M groups); ``chapter_order`` is that chapter's
    full ordered segment-id list for the same language. Returns the
    contiguous corpus-order slice from ``min(anchor) - w`` to
    ``max(anchor) + w`` (clamped at chapter edges), each entry flagged
    whether it belongs to the anchor group.

    An empty anchor (unaligned side) yields an empty window — there is no
    anchor to expand. An anchor id missing from ``chapter_order`` is a data
    integrity failure and raises ``ValueError``.
    """
    if not anchor_ids:
        return []
    index = {sid: i for i, sid in enumerate(chapter_order)}
    missing = [sid for sid in anchor_ids if sid not in index]
    if missing:
        raise ValueError(f"anchor segment ids not found in chapter order: {missing[:3]}")
    positions = [index[sid] for sid in anchor_ids]
    lo = max(0, min(positions) - w)
    hi = min(len(chapter_order), max(positions) + w + 1)
    anchor_set = set(anchor_ids)
    return [
        WindowCandidate(segment_id=sid, in_anchor=sid in anchor_set) for sid in chapter_order[lo:hi]
    ]


def window_ids(
    anchor_ids: list[str],
    chapter_order: list[str],
    w: int = 1,
) -> list[str]:
    """Convenience: just the ordered ids of :func:`candidate_window`."""
    return [c.segment_id for c in candidate_window(anchor_ids, chapter_order, w)]


def neighbor_bracket(
    prev_anchor_ids: list[str],
    next_anchor_ids: list[str],
    chapter_order: list[str],
    cap: int = 6,
) -> list[str]:
    """Bracket the target region for an anchorless sentence.

    When a DE sentence has no aligned target (a DP 1:0 gap, typically
    because the translation merged it into a neighbouring sentence), the
    translation lives between the target anchors of the nearest DE
    neighbours that *do* align. This returns the contiguous corpus-order
    slice from the last id of the previous neighbour's anchor group to
    the first id of the next neighbour's anchor group, inclusive.

    One side may be empty (no aligned neighbour in that direction —
    clamp to the chapter edge). Both empty, an anchor id missing from
    ``chapter_order``, or a bracket longer than ``cap`` sentences all
    yield ``[]`` — the caller marks the side for manual review rather
    than trusting a wide fallback.
    """
    if not prev_anchor_ids and not next_anchor_ids:
        return []
    index = {sid: i for i, sid in enumerate(chapter_order)}
    missing = [sid for sid in (*prev_anchor_ids, *next_anchor_ids) if sid not in index]
    if missing:
        raise ValueError(f"anchor segment ids not found in chapter order: {missing[:3]}")
    lo = index[max(prev_anchor_ids, key=index.get)] if prev_anchor_ids else 0
    hi = index[min(next_anchor_ids, key=index.get)] + 1 if next_anchor_ids else len(chapter_order)
    if hi - lo > cap:
        return []
    return chapter_order[lo:hi]
