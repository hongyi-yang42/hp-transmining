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
