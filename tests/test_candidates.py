"""Tests for the retrieval-view candidate window (hp_corpus.candidates)."""

from __future__ import annotations

import pytest

from hp_corpus.candidates import (
    WindowCandidate,
    budgeted_window,
    budgeted_window_ids,
    candidate_window,
    neighbor_bracket,
    window_ids,
)


def _order(n: int = 10, prefix: str = "hp1_zh_ch01_p0001_s") -> list[str]:
    return [f"{prefix}{i:03d}" for i in range(1, n + 1)]


def test_window_middle_expands_both_sides() -> None:
    order = _order()
    win = candidate_window([order[4]], order, w=1)
    assert [c.segment_id for c in win] == [order[3], order[4], order[5]]
    assert [c.in_anchor for c in win] == [False, True, False]


def test_window_group_anchor_flags_members() -> None:
    order = _order()
    win = candidate_window([order[2], order[3]], order, w=1)
    assert [c.segment_id for c in win] == order[1:5]
    assert [c.in_anchor for c in win] == [False, True, True, False]


def test_window_clamps_at_chapter_edges() -> None:
    order = _order()
    first = candidate_window([order[0]], order, w=2)
    assert [c.segment_id for c in first] == order[:3]
    last = candidate_window([order[-1]], order, w=2)
    assert [c.segment_id for c in last] == order[-3:]


def test_window_w_zero_is_anchor_only() -> None:
    order = _order()
    win = candidate_window([order[3], order[5]], order, w=0)
    assert [c.segment_id for c in win] == [order[3], order[4], order[5]]
    assert [c.in_anchor for c in win] == [True, False, True]


def test_window_empty_anchor_is_empty() -> None:
    assert candidate_window([], _order(), w=1) == []
    assert window_ids([], _order()) == []


def test_window_unknown_anchor_raises() -> None:
    with pytest.raises(ValueError, match="not found in chapter order"):
        candidate_window(["bogus_id"], _order(), w=1)


def test_window_candidate_is_frozen_dataclass() -> None:
    c = WindowCandidate(segment_id="x", in_anchor=True)
    with pytest.raises((AttributeError, TypeError)):
        c.in_anchor = False  # type: ignore[misc]


# --------------------------------------------------------------- budgeted_window


def test_budgeted_single_anchor_gets_five_segment_view() -> None:
    order = _order()
    win = budgeted_window([order[4]], order, max_w=2, budget=5)
    assert [c.segment_id for c in win] == order[2:7]
    assert [c.in_anchor for c in win] == [False, False, True, False, False]


def test_budgeted_two_segment_anchor_splits_remaining_right() -> None:
    # Anchor span 2 leaves budget for 3 neighbours: one left, two right
    # (the odd slot goes to the right, in reading order).
    order = _order()
    win = budgeted_window([order[4], order[5]], order, max_w=2, budget=5)
    assert [c.segment_id for c in win] == order[3:8]
    assert [c.in_anchor for c in win] == [False, True, True, False, False]


def test_budgeted_four_segment_anchor_adds_one_right() -> None:
    order = _order()
    assert budgeted_window_ids([order[3], order[4], order[5], order[6]], order) == order[3:8]


def test_budgeted_anchor_filling_budget_gets_no_added_context() -> None:
    # The budget limits added neighbours, never the anchor group itself.
    order = _order()
    assert budgeted_window_ids([order[2], order[3], order[4], order[5], order[6]], order) \
        == order[2:7]
    assert budgeted_window_ids(order[0:8], order) == order[0:8]


def test_budgeted_noncontiguous_anchor_spans_min_to_max() -> None:
    # Like candidate_window, the slice runs min..max of the anchor
    # positions; in-between sentences belong to the window but not the
    # anchor.
    order = _order()
    win = budgeted_window([order[2], order[4]], order)
    assert [c.segment_id for c in win] == order[1:6]
    assert [c.in_anchor for c in win] == [False, True, False, True, False]


def test_budgeted_clamps_at_chapter_edges_without_compensation() -> None:
    # Edge clamping is never compensated on the other side.
    order = _order()
    assert budgeted_window_ids([order[0]], order) == order[0:3]
    assert budgeted_window_ids([order[-1]], order) == order[-3:]


def test_budgeted_empty_and_unknown_anchors() -> None:
    assert budgeted_window([], _order()) == []
    with pytest.raises(ValueError, match="not found in chapter order"):
        budgeted_window(["bogus_id"], _order())


# --------------------------------------------------------------- neighbor_bracket


def test_bracket_between_single_anchors() -> None:
    order = _order()
    assert neighbor_bracket([order[2]], [order[5]], order) == order[2:6]


def test_bracket_uses_corpus_order_edges_of_groups() -> None:
    # An N:M anchor group: the bracket runs from the group's LAST id (in
    # corpus order) to the next group's FIRST id, even if the group's ids
    # are listed out of order.
    order = _order()
    assert neighbor_bracket([order[4], order[2]], [order[7], order[6]], order) == order[4:7]


def test_bracket_adjacent_anchors_is_the_two_sentences() -> None:
    order = _order()
    assert neighbor_bracket([order[5]], [order[6]], order) == order[5:7]


def test_bracket_clamps_when_one_side_missing() -> None:
    order = _order()
    assert neighbor_bracket([], [order[3]], order) == order[:4]
    assert neighbor_bracket([order[7]], [], order) == order[7:]


def test_bracket_both_sides_missing_is_empty() -> None:
    assert neighbor_bracket([], [], _order()) == []


def test_bracket_over_cap_is_empty() -> None:
    order = _order()
    assert neighbor_bracket([order[0]], [order[9]], order, cap=6) == []
    assert neighbor_bracket([order[0]], [order[5]], order, cap=6) == order[0:6]


def test_bracket_unknown_anchor_raises() -> None:
    with pytest.raises(ValueError, match="not found in chapter order"):
        neighbor_bracket(["bogus_id"], [], _order())
