"""Tests for the retrieval-view candidate window (hp_corpus.candidates)."""

from __future__ import annotations

import pytest

from hp_corpus.candidates import WindowCandidate, candidate_window, window_ids


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
