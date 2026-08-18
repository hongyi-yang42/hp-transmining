"""Tests for the bilingual lexical-anchor prior (hp_corpus.lexicon)."""

from __future__ import annotations

from hp_corpus.lexicon import (
    DE_ZH_LEXICON,
    anchor_weights,
    extract_anchors,
    lexical_bonus_matrix,
)


def test_lexicon_zh_values_unique() -> None:
    """Each ZH surface form must map back to exactly one anchor ID, or the
    reverse index used implicitly by extraction would be ambiguous."""
    values = list(DE_ZH_LEXICON.values())
    assert len(values) == len(set(values))
    assert len(DE_ZH_LEXICON) >= 50


def test_extract_anchors_matches_both_sides() -> None:
    de = extract_anchors("Hermine sagte Firenze nichts über 3/4.")
    zh = extract_anchors("赫敏没有对费伦泽说起过3/4。")
    assert "Hermine" in de and "Firenze" in de
    assert "Hermine" in zh and "Firenze" in zh
    assert de & zh >= {"Hermine", "Firenze", "3/4"}


def test_extract_anchors_ignores_lone_digits_and_short_numbers() -> None:
    anchors = extract_anchors("Zauberstab 9 und 4 und 42")
    assert "9" not in anchors
    assert "4" not in anchors
    assert "42" in anchors


def test_anchor_weights_rare_beats_frequent() -> None:
    """A name appearing in most sentences weighs little; a one-off name
    weighs nearly 1."""
    sets = [{"Harry"} for _ in range(20)] + [{"Firenze"}, set()]
    weights = anchor_weights(sets)
    assert weights["Firenze"] > weights["Harry"]
    assert weights["Firenze"] > 0.9
    assert weights["Harry"] < 0.2


def test_bonus_matrix_capped_and_zero_without_shared_anchors() -> None:
    src = [{"Firenze", "Bane"}]
    tgt = [{"Firenze"}, {"哈利"}]
    weights = {"Firenze": 1.0, "Bane": 0.5, "哈利": 1.0}
    bonus = lexical_bonus_matrix(src, tgt, weights, weight=10.0, cap=0.15)
    assert bonus[0, 0] == 0.15  # 10 * (1.0 + 0.5) capped
    assert bonus[0, 1] == 0.0  # no shared anchor
