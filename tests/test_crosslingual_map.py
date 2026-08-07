"""Synthetic-fixture tests for hp_corpus.crosslingual_map.

Every fixture here uses invented, non-novel text. Tests cover the
methodology boundaries: PP extraction, scoring signals, threshold
classification, and unmappable-reason reporting.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from hp_corpus.crosslingual_map import (
    CANDIDATE_THRESHOLD,
    DE_EN_PREP_MAP,
    DE_ZH_PREP_MAP,
    MATCH_THRESHOLD,
    PPElement,
    Sentence,
    Token,
    extract_pps,
    parse_conllu,
    propose_for_sides,
    score_candidate,
    select_best,
)

# --------------------------------------------------------------------- helpers


def _token(
    token_id: int,
    form: str,
    lemma: str = "",
    upos: str = "NOUN",
    head: int = 0,
    deprel: str = "root",
) -> Token:
    return Token(
        token_id=token_id,
        form=form,
        lemma=lemma or form.lower(),
        upos=upos,
        head=head,
        deprel=deprel,
    )


def _sentence(sent_id: str, text: str, tokens: Iterable[Token]) -> Sentence:
    return Sentence(sent_id=sent_id, text=text, tokens=tuple(tokens))


def _pp(
    sent_id: str = "syn_001",
    prep_surface: str = "in",
    prep_lemma: str = "in",
    prep_token_id: int = 1,
    head_surface: str = "house",
    head_lemma: str = "house",
    head_token_id: int = 2,
    head_upos: str = "NOUN",
    span_text: str = "in house",
    span_token_ids: tuple[int, ...] = (1, 2),
    char_start: int = 0,
    char_end: int = 8,
) -> PPElement:
    return PPElement(
        sent_id=sent_id,
        prep_surface=prep_surface,
        prep_lemma=prep_lemma,
        prep_token_id=prep_token_id,
        head_surface=head_surface,
        head_lemma=head_lemma,
        head_token_id=head_token_id,
        head_upos=head_upos,
        span_text=span_text,
        span_token_ids=span_token_ids,
        char_start=char_start,
        char_end=char_end,
    )


# --------------------------------------------------------------------- parsing


def test_parse_conllu_skips_mwt_range_lines(tmp_path: Path) -> None:
    """MWT range lines like ``5-6`` must not appear in the token tuple."""
    path = tmp_path / "syn.conllu"
    path.write_text(
        "# sent_id = syn_001\n"
        "# text = im Haus\n"
        "1-2	im	_	_	_	_	_	_	_	_\n"
        "1	in	in	ADP	APPR	_	2	case	_	_\n"
        "2	Haus	Haus	NOUN	NN	_	0	root	_	_\n"
        "3	.	.	PUNCT	.	_	2	punct	_	_\n"
        "\n",
        encoding="utf-8",
    )
    sents = parse_conllu(path)
    assert set(sents) == {"syn_001"}
    s = sents["syn_001"]
    assert s.sent_id == "syn_001"
    assert s.text == "im Haus"
    # Range line skipped — only single-token rows kept.
    assert tuple(t.token_id for t in s.tokens) == (1, 2, 3)


def test_parse_conllu_handles_blank_input(tmp_path: Path) -> None:
    path = tmp_path / "empty.conllu"
    path.write_text("", encoding="utf-8")
    assert parse_conllu(path) == {}


# --------------------------------------------------------------------- extract_pps


def test_extract_pps_finds_simple_adp_case() -> None:
    """One ADP with deprel=case + its head noun → one PP candidate."""
    s = _sentence(
        "syn_001",
        "in the house",
        [
            _token(1, "in", upos="ADP", head=3, deprel="case"),
            _token(2, "the", upos="DET", head=3, deprel="det"),
            _token(3, "house", upos="NOUN", head=0, deprel="root"),
        ],
    )
    pps = extract_pps(s)
    assert len(pps) == 1
    pp = pps[0]
    assert pp.prep_surface == "in"
    assert pp.prep_token_id == 1
    assert pp.head_surface == "house"
    assert pp.head_token_id == 3
    assert set(pp.span_token_ids) == {1, 2, 3}


def test_extract_pps_ignores_adp_without_case_deprel() -> None:
    """An ADP that is itself the root (e.g. a stranded prep) is not a PP."""
    s = _sentence(
        "syn_002",
        "up",
        [_token(1, "up", upos="ADP", head=0, deprel="root")],
    )
    assert extract_pps(s) == []


def test_extract_pps_returns_empty_for_no_adp() -> None:
    s = _sentence(
        "syn_003",
        "house",
        [_token(1, "house", upos="NOUN", head=0, deprel="root")],
    )
    assert extract_pps(s) == []


def test_extract_pps_strips_trailing_punct() -> None:
    s = _sentence(
        "syn_004",
        "of number four,",
        [
            _token(1, "of", upos="ADP", head=2, deprel="case"),
            _token(2, "number", upos="NOUN", head=0, deprel="root"),
            _token(3, "four", upos="NUM", head=2, deprel="flat"),
            _token(4, ",", upos="PUNCT", head=2, deprel="punct"),
        ],
    )
    pps = extract_pps(s)
    assert len(pps) == 1
    # Trailing punct (id=4) stripped from span.
    assert 4 not in pps[0].span_token_ids


def test_char_span_disambiguates_duplicate_forms() -> None:
    """When the same form appears twice (e.g. two ``of``s), char positions
    must resolve to the SECOND occurrence for the second PP — not the
    first. Regression for a bug where ``re.search`` matched the first
    word-boundary occurrence regardless of token order.
    """
    s = _sentence(
        "syn_dup",
        "edge of the sofa of the room",
        [
            _token(1, "edge", upos="NOUN", head=0, deprel="root"),
            _token(2, "of", upos="ADP", head=4, deprel="case"),
            _token(3, "the", upos="DET", head=4, deprel="det"),
            _token(4, "sofa", upos="NOUN", head=1, deprel="nmod"),
            _token(5, "of", upos="ADP", head=7, deprel="case"),
            _token(6, "the", upos="DET", head=7, deprel="det"),
            _token(7, "room", upos="NOUN", head=1, deprel="nmod"),
        ],
    )
    pps = extract_pps(s)
    # Two PPs: "of the sofa" (tokens 2-4) and "of the room" (tokens 5-7).
    assert len(pps) == 2
    second = next(p for p in pps if p.head_surface == "room")
    # Second PP's char range must start at the second "of", which is at
    # offset 17 in "edge of the sofa of the room" (e-d-g-e-_-o-f-_-t-h-e
    # -_-s-o-f-a-_-o-f → second 'o' is at index 17).
    assert second.char_start == 17
    assert second.char_end == 28  # end of "room"


def test_char_span_returns_minus_one_when_form_unfindable() -> None:
    """If the CoNLL-U form doesn't appear in the text (rare Stanza
    quirk), char_range must be (-1, -1) — not a wrong position."""
    s = _sentence(
        "syn_unfindable",
        "real text here",
        [
            _token(1, "nonexistent", upos="NOUN", head=0, deprel="root"),
        ],
    )
    # No PPs to extract, but verify the helper directly.
    from hp_corpus.crosslingual_map import _char_span_for_tokens

    start, end = _char_span_for_tokens(s, [1])
    assert (start, end) == (-1, -1)


# --------------------------------------------------------------------- scoring


def test_score_candidate_prep_semantic_match_only() -> None:
    """DE 'im' (canonical 'in') + EN 'in' → prep_semantic_match fires.

    Note: 'Haus'/'house' do NOT substring-overlap after lowercasing
    ('haus' vs 'house'), so lemma_overlap correctly does not fire.
    Use Hund/hound below for the cognate case.
    """
    cand = _pp(prep_surface="in", head_lemma="house", head_upos="NOUN")
    scored = score_candidate(
        de_prep_normalized="in",
        de_pp_surface="im Haus",
        de_head_lemma="Haus",
        cand_pp=cand,
        cand_prep_surface="in",
        prep_map=DE_EN_PREP_MAP,
        position_de=1,
        n_de_pps=1,
        position_cand=1,
        n_cand_pps=1,
        align_confidence=0.5,  # below EN threshold
        side="en",
    )
    signals = {s for s, _ in scored.components}
    assert "prep_semantic_match" in signals
    assert "lemma_overlap" not in signals  # Haus / house don't substring-overlap
    assert "position_single_side" in signals  # n_de=1 → automatic
    assert "align_confidence_bonus" not in signals  # conf below threshold


def test_score_candidate_lemma_overlap_for_cognates() -> None:
    """Identical lowercased lemma ('Hand'/'hand') substring-overlaps →
    lemma_overlap fires. German↔English substring cognates are rare;
    identical strings are the cleanest test case."""
    cand = _pp(prep_surface="of", head_lemma="hand", head_upos="NOUN")
    scored = score_candidate(
        de_prep_normalized="von",
        de_pp_surface="von der Hand",
        de_head_lemma="Hand",
        cand_pp=cand,
        cand_prep_surface="of",
        prep_map=DE_EN_PREP_MAP,
        position_de=1,
        n_de_pps=2,
        position_cand=1,
        n_cand_pps=2,
        align_confidence=0.5,
        side="en",
    )
    signals = {s for s, _ in scored.components}
    assert "lemma_overlap" in signals


def test_score_candidate_position_match_with_multiple_pps() -> None:
    """DE has 2 PPs, candidate is the 2nd of 2 → position_match fires
    only if positions are equal; otherwise neither single-side nor
    match fires."""
    cand = _pp(prep_surface="on", head_lemma="table")
    # n_de=2, n_cand=2; positions 1 vs 2 → no position signal
    scored = score_candidate(
        de_prep_normalized="auf",
        de_pp_surface="auf dem Tisch",
        de_head_lemma="Tisch",
        cand_pp=cand,
        cand_prep_surface="on",
        prep_map=DE_EN_PREP_MAP,
        position_de=1,
        n_de_pps=2,
        position_cand=2,
        n_cand_pps=2,
        align_confidence=0.5,
        side="en",
    )
    signals = {s for s, _ in scored.components}
    assert "position_single_side" not in signals
    assert "position_match" not in signals


def test_score_candidate_zh_threshold_lower_than_en() -> None:
    """Same alignment confidence (0.85) gives bonus for ZH but not EN."""
    cand = _pp(prep_surface="在", head_lemma="house")
    en_scored = score_candidate(
        de_prep_normalized="in",
        de_pp_surface="im Haus",
        de_head_lemma="Haus",
        cand_pp=cand,
        cand_prep_surface="在",
        prep_map=DE_ZH_PREP_MAP,
        position_de=1,
        n_de_pps=1,
        position_cand=1,
        n_cand_pps=1,
        align_confidence=0.85,
        side="en",
    )
    zh_scored = score_candidate(
        de_prep_normalized="in",
        de_pp_surface="im Haus",
        de_head_lemma="Haus",
        cand_pp=cand,
        cand_prep_surface="在",
        prep_map=DE_ZH_PREP_MAP,
        position_de=1,
        n_de_pps=1,
        position_cand=1,
        n_cand_pps=1,
        align_confidence=0.85,
        side="zh",
    )
    en_signals = {s for s, _ in en_scored.components}
    zh_signals = {s for s, _ in zh_scored.components}
    assert "align_confidence_bonus" not in en_signals
    assert "align_confidence_bonus" in zh_signals


def test_score_candidate_proper_name_overlap_case_insensitive() -> None:
    """DE 'Alice' ↔ EN candidate span containing 'Alice' → proper-name
    overlap fires (case-insensitive)."""
    cand = _pp(
        prep_surface="to",
        head_surface="Alice",
        head_lemma="Alice",
        head_upos="PROPN",
        span_text="to Alice",
    )
    scored = score_candidate(
        de_prep_normalized="zu",
        de_pp_surface="zu Alice",
        de_head_lemma="Alice",
        cand_pp=cand,
        cand_prep_surface="to",
        prep_map=DE_EN_PREP_MAP,
        position_de=1,
        n_de_pps=1,
        position_cand=1,
        n_cand_pps=1,
        align_confidence=0.95,
        side="en",
    )
    signals = {s for s, _ in scored.components}
    assert "proper_name_overlap" in signals


# --------------------------------------------------------------------- select_best


def test_select_best_returns_unmappable_when_no_candidates() -> None:
    result = select_best(
        de_prep_normalized="in",
        de_pp_surface="im Haus",
        de_head_lemma="Haus",
        candidates=[],
        prep_map=DE_EN_PREP_MAP,
        align_confidence=0.95,
        side="en",
    )
    assert result.status == "unmappable"
    assert result.reason == "no_pp_in_target_sentence"
    assert result.best is None
    assert result.n_candidates_considered == 0


def test_select_best_matched_when_score_exceeds_threshold() -> None:
    cand = _pp(
        prep_surface="in",
        head_surface="house",
        head_lemma="house",
        head_upos="NOUN",
    )
    result = select_best(
        de_prep_normalized="in",
        de_pp_surface="im Haus",
        de_head_lemma="Haus",
        candidates=[cand],
        prep_map=DE_EN_PREP_MAP,
        align_confidence=0.5,
        side="en",
    )
    # prep_semantic(3) + position_single_side(2) = 5
    assert result.status == "matched"
    assert result.best is cand
    assert result.best_score == pytest.approx(5.0)
    assert result.n_candidates_considered == 1


def test_select_best_candidate_when_below_match_threshold() -> None:
    """A position-only signal (no prep/lemma/proper match) scores 2 →
    'candidate' status. We force 'no prep match' by using a candidate
    prep that does not appear in any DE_EN_PREP_MAP bucket."""
    cand = _pp(
        prep_surface="aboard",  # not in any DE_EN_PREP_MAP value set
        head_surface="thing",
        head_lemma="thing",
    )
    result = select_best(
        de_prep_normalized="in",
        de_pp_surface="im Ding",
        de_head_lemma="Ding",
        candidates=[cand],
        prep_map=DE_EN_PREP_MAP,
        align_confidence=0.5,
        side="en",
    )
    assert result.status == "candidate"
    assert result.best_score == pytest.approx(2.0)


def test_select_best_unmappable_when_best_below_candidate_threshold() -> None:
    """n_de>1, n_cand>1, no prep match, no overlap, low conf → score 0
    → from score_candidate's perspective, no signal fires.

    select_best forces n_cand=1 internally (single-PP side), which
    always triggers position_single_side and floors the score at 2.
    Test the truly-zero-score path via score_candidate directly."""
    cand = _pp(
        prep_surface="aboard",
        head_surface="thing",
        head_lemma="thing",
    )
    scored = score_candidate(
        de_prep_normalized="in",
        de_pp_surface="im Ding",
        de_head_lemma="Ding",
        cand_pp=cand,
        cand_prep_surface="aboard",
        prep_map=DE_EN_PREP_MAP,
        position_de=1,
        n_de_pps=3,
        position_cand=2,
        n_cand_pps=3,
        align_confidence=0.5,
        side="en",
    )
    assert scored.score == 0.0
    assert scored.components == ()


# --------------------------------------------------------------------- propose_for_sides


def test_propose_for_sides_returns_en_and_zh_results() -> None:
    en = _sentence(
        "en_syn_001",
        "in the house",
        [
            _token(1, "in", upos="ADP", head=3, deprel="case"),
            _token(2, "the", upos="DET", head=3, deprel="det"),
            _token(3, "house", upos="NOUN", head=0, deprel="root"),
        ],
    )
    zh = _sentence(
        "zh_syn_001",
        "在房子里",
        [
            _token(1, "在", upos="ADP", head=2, deprel="case"),
            _token(2, "房子", upos="NOUN", head=0, deprel="root"),
            _token(3, "里", upos="ADP", head=2, deprel="case"),
        ],
    )
    en_r, zh_r = propose_for_sides(
        de_prep_normalized="in",
        de_pp_surface="im Haus",
        de_head_lemma="Haus",
        en_sentences=[en],
        zh_sentences=[zh],
        en_align_confidence=0.95,
        zh_align_confidence=0.85,
    )
    assert en_r.status == "matched"
    assert zh_r.status == "matched"


def test_propose_for_sides_handles_empty_target_sentences() -> None:
    """When both EN and ZH sentences are missing (broken alignment),
    both sides return unmappable with no_pp_in_target_sentence."""
    en_r, zh_r = propose_for_sides(
        de_prep_normalized="in",
        de_pp_surface="im Haus",
        de_head_lemma="Haus",
        en_sentences=[],
        zh_sentences=[],
        en_align_confidence=0.95,
        zh_align_confidence=0.85,
    )
    assert en_r.status == "unmappable"
    assert zh_r.status == "unmappable"
    assert en_r.reason == "no_pp_in_target_sentence"
    assert zh_r.reason == "no_pp_in_target_sentence"


# --------------------------------------------------------------------- thresholds


def test_thresholds_are_sane() -> None:
    """Match threshold strictly greater than candidate threshold, and
    both are positive."""
    assert MATCH_THRESHOLD > CANDIDATE_THRESHOLD > 0


def test_de_zh_prep_map_covers_core_prepositions() -> None:
    """The ZH prep map must cover every canonical German preposition
    the project's contraction table produces."""
    canonical = set(DE_EN_PREP_MAP.keys())
    assert set(DE_ZH_PREP_MAP.keys()) >= canonical
