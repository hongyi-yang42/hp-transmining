"""Synthetic-fixture tests for the hybrid arbitration layer.

All fixture text is invented; no novel text. Covers span → parse-token
mapping, deterministic form recomputation (verbal / numeral+classifier
/ possessive / quantifier guards), the omission guard, the
semantic-drift guard, the routing policy branches, and that the saved
GLM / local cells pass through unchanged.
"""

from __future__ import annotations

from hp_corpus.deterministic_tuples import SideLayout, Token
from hp_corpus.hybrid_arbitration import (
    ELIGIBILITY_COLUMNS,
    HYBRID_COLUMNS,
    CandidateRef,
    EligibilityDecision,
    GlmSide,
    LocalSide,
    SideFacts,
    SpanMapping,
    arbitrate_side,
    build_eligibility_row,
    build_hybrid_row,
    eligibility_side,
    map_span_to_tokens,
    realization_type,
    recompute_form,
    recover_containing_constituent,
    row_eligibility_for,
)


def tok(form: str, upos: str, deprel: str = "", head: str = "") -> Token:
    return Token(form=form, upos=upos, feats="", head=head, deprel=deprel)


def layout(block_text: str, tokens: list[Token]) -> SideLayout:
    return SideLayout(
        block_ids=["syn#b001"],
        block_texts=[block_text],
        token_block=[0] * len(tokens),
        tokens=tokens,
    )


def facts(**overrides) -> SideFacts:
    base = dict(
        locus="reliable",
        span_in_context=True,
        mapping=None,
        in_locus_blocks=True,
        eflomal_covers_span=False,
        contextual_top1_covers=False,
    )
    base.update(overrides)
    return SideFacts(**base)


def glm(span: str = "", status: str = "aligned", paper: str = "", detail: str = "") -> GlmSide:
    return GlmSide(status=status, span=span, paper_form=paper, detail_form=detail)


def local(state: str = "", chosen: str = "", evidence: str = "none") -> LocalSide:
    return LocalSide(state=state, evidence=evidence, chosen_span=chosen, form=("", ""), category="")


# --- span → parse tokens -----------------------------------------------------------


def test_map_span_to_tokens_exact():
    lay = layout(
        "the claw sat near the rusty lantern",
        [
            tok("the", "DET"),
            tok("claw", "NOUN"),
            tok("sat", "VERB"),
            tok("near", "ADP"),
            tok("the", "DET"),
            tok("rusty", "ADJ"),
            tok("lantern", "NOUN"),
        ],
    )
    m = map_span_to_tokens(lay, "the rusty lantern")
    assert m is not None
    assert [t.form for t in m.tokens] == ["the", "rusty", "lantern"]


def test_map_span_to_tokens_subword_cut_keeps_full_token():
    lay = layout("窗台上有一只杯子", [tok("窗台上", "NOUN"), tok("有", "VERB")])
    m = map_span_to_tokens(lay, "窗台")
    assert m is not None
    assert [t.form for t in m.tokens] == ["窗台上"]


def test_map_span_to_tokens_absent_returns_none():
    lay = layout("a b c", [tok("a", "NOUN"), tok("b", "NOUN"), tok("c", "NOUN")])
    assert map_span_to_tokens(lay, "zzz") is None
    assert map_span_to_tokens(lay, "") is None


def test_map_span_to_tokens_text_hit_without_tokens_tries_next_block():
    # parse artifact: both blocks repeat the same # text, but each
    # tokenizes only part of it — the span's tokens live in block 2
    text = "she wandered off and finally the compass gleamed"
    lay = SideLayout(
        block_ids=["syn#b001", "syn#b002"],
        block_texts=[text, text],
        token_block=[0] * 4 + [1] * 4,
        tokens=[
            tok("she", "PRON"),
            tok("wandered", "VERB"),
            tok("off", "ADP"),
            tok("and", "CCONJ"),
            tok("finally", "ADV"),
            tok("the", "DET"),
            tok("compass", "NOUN"),
            tok("gleamed", "VERB"),
        ],
    )
    m = map_span_to_tokens(lay, "the compass")
    assert m is not None
    assert m.block_slot == 1
    assert [t.form for t in m.tokens] == ["the", "compass"]


# --- realization typing + form recomputation ----------------------------------------


def test_realization_type_verbal_led():
    assert (
        realization_type([tok("to", "PART"), tok("wear", "VERB"), tok("gloves", "NOUN")])
        == "verbal"
    )
    assert realization_type([tok("stumbled", "VERB")]) == "verbal"


def test_realization_type_pronoun_and_proper_name():
    assert realization_type([tok("she", "PRON")]) == "pronoun"
    # pronoun-led clause is a non-nominal realization, not a pronoun
    assert realization_type([tok("she", "PRON"), tok("said", "VERB")]) == "verbal"
    assert realization_type([tok("we", "PRON"), tok("villagers", "NOUN")]) == "nominal"
    assert realization_type([tok("Bram", "PROPN"), tok("Hold", "PROPN")]) == "proper_name"


def test_recompute_verbal_span_never_bare_singular():
    caught = [tok("Caught", "VERB"), tok("by", "ADP"), tok("surprise", "NOUN")]
    assert recompute_form(caught, "en") == ("other", "non_nominal")


def test_recompute_zh_numeral_classifier_initial_pair():
    span = [tok("一", "NUM"), tok("个", "CL", deprel="clf"), tok("灰", "ADJ"), tok("瓦罐", "NOUN")]
    assert recompute_form(span, "zh") == ("other", "numeral_classifier")


def test_recompute_zh_numeral_presence_alone_is_not_classifier():
    # 绺 is semantically a measure word but parsed as plain NOUN: no
    # CL/M tag or clf deprel, so no numeral+classifier construction —
    # numeral presence alone must not trigger the label
    span = [tok("前面", "NOUN"), tok("一", "NUM"), tok("绺", "NOUN"), tok("头发", "NOUN")]
    assert recompute_form(span, "zh") == ("bare", "")


def test_recompute_zh_ordinal_and_floor_numeral_not_classifier():
    # 第二天: ordinal 第-prefix; 九楼: numeral as bare compound modifier
    ordinal = [tok("第二", "NUM"), tok("天", "NOUN", deprel="clf")]
    assert recompute_form(ordinal, "zh") == ("bare", "")
    floor = [tok("九", "NUM"), tok("楼", "NOUN", deprel="nmod")]
    assert recompute_form(floor, "zh") == ("bare", "")


def test_recompute_zh_attributive_de_not_possessive():
    span = [tok("浓云", "ADJ"), tok("低垂", "ADJ"), tok("的", "SCONJ"), tok("天空", "NOUN")]
    assert recompute_form(span, "zh") == ("bare", "")


def test_recompute_zh_genuine_possessive_kept():
    span = [tok("他", "PRON"), tok("的", "PART"), tok("弹珠", "NOUN")]
    assert recompute_form(span, "zh") == ("other", "possessive")


def test_recompute_zh_plain_bare_survives():
    assert recompute_form([tok("窗台上", "NOUN")], "zh") == ("bare", "")


def test_recompute_en_possessive_not_ordinary_definite():
    span = [tok("his", "DET"), tok("marbles", "NOUN")]
    assert recompute_form(span, "en") == ("other", "possessive")


def test_recompute_en_quantifier_not_bare_singular():
    span = [tok("in", "ADP"), tok("every", "DET"), tok("direction", "NOUN")]
    assert recompute_form(span, "en") == ("other", "quantifier")


def test_recompute_en_definite_untouched():
    span = [tok("the", "DET"), tok("lantern", "NOUN")]
    assert recompute_form(span, "en") == ("definite", "")


# --- routing: agreement ---------------------------------------------------------------


def _agree_mapping():
    lay = layout(
        "the rusty lantern gleamed",
        [
            tok("the", "DET"),
            tok("rusty", "ADJ"),
            tok("lantern", "NOUN"),
            tok("gleamed", "VERB"),
        ],
    )
    return map_span_to_tokens(lay, "the rusty lantern")


def test_route_machine_agreement_on_exact_span():
    m = _agree_mapping()
    dec = arbitrate_side(
        glm(span="the rusty lantern", paper="definite", detail=""),
        local(state="nominal_counterpart", chosen="the rusty lantern", evidence="both"),
        facts(mapping=m),
        "en",
    )
    assert dec.route == "machine_agreement"
    assert dec.counterpart == "the rusty lantern"
    assert dec.paper_form == "definite"
    assert not dec.requires_review
    assert dec.form_correction is None


def test_route_agreement_records_form_correction():
    # GLM labelled the span bare; the deterministic recomputation says
    # definite — the span survives, the form does not
    m = _agree_mapping()
    dec = arbitrate_side(
        glm(span="the rusty lantern", paper="bare_singular"),
        local(state="nominal_counterpart", chosen="lantern", evidence="both"),
        facts(mapping=m),
        "en",
    )
    assert dec.route == "machine_agreement"
    assert dec.paper_form == "definite"
    assert dec.form_correction == ("bare_singular", "")


def test_route_agreement_internal_disagreement_requires_review():
    m = _agree_mapping()
    dec = arbitrate_side(
        glm(span="the rusty lantern", paper="definite"),
        local(state="nominal_counterpart", chosen="the rusty lantern", evidence="disagreement"),
        facts(mapping=m),
        "en",
    )
    assert dec.route == "machine_agreement"
    assert dec.requires_review


# --- routing: omission guard ------------------------------------------------------------


def test_route_omission_blocked_by_local_overt_candidate():
    dec = arbitrate_side(
        glm(span="", paper="other", detail="omitted"),
        local(state="nominal_counterpart", chosen="his marbles", evidence="disagreement"),
        facts(),
        "en",
    )
    assert dec.route == "local_only"
    assert dec.counterpart == "his marbles"
    assert dec.evidence == "omission_blocked_local_overt"
    assert dec.requires_review


def test_route_omission_blocked_by_verbal_links():
    dec = arbitrate_side(
        glm(span="", paper="other", detail="omitted"),
        local(state="non_nominal_realization"),
        facts(),
        "en",
    )
    assert dec.route == "non_nominal_or_restructured"
    assert dec.evidence == "omission_blocked_local_verbal_links"


def test_route_omission_never_final_without_local_evidence():
    for state in ("no_overt_candidate", "ambiguous_multiple", "unresolved"):
        dec = arbitrate_side(
            glm(span="", paper="other", detail="omitted"),
            local(state=state),
            facts(),
            "en",
        )
        assert dec.route == "omission_candidate", state
        assert dec.requires_review


def test_route_omission_claim_requires_omitted_detail():
    # aligned + blank WITHOUT the omitted detail is a GLM retrieval
    # failure, not an omission claim
    dec = arbitrate_side(
        glm(span="", status="not_aligned"),
        local(state="no_overt_candidate"),
        facts(),
        "en",
    )
    assert dec.route == "unresolved"


def test_route_omission_no_anchor_is_candidate_not_omitted():
    dec = arbitrate_side(
        glm(span="", paper="other", detail="omitted"),
        local(state="no_overt_candidate"),
        facts(locus="no_anchor_context_available"),
        "zh",
    )
    assert dec.route == "omission_candidate"
    assert dec.evidence == "omission_candidate_no_anchor"


# --- routing: drift + locus -----------------------------------------------------------


def test_route_span_outside_aligned_locus_is_alignment_scope_conflict():
    # valid substring of the bounded retrieval context, absent from the
    # strict DP anchor: scope conflict, NOT a semantic-disagreement claim
    dec = arbitrate_side(
        glm(span="a crate of drills", paper="other", detail="indefinite"),
        local(state="ambiguous_multiple"),
        facts(in_locus_blocks=False),
        "en",
    )
    assert dec.route == "alignment_scope_conflict"
    assert dec.evidence == "glm_span_outside_aligned_locus"


def test_route_scope_conflict_fires_even_with_strong_local_choice():
    # the scope conflict precedes any referent comparison — a strong
    # local candidate cannot turn it into a semantic claim either way
    dec = arbitrate_side(
        glm(span="a crate of drills", paper="other"),
        local(state="nominal_counterpart", chosen="the cracked vase", evidence="both"),
        facts(in_locus_blocks=False),
        "en",
    )
    assert dec.route == "alignment_scope_conflict"


def test_route_in_block_unmappable_span_stays_llm_only_without_form():
    dec = arbitrate_side(
        glm(span="the rusty lantern", paper="definite"),
        local(state="ambiguous_multiple"),
        facts(in_locus_blocks=True, mapping=None),
        "en",
    )
    assert dec.route == "llm_only"
    assert dec.evidence == "llm_span_unmappable"
    assert dec.paper_form == ""


def test_route_outside_locus_without_strict_anchor_stays_llm_only():
    dec = arbitrate_side(
        glm(span="a crate of drills", paper="other"),
        local(state="unresolved"),
        facts(locus="no_anchor_context_available", in_locus_blocks=False),
        "en",
    )
    assert dec.route == "llm_only"
    assert dec.paper_form == ""


def test_route_divergent_local_strong_is_disagreement():
    m = _agree_mapping()
    dec = arbitrate_side(
        glm(span="the rusty lantern", paper="definite"),
        local(state="nominal_counterpart", chosen="the cracked vase", evidence="both"),
        facts(mapping=m),
        "en",
    )
    assert dec.route == "semantic_disagreement"
    assert dec.evidence == "divergent_local_strong"


def test_route_divergent_local_weak_is_disagreement():
    m = _agree_mapping()
    dec = arbitrate_side(
        glm(span="the rusty lantern", paper="definite"),
        local(state="alternate_referential_form", chosen="it", evidence="contextual_no_anchor"),
        facts(mapping=m),
        "en",
    )
    assert dec.route == "semantic_disagreement"
    assert dec.evidence == "divergent_local_weak"


def test_route_unreliable_locus_wins_over_everything():
    dec = arbitrate_side(
        glm(span="the rusty lantern", paper="definite"),
        local(state="nominal_counterpart", chosen="the rusty lantern", evidence="both"),
        facts(locus="retrieval_not_aligned", in_locus_blocks=False),
        "en",
    )
    assert dec.route == "retrieval_not_aligned"


def test_route_span_not_in_context_fails_closed():
    dec = arbitrate_side(
        glm(span="invented text", paper="other"),
        local(state="nominal_counterpart", chosen="the lantern", evidence="both"),
        facts(span_in_context=False),
        "en",
    )
    assert dec.route == "unresolved"
    assert dec.evidence == "glm_span_not_in_context"


# --- routing: verbal / llm_only / local_only ---------------------------------------------


def test_route_verbal_span_is_non_nominal_even_with_local_overlap():
    lay = layout(
        "she stumbled on the mat",
        [
            tok("she", "PRON"),
            tok("stumbled", "VERB"),
            tok("on", "ADP"),
            tok("the", "DET"),
            tok("mat", "NOUN"),
        ],
    )
    m = map_span_to_tokens(lay, "stumbled")
    dec = arbitrate_side(
        glm(span="stumbled", paper="other"),
        local(state="ambiguous_multiple"),
        facts(mapping=m),
        "en",
    )
    assert dec.route == "non_nominal_or_restructured"
    assert dec.detail_form == "non_nominal"


def test_route_llm_only_when_local_undecided():
    m = _agree_mapping()
    dec = arbitrate_side(
        glm(span="the rusty lantern", paper="definite"),
        local(state="ambiguous_multiple"),
        facts(mapping=m),
        "en",
    )
    assert dec.route == "llm_only"
    assert dec.requires_review
    assert dec.counterpart == "the rusty lantern"


def test_route_llm_only_evidence_records_eflomal_and_contextual_support():
    m = _agree_mapping()
    dec = arbitrate_side(
        glm(span="the rusty lantern"),
        local(state="ambiguous_multiple"),
        facts(mapping=m, eflomal_covers_span=True),
        "en",
    )
    assert dec.evidence == "llm_eflomal_supported"
    dec = arbitrate_side(
        glm(span="the rusty lantern"),
        local(state="unresolved"),
        facts(mapping=m, contextual_top1_covers=True),
        "en",
    )
    assert dec.evidence == "llm_contextual_top1"


def test_route_local_only_when_glm_proposes_nothing():
    dec = arbitrate_side(
        glm(span="", status="not_aligned"),
        local(state="nominal_counterpart", chosen="the lantern", evidence="contextual_supported"),
        facts(),
        "en",
    )
    assert dec.route == "local_only"
    assert dec.evidence == "local_only_no_glm_proposal"


def test_route_local_verbal_links_without_glm_span():
    dec = arbitrate_side(
        glm(span="", status="uncertain"),
        local(state="non_nominal_realization"),
        facts(),
        "zh",
    )
    assert dec.route == "non_nominal_or_restructured"


def test_route_unresolved_when_both_sides_empty():
    dec = arbitrate_side(
        glm(span="", status="not_aligned"),
        local(state="unresolved"),
        facts(),
        "en",
    )
    assert dec.route == "unresolved"


# --- artifact row: originals untouched ----------------------------------------------------


def test_build_row_preserves_glm_cells_and_adds_hybrid_columns():
    glm_row = {
        "id": "syn_dp_001",
        "german_pp": "im Schrank",
        "machine_en_counterpart": "his marbles",
        "machine_en_paper_form": "other",
        "machine_en_detail_form": "omitted",
        "machine_en_status": "aligned",
        "machine_annotation_model": "glm-test",
    }
    dec = arbitrate_side(
        glm(span="", paper="other", detail="omitted"),
        local(state="nominal_counterpart", chosen="his marbles", evidence="disagreement"),
        facts(),
        "en",
    )
    loc = local(state="nominal_counterpart", chosen="his marbles", evidence="disagreement")
    row = build_hybrid_row(glm_row, {"en": dec}, {"en": loc})
    # saved GLM cells pass through verbatim
    for key, value in glm_row.items():
        assert row[key] == value
    assert row["hybrid_en_route"] == "local_only"
    assert row["local_en_state"] == "nominal_counterpart"
    assert row["hybrid_zh_route"] == ""
    assert "hybrid_annotation_method" in row
    assert set(row) == set(HYBRID_COLUMNS)


def test_spanmapping_dataclass_carries_tokens():
    m = SpanMapping(block_slot=0, start_char=0, end_char=3, tokens=[tok("abc", "NOUN")])
    assert m.tokens[0].form == "abc"


# --- eligibility gate: constituent recovery ---------------------------------------------


def _zh_bed_layout():
    """床-pattern parse: modifiers attach as siblings, not dependents of
    the head, so recovery needs the attributive extension."""
    tokens = [
        tok("她", "PRON", "nsubj", "2"),
        tok("睡", "VERB", "root", "0"),
        tok("了", "PART", "aux", "2"),
        tok("一", "NUM", "nummod", "5"),
        tok("张", "NOUN", "clf", "4"),
        tok("坑", "NOUN", "obj", "2"),
        tok("洼", "NOUN", "flat", "6"),
        tok("、", "PUNCT", "punct", "6"),
        tok("高", "ADJ", "amod", "6"),
        tok("低", "ADJ", "flat", "9"),
        tok("的", "SCONJ", "mark:rel", "9"),
        tok("床上", "NOUN", "obj", "2"),
        tok("。", "PUNCT", "punct", "2"),
    ]
    text = "她睡了一张坑洼、高低的床上。"
    return SideLayout(
        block_ids=["syn#b001"], block_texts=[text], token_block=[0] * len(tokens), tokens=tokens
    )


def test_recovery_expands_anchor_to_full_constituent():
    lay = _zh_bed_layout()
    m = map_span_to_tokens(lay, "床")
    rec = recover_containing_constituent(m, lay, "zh")
    span = lay.block_texts[0][rec.start_char : rec.end_char]
    assert span == "一张坑洼、高低的床上"
    assert recompute_form(rec.tokens, "zh") == ("other", "numeral_classifier")


def test_recovery_stops_at_aspect_particle():
    # 了 (PART) terminates the attributive extension; only 的 passes
    lay = _zh_bed_layout()
    m = map_span_to_tokens(lay, "床上")
    rec = recover_containing_constituent(m, lay, "zh")
    span = lay.block_texts[0][rec.start_char : rec.end_char]
    assert span == "一张坑洼、高低的床上"


def test_recovery_head_skips_compound_modifier():
    tokens = [
        tok("the", "DET", "det", "3"),
        tok("living", "NOUN", "compound", "3"),
        tok("room", "NOUN", "obj", "4"),
        tok("gleamed", "VERB", "root", "0"),
    ]
    lay = SideLayout(
        block_ids=["syn#b001"],
        block_texts=["the living room gleamed"],
        token_block=[0, 0, 0, 0],
        tokens=tokens,
    )
    m = map_span_to_tokens(lay, "the living room")
    rec = recover_containing_constituent(m, lay, "en")
    span = lay.block_texts[0][rec.start_char : rec.end_char]
    assert span == "the living room"
    assert recompute_form(rec.tokens, "en") == ("definite", "")


def test_recovery_extension_stops_at_sentence_comma():
    tokens = [
        tok("醒来", "VERB", "root", "0"),
        tok("，", "PUNCT", "punct", "1"),
        tok("浓云", "ADJ", "amod", "5"),
        tok("低垂", "ADJ", "flat", "3"),
        tok("的", "SCONJ", "mark:rel", "3"),
        tok("天空", "NOUN", "nsubj", "1"),
    ]
    lay = SideLayout(
        block_ids=["syn#b001"],
        block_texts=["醒来，浓云低垂的天空"],
        token_block=[0] * 6,
        tokens=tokens,
    )
    m = map_span_to_tokens(lay, "天空")
    rec = recover_containing_constituent(m, lay, "zh")
    span = lay.block_texts[0][rec.start_char : rec.end_char]
    assert span == "浓云低垂的天空"
    assert recompute_form(rec.tokens, "zh") == ("bare", "")


# --- eligibility gate: routing -----------------------------------------------------------


def _en_small_layout():
    tokens = [
        tok("she", "PRON", "nsubj", "2"),
        tok("stumbled", "VERB", "root", "0"),
        tok("on", "ADP", "case", "4"),
        tok("the", "DET", "det", "5"),
        tok("mat", "NOUN", "obl", "2"),
    ]
    lay = SideLayout(
        block_ids=["syn#b001"],
        block_texts=["she stumbled on the mat"],
        token_block=[0] * 5,
        tokens=tokens,
    )
    return lay


def test_eligibility_anchor_nominal_is_comparable_with_constituent():
    lay = _en_small_layout()
    m = map_span_to_tokens(lay, "mat")
    dec = eligibility_side(
        glm(span="mat"),
        local(state="nominal_counterpart", chosen="the mat", evidence="both"),
        facts(mapping=m, layout=lay),
        [],
        "en",
    )
    assert dec.eligibility == "comparable_counterpart"
    assert dec.counterpart == "the mat"
    assert dec.paper_form == "definite"
    assert not dec.requires_review


def test_eligibility_verbal_anchor_without_candidate_not_comparable():
    lay = _en_small_layout()
    m = map_span_to_tokens(lay, "stumbled")
    dec = eligibility_side(
        glm(span="stumbled"),
        local(state="ambiguous_multiple"),
        facts(mapping=m, layout=lay),
        [],
        "en",
    )
    assert dec.eligibility == "no_comparable_nominal_counterpart"
    assert dec.evidence == "anchor_non_nominal_no_candidate"
    assert dec.requires_review


def test_eligibility_verbal_anchor_with_inner_nominal_recovers_constituent():
    # the anchor is verb-led but contains nominal tokens: the head
    # selection picks the inner nominal and recovers its constituent
    lay = SideLayout(
        block_ids=["syn#b001"],
        block_texts=["她打了个趔趄"],
        token_block=[0] * 4,
        tokens=[
            tok("她", "PRON", "nsubj", "2"),
            tok("打了", "VERB", "root", "0"),
            tok("个", "NOUN", "clf", "4"),
            tok("趔趄", "NOUN", "obj", "2"),
        ],
    )
    m = map_span_to_tokens(lay, "打了个趔趄")
    dec = eligibility_side(
        glm(span="打了个趔趄"),
        local(state="nominal_counterpart", chosen="个趔趄", evidence="both"),
        facts(mapping=m, layout=lay),
        [],
        "zh",
    )
    assert dec.eligibility == "comparable_counterpart"
    assert dec.counterpart == "个趔趄"
    assert dec.evidence == "anchor_nominal_local_agreement"


def test_eligibility_pure_verbal_anchor_falls_back_to_candidates():
    # a purely verbal anchor (no nominal token at all) consults the
    # saved candidate set before declaring no comparable counterpart
    lay = SideLayout(
        block_ids=["syn#b001"],
        block_texts=["她打了趔趄"],
        token_block=[0] * 3,
        tokens=[
            tok("她", "PRON", "nsubj", "2"),
            tok("打了", "VERB", "root", "0"),
            tok("趔趄", "VERB", "xcomp", "2"),
        ],
    )
    m = map_span_to_tokens(lay, "打了趔趄")
    dec = eligibility_side(
        glm(span="打了趔趄"),
        local(state="ambiguous_multiple"),
        facts(mapping=m, layout=lay),
        [CandidateRef(span="趔趄", category="noun", ctx_rank=1)],
        "zh",
    )
    assert dec.eligibility == "comparable_counterpart"
    assert dec.evidence == "anchor_non_nominal_candidate_overlap"


def test_eligibility_anchor_outside_locus_unresolved():
    dec = eligibility_side(
        glm(span="the fireplace"),
        local(state="ambiguous_multiple"),
        facts(mapping=None, in_locus_blocks=False),
        [],
        "en",
    )
    assert dec.eligibility == "unresolved"
    assert dec.evidence == "anchor_outside_aligned_locus"


def test_eligibility_unreliable_locus_unresolved():
    dec = eligibility_side(
        glm(span=""),
        local(state="retrieval_not_aligned"),
        facts(locus="retrieval_not_aligned"),
        [],
        "zh",
    )
    assert dec.eligibility == "unresolved"


def test_eligibility_blank_glm_local_candidate():
    lay = _en_small_layout()
    dec = eligibility_side(
        glm(span="", paper="other", detail="omitted"),
        local(state="nominal_counterpart", chosen="the mat", evidence="both"),
        facts(layout=lay),
        [],
        "en",
    )
    assert dec.eligibility == "comparable_counterpart"
    # span and form fields are never swapped
    assert dec.counterpart == "the mat"
    assert dec.paper_form == "definite"
    assert dec.detail_form == ""
    assert dec.evidence == "blank_glm_local_candidate"


def test_eligibility_blank_glm_ambiguous_unresolved():
    dec = eligibility_side(
        glm(span="", paper="other", detail="omitted"),
        local(state="ambiguous_multiple"),
        facts(),
        [],
        "zh",
    )
    assert dec.eligibility == "unresolved"
    assert dec.evidence == "blank_glm_local_ambiguous"


def test_eligibility_blank_glm_verbal_links_not_comparable_not_omitted():
    dec = eligibility_side(
        glm(span="", paper="other", detail="omitted"),
        local(state="non_nominal_realization"),
        facts(),
        [],
        "en",
    )
    assert dec.eligibility == "no_comparable_nominal_counterpart"
    assert dec.evidence == "blank_glm_local_verbal_realization"


def test_eligibility_blank_glm_local_absence_not_comparable():
    dec = eligibility_side(
        glm(span="", paper="other", detail="omitted"),
        local(state="no_overt_candidate"),
        facts(),
        [],
        "zh",
    )
    assert dec.eligibility == "no_comparable_nominal_counterpart"


# --- eligibility gate: row level + artifact ----------------------------------------------


def _dec(state):
    return EligibilityDecision(eligibility=state)


def test_row_eligibility_priority():
    both = {"en": _dec("comparable_counterpart"), "zh": _dec("comparable_counterpart")}
    assert row_eligibility_for(True, both) == "core_tuple_eligible"
    excluded = {
        "en": _dec("comparable_counterpart"),
        "zh": _dec("no_comparable_nominal_counterpart"),
    }
    assert row_eligibility_for(True, excluded) == "excluded_no_comparable_counterpart"
    # unresolved outranks exclusion: an undecided side is never read as
    # a linguistic absence
    mixed = {"en": _dec("no_comparable_nominal_counterpart"), "zh": _dec("unresolved")}
    assert row_eligibility_for(True, mixed) == "unresolved"
    assert row_eligibility_for(False, both) == "unresolved"


def test_build_eligibility_row_preserves_glm_cells():
    glm_row = {
        "id": "syn_dp_002",
        "german_pp": "ins Wohnzimmer",
        "machine_en_counterpart": "the living room",
        "machine_en_status": "aligned",
        "machine_zh_counterpart": "起居室",
        "machine_zh_status": "aligned",
    }
    decisions = {
        "en": EligibilityDecision(
            eligibility="comparable_counterpart",
            counterpart="the living room",
            paper_form="definite",
        ),
        "zh": EligibilityDecision(
            eligibility="no_comparable_nominal_counterpart",
            evidence="anchor_non_nominal_no_candidate",
        ),
    }
    row = build_eligibility_row(glm_row, decisions, "excluded_no_comparable_counterpart")
    for key, value in glm_row.items():
        assert row[key] == value
    assert row["elig_en_state"] == "comparable_counterpart"
    assert row["elig_zh_state"] == "no_comparable_nominal_counterpart"
    assert row["elig_zh_counterpart"] == ""
    assert row["row_eligibility"] == "excluded_no_comparable_counterpart"
    assert set(row) == set(ELIGIBILITY_COLUMNS)
