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
    HYBRID_COLUMNS,
    GlmSide,
    LocalSide,
    SideFacts,
    SpanMapping,
    arbitrate_side,
    build_hybrid_row,
    map_span_to_tokens,
    realization_type,
    recompute_form,
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


def test_recompute_zh_overt_numeral_anywhere_blocks_bare():
    # classifier mistagged as plain NOUN by the parser — the overt
    # numeral alone still defeats a bare reading
    span = [tok("前面", "NOUN"), tok("一", "NUM"), tok("绺", "NOUN"), tok("头发", "NOUN")]
    assert recompute_form(span, "zh") == ("other", "numeral_classifier")


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
