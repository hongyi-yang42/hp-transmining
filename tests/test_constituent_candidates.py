"""Synthetic-fixture tests for the constituent-constrained pilot (v2).

All fixture text is invented; no novel text. Covers parse-driven
nominal constituent recovery (adposition out, determiners /
demonstratives / numeral+classifier in), the pronoun and proper-name
realizations, the agreement-based routing state machine (incl. the
no-link ≠ omission and omission_candidate ≠ omitted distinctions),
reproducibility, and that the production annotation path is unchanged.
The contextual signal is injected as plain similarities — the encoder
itself is not needed here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from hp_corpus.annotation_csv import ANNOTATOR_COLUMNS
from hp_corpus.constituent_candidates import (
    CONSTITUENT_COLUMNS,
    CONTEXTUAL_FLOOR,
    CONTEXTUAL_MARGIN,
    RouteResult,
    build_constituent_row,
    eflomal_supported_indices,
    generate_candidates,
    nominal_expression_indices,
    route_side,
)
from hp_corpus.contextual_similarity import cosine, mean_vector, rank_candidates
from hp_corpus.deterministic_tuples import (
    SideLayout,
    Token,
    chinese_form,
    english_form,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tok(
    form: str,
    upos: str = "NOUN",
    head: str = "0",
    deprel: str = "root",
    feats: str = "",
) -> Token:
    return Token(form=form, upos=upos, feats=feats, head=head, deprel=deprel)


def _layout(tokens: list[Token], text: str) -> SideLayout:
    return SideLayout(
        block_ids=["b1"],
        block_texts=[text],
        token_block=[0] * len(tokens),
        tokens=tokens,
    )


# --- nominal constituent recovery ---------------------------------------------

# "He looked in that old cupboard ."
_EN_TOKENS = [
    _tok("He", "PRON", "2", "nsubj"),
    _tok("looked", "VERB", "0", "root"),
    _tok("in", "ADP", "6", "case"),
    _tok("that", "DET", "6", "det"),
    _tok("old", "ADJ", "6", "amod"),
    _tok("cupboard", "NOUN", "2", "obl", feats="Number=Sing"),
    _tok(".", "PUNCT", "2", "punct"),
]
_EN_TEXT = "He looked in that old cupboard."

# "他 在 那个 柜子 里 找 到 了 钥匙 。"
_ZH_TOKENS = [
    _tok("他", "PRON", "6", "nsubj"),
    _tok("在", "ADP", "4", "case"),
    _tok("那个", "DET", "4", "det"),
    _tok("柜子", "NOUN", "6", "obl"),
    _tok("里", "ADP", "4", "case"),
    _tok("找", "VERB", "0", "root"),
    _tok("到", "VERB", "6", "comp:verb"),
    _tok("了", "PART", "6", "aux"),
    _tok("钥匙", "NOUN", "6", "obj"),
    _tok("。", "PUNCT", "6", "punct"),
]
_ZH_TEXT = "他在那个柜子里找到了钥匙。"

# "她 在 一 家 公司 工作 。" — classifier attaches to the numeral
# (heads are CoNLL-U 1-based)
_ZH_NUMCL_TOKENS = [
    _tok("她", "PRON", "6", "nsubj"),
    _tok("在", "ADP", "5", "case"),
    _tok("一", "NUM", "5", "nummod"),
    _tok("家", "NOUN", "3", "clf"),
    _tok("公司", "NOUN", "6", "obl"),
    _tok("工作", "VERB", "0", "root"),
    _tok("。", "PUNCT", "6", "punct"),
]
_ZH_NUMCL_TEXT = "她在一家公司工作。"


class TestNominalExpressionRecovery:
    def test_adposition_excluded_determiner_retained_en(self):
        lo, hi = nominal_expression_indices(_EN_TOKENS, 5, "en")  # cupboard
        assert [_EN_TOKENS[i].form for i in range(lo, hi + 1)] == ["that", "old", "cupboard"]

    def test_boundary_convention_adposition_never_in_span(self):
        for head_local in range(len(_EN_TOKENS)):
            lo, hi = nominal_expression_indices(_EN_TOKENS, head_local, "en")
            for i in range(lo, hi + 1):
                if i == head_local:
                    continue
                assert _EN_TOKENS[i].upos != "ADP", (head_local, i)

    def test_pronoun_head_is_own_candidate(self):
        lo, hi = nominal_expression_indices(_EN_TOKENS, 0, "en")
        assert (lo, hi) == (0, 0)

    def test_zh_demonstrative_retained(self):
        lo, hi = nominal_expression_indices(_ZH_TOKENS, 3, "zh")  # 柜子
        forms = [_ZH_TOKENS[i].form for i in range(lo, hi + 1)]
        assert forms == ["那个", "柜子"]

    def test_zh_numeral_classifier_retained_via_two_level_rule(self):
        lo, hi = nominal_expression_indices(_ZH_NUMCL_TOKENS, 4, "zh")  # 公司
        forms = [_ZH_NUMCL_TOKENS[i].form for i in range(lo, hi + 1)]
        assert forms == ["一", "家", "公司"]

    def test_hull_cap_falls_back_to_head(self):
        # head with 9 amod dependents > MAX_EXPRESSION_TOKENS
        toks = [_tok(f"m{i}", "ADJ", "9", "amod") for i in range(8)]
        toks.append(_tok("thing", "NOUN", "0", "root"))
        lo, hi = nominal_expression_indices(toks, 8, "en")
        assert (lo, hi) == (8, 8)


class TestGenerateCandidates:
    def test_candidate_spans_are_exact_substrings_with_categories(self):
        layout = _layout(_EN_TOKENS, _EN_TEXT)
        cands = generate_candidates(layout, "en")
        by_head = {c.head_global: c for c in cands}
        cupboard = by_head[5]
        assert cupboard.span_text == "that old cupboard"
        assert cupboard.category == "noun"
        assert by_head[0].category == "pronoun"
        assert by_head[0].span_text == "He"

    def test_zh_numeral_classifier_candidate(self):
        layout = _layout(_ZH_NUMCL_TOKENS, _ZH_NUMCL_TEXT)
        cands = generate_candidates(layout, "zh")
        by_head = {c.head_global: c for c in cands}
        assert by_head[4].span_text == "一家公司"

    def test_only_referential_heads_generate(self):
        layout = _layout(_EN_TOKENS, _EN_TEXT)
        for c in generate_candidates(layout, "en"):
            assert _EN_TOKENS[c.head_global].upos in ("NOUN", "PROPN", "PRON")

    def test_generation_is_eflomal_free_and_deterministic(self):
        layout = _layout(_ZH_TOKENS, _ZH_TEXT)
        first = [(c.lo, c.hi, c.span_text) for c in generate_candidates(layout, "zh")]
        second = [(c.lo, c.hi, c.span_text) for c in generate_candidates(layout, "zh")]
        assert first == second


# --- form classification on the nominal expression ------------------------------


class TestFormOnNominalExpression:
    def test_en_demonstrative_form(self):
        toks = [_EN_TOKENS[i] for i in (3, 4, 5)]
        assert english_form(toks) == ("demonstrative", "")

    def test_en_definite_form(self):
        toks = [
            _tok("the", "DET", "2", "det"),
            _tok("hallway", "NOUN", "0", "root", feats="Number=Sing"),
        ]
        assert english_form(toks) == ("definite", "")

    def test_zh_numeral_classifier_form(self):
        toks = [_ZH_NUMCL_TOKENS[i] for i in (2, 3, 4)]
        assert chinese_form(toks) == ("other", "numeral_classifier")

    def test_zh_classifier_deprel_recognized_even_as_noun_upos(self):
        toks = [
            _tok("一", "NUM", "3", "nummod"),
            _tok("个", "NOUN", "1", "clf"),
            _tok("房子", "NOUN", "0", "root"),
        ]
        assert chinese_form(toks) == ("other", "numeral_classifier")

    def test_zh_demonstrative_form(self):
        toks = [_ZH_TOKENS[i] for i in (2, 3)]
        assert chinese_form(toks) == ("demonstrative", "")

    def test_zh_bare_form(self):
        toks = [_ZH_TOKENS[i] for i in (3,)]
        assert chinese_form(toks) == ("bare", "")

    def test_pronoun_and_proper_name_details(self):
        assert english_form([_tok("it", "PRON")]) == ("other", "pronoun")
        assert chinese_form([_tok("他", "PRON")]) == ("other", "pronoun")
        assert english_form([_tok("Privet", "PROPN"), _tok("Drive", "PROPN", "1", "flat")]) == (
            "other", "proper_name"
        )


# --- eflomal evidence --------------------------------------------------------------


class TestEflomalSupport:
    def test_support_from_consensus_links_into_span(self):
        layout = _layout(_EN_TOKENS, _EN_TEXT)
        cands = generate_candidates(layout, "en")
        cupboard = next(c for c in cands if c.head_global == 5)
        # consensus link from DE token 7 into "that"(3)/"old"(4)/"cupboard"(5)
        fwd = {(7, 5)}
        rev = {(7, 5)}
        supported = eflomal_supported_indices(cands, range(7, 8), fwd, rev)
        assert supported == {cands.index(cupboard)}

    def test_no_consensus_no_support(self):
        layout = _layout(_EN_TOKENS, _EN_TEXT)
        cands = generate_candidates(layout, "en")
        assert eflomal_supported_indices(cands, range(7, 8), {(7, 5)}, {(5, 6)}) == set()


# --- routing -------------------------------------------------------------------------


def _cands_en():
    layout = _layout(_EN_TOKENS, _EN_TEXT)
    return generate_candidates(layout, "en"), layout


class TestRouting:
    def test_high_confidence_both_signals_agree(self):
        cands, _ = _cands_en()
        idx = next(i for i, c in enumerate(cands) if c.head_global == 5)
        sims = {i: 0.10 for i in range(len(cands))}
        sims[idx] = 0.80
        result = route_side(cands, {idx}, sims, locus_reliable=True)
        assert result.state == "nominal_counterpart"
        assert result.evidence == "both"
        assert result.chosen is cands[idx]

    def test_eflomal_only_when_contextual_prefers_other(self):
        cands, _ = _cands_en()
        idx = next(i for i, c in enumerate(cands) if c.head_global == 5)
        other = next(i for i, c in enumerate(cands) if c.head_global == 0)
        sims = {i: 0.10 for i in range(len(cands))}
        sims[other] = 0.80
        result = route_side(cands, {idx}, sims, locus_reliable=True)
        assert result.state == "nominal_counterpart"
        assert result.evidence == "eflomal_only"

    def test_contextual_only_with_clear_margin_above_floor(self):
        cands, _ = _cands_en()
        idx = next(i for i, c in enumerate(cands) if c.head_global == 5)
        sims = {i: 0.10 for i in range(len(cands))}
        sims[idx] = 0.80
        result = route_side(cands, set(), sims, locus_reliable=True)
        assert result.state == "nominal_counterpart"
        assert result.evidence == "contextual_only"

    def test_disagreement_with_multiple_supported_is_ambiguous(self):
        # three candidates: contextual prefers the one eflomal does NOT
        # support, while two others carry eflomal links
        layout = _layout(_ZH_TOKENS, _ZH_TEXT)
        cands = generate_candidates(layout, "zh")
        by_head = {c.head_global: c for c in cands}
        supported = {
            cands.index(by_head[3]),  # 那个柜子
            cands.index(by_head[8]),  # 钥匙
        }
        sims = {i: 0.10 for i in range(len(cands))}
        sims[cands.index(by_head[0])] = 0.80  # 他 — top, unsupported
        result = route_side(cands, supported, sims, locus_reliable=True)
        assert result.state == "ambiguous_multiple"
        assert result.chosen is None

    def test_too_close_top2_with_no_eflomal_is_ambiguous(self):
        cands, _ = _cands_en()
        sims = {i: 0.55 for i in range(len(cands))}
        result = route_side(cands, set(), sims, locus_reliable=True)
        assert result.state == "ambiguous_multiple"

    def test_no_link_is_not_omission(self):
        cands, _ = _cands_en()
        sims = {i: 0.55 for i in range(len(cands))}  # plausible, unclear
        result = route_side(cands, set(), sims, locus_reliable=True)
        assert result.state == "ambiguous_multiple"  # NOT omission_candidate

    def test_omission_candidate_requires_positive_absence(self):
        cands, _ = _cands_en()
        sims = {i: 0.20 for i in range(len(cands))}  # all below floor
        result = route_side(cands, set(), sims, locus_reliable=True)
        assert result.state == "omission_candidate"

    def test_omission_candidate_is_not_final_omitted(self):
        cands, _ = _cands_en()
        sims = {i: 0.20 for i in range(len(cands))}
        result = route_side(cands, set(), sims, locus_reliable=True)
        assert result.state == "omission_candidate"
        assert result.state != "omitted"
        assert result.chosen is None

    def test_no_candidates_on_reliable_locus_is_omission_candidate(self):
        result = route_side([], set(), {}, locus_reliable=True)
        assert result.state == "omission_candidate"

    def test_unreliable_locus_is_not_aligned(self):
        cands, _ = _cands_en()
        sims = {i: 0.8 for i in range(len(cands))}
        result = route_side(cands, set(), sims, locus_reliable=False)
        assert result.state == "not_aligned"

    def test_pronoun_realization_routes_non_nominal(self):
        cands, _ = _cands_en()
        idx = next(i for i, c in enumerate(cands) if c.head_global == 0)
        sims = {i: 0.10 for i in range(len(cands))}
        sims[idx] = 0.90
        result = route_side(cands, {idx}, sims, locus_reliable=True)
        assert result.state == "non_nominal_counterpart"

    def test_proper_name_realization_routes_non_nominal(self):
        toks = [
            _tok("at", "ADP", "3", "case"),
            _tok("Privet", "PROPN", "3", "compound"),
            _tok("Drive", "PROPN", "0", "root"),
        ]
        layout = _layout(toks, "at Privet Drive")
        cands = generate_candidates(layout, "en")
        idx = next(i for i, c in enumerate(cands) if c.category == "proper_name")
        sims = {i: 0.10 for i in range(len(cands))}
        sims[idx] = 0.90
        result = route_side(cands, {idx}, sims, locus_reliable=True)
        assert result.state == "non_nominal_counterpart"

    def test_routing_is_reproducible(self):
        cands, _ = _cands_en()
        idx = next(i for i, c in enumerate(cands) if c.head_global == 5)
        sims = {i: 0.10 for i in range(len(cands))}
        sims[idx] = 0.80
        first = route_side(cands, {idx}, sims, locus_reliable=True)
        for _ in range(3):
            again = route_side(cands, {idx}, sims, locus_reliable=True)
            assert (again.state, again.evidence, again.chosen) == (
                first.state, first.evidence, first.chosen
            )


# --- artifact row ----------------------------------------------------------------------


class TestArtifactRow:
    def _row(self):
        cands, layout = _cands_en()
        idx = next(i for i, c in enumerate(cands) if c.head_global == 5)
        routed = {"en": RouteResult("nominal_counterpart", chosen=cands[idx], evidence="both")}
        forms = {"en": english_form(layout.tokens[cands[idx].lo : cands[idx].hi + 1])}
        master = {
            "datapoint_id": "dp1",
            "chapter": "1",
            "de_form": "contracted",
            "de_pp_surface": "im Schrank",
            "de_head_lemma": "Schrank",
            "de_sentence_text": "Er sah etwas im Schrank.",
            "en_context_text": "He looked in that old cupboard.",
            "en_context_provenance": "anchor_window",
            "zh_context_text": "他在那个柜子里找到了钥匙。",
            "zh_context_provenance": "anchor_window",
            "de_det_xpos": "-",
            "de_det_deprel": "-",
            "source_row_sha256": "d" * 64,
        }
        return build_constituent_row(
            master, routed, {"en": forms["en"], "zh": ("", "")}, {"en": len(cands), "zh": 0},
            pack_stratum="routine", method_params_id="x",
        )

    def test_schema_and_values(self):
        row = self._row()
        assert tuple(row) == CONSTITUENT_COLUMNS
        assert row["machine_en_status"] == "nominal_counterpart"
        assert row["machine_en_counterpart"] == "that old cupboard"
        assert row["machine_en_paper_form"] == "demonstrative"
        assert row["machine_en_evidence"] == "both"

    def test_omission_candidate_row_has_blank_form_not_omitted(self):
        master = {
            "datapoint_id": "dp2", "chapter": "1", "de_form": "contracted",
            "de_pp_surface": "x", "de_head_lemma": "x", "de_sentence_text": "x.",
            "en_context_text": "y.", "en_context_provenance": "anchor_window",
            "zh_context_text": "z。", "zh_context_provenance": "anchor_window",
            "de_det_xpos": "-", "de_det_deprel": "-", "source_row_sha256": "e" * 64,
        }
        row = build_constituent_row(
            master,
            {"en": RouteResult("omission_candidate"), "zh": RouteResult("not_aligned")},
            {"en": ("", ""), "zh": ("", "")},
            {"en": 3, "zh": 0},
            pack_stratum="routine", method_params_id="x",
        )
        assert row["machine_en_status"] == "omission_candidate"
        assert row["machine_en_paper_form"] == ""
        assert row["machine_en_detail_form"] == ""
        assert row["machine_zh_status"] == "not_aligned"

    def test_machine_and_human_columns_still_disjoint(self):
        assert set(CONSTITUENT_COLUMNS).isdisjoint(ANNOTATOR_COLUMNS)


# --- contextual helpers -------------------------------------------------------------------


class TestContextualHelpers:
    def test_cosine_and_mean(self):
        a = [1.0, 0.0]
        b = [0.0, 2.0]
        assert cosine(a, a) == 1.0
        assert abs(cosine(a, b)) < 1e-9
        assert mean_vector([a, b]) == [0.5, 1.0]

    def test_rank_candidates_pure_function(self):
        vecs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

        class C:
            lo = 0
            hi = 0

        class C2:
            lo = 2
            hi = 2

        r1 = rank_candidates(vecs, range(0, 1), vecs, [C(), C2()])
        r2 = rank_candidates(vecs, range(0, 1), vecs, [C(), C2()])
        assert r1 == r2
        assert r1[0] == 1.0  # identical vectors

    def test_frozen_operational_constants(self):
        assert CONTEXTUAL_MARGIN == 0.05
        assert CONTEXTUAL_FLOOR == 0.50


# --- production path unchanged ---------------------------------------------------------------


class TestProductionPathUntouched:
    def test_production_builder_leaves_annotator_columns_blank(self):
        build = _load_script("build_annotation_csv")
        master = {
            "datapoint_id": "dp1", "chapter": "1", "de_form": "contracted",
            "de_pp_surface": "x", "de_head_lemma": "x", "de_sentence_text": "x.",
            "en_context_text": "y.", "en_context_provenance": "anchor_window",
            "zh_context_text": "z。", "zh_context_provenance": "anchor_window",
            "source_row_sha256": "f" * 64,
        }
        rows = build.build_csv_rows([master])
        for col in ANNOTATOR_COLUMNS:
            assert rows[0][col] == ""
