"""Synthetic-fixture tests for the deterministic-first tuple baseline.

All fixture text is invented; no novel text. Covers the CoNLL-U token
reader, token-link → exact-substring span recovery (incl. determiner
absorption and the failure modes), the EN/ZH deterministic form rules,
the DE structural-gate proposal, reproducibility of the pure pipeline,
and that the production annotation behavior is unchanged. eflomal
itself is only exercised by an opt-in integration test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from hp_corpus.annotation_csv import ANNOTATOR_COLUMNS
from hp_corpus.deterministic_tuples import (
    DETERMINISTIC_METHOD_ID,
    MAX_SPAN_TOKENS,
    SpanRecovery,
    Token,
    blocks_for_segment,
    build_deterministic_row,
    build_side_layout,
    chinese_form,
    de_valid_proposal,
    eflomal_input_line,
    english_form,
    intersection_links,
    parse_links_file,
    read_conllu,
    recover_target_span,
)
from hp_corpus.machine_preannotation import MACHINE_TUPLE_COLUMNS

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MINI_CONLLU_EN = """# sent_id = syn_en_ch01_p0001_s001#b001
# text = Mr. Grunt saw the cabinet door in the hallway.
1\tMr.\t_\tPROPN\t_\t_\t2\tflat\t_\t_
2\tGrunt\t_\tPROPN\t_\tNumber=Sing\t3\tnsubj\t_\t_
3\tsaw\t_\tVERB\t_\t_\t0\troot\t_\t_
4\tthe\t_\tDET\t_\tDefinite=Def\t6\tdet\t_\t_
5-6\tcabinet-door\t_\t_\t_\t_\t_\t_\t_\t_
5\tcabinet\t_\tNOUN\t_\tNumber=Sing\t7\tobj\t_\t_
6\tdoor\t_\tNOUN\t_\tNumber=Sing\t3\tobj\t_\t_
7\tin\t_\tADP\t_\t_\t9\tcase\t_\t_
8\tthe\t_\tDET\t_\tDefinite=Def\t9\tdet\t_\t_
9\thallway\t_\tNOUN\t_\tNumber=Sing\t3\tobl\t_\t_
10\t.\t_\tPUNCT\t_\t_\t3\tpunct\t_\t_

# sent_id = syn_en_ch01_p0001_s002#b001
# text = It was empty.
1\tIt\t_\tPRON\t_\t_\t2\tnsubj\t_\t_
2\twas\t_\tAUX\t_\t_\t0\troot\t_\t_
3\tempty\t_\tADJ\t_\t_\t2\tobj\t_\t_
4\t.\t_\tPUNCT\t_\t_\t2\tpunct\t_\t_
"""

MINI_CONLLU_ZH = """# sent_id = syn_zh_ch01_p0001_s001#b001
# text = 他在那个柜子里找到了钥匙。
1\t他\t_\tPRON\t_\t_\t4\tnsubj\t_\t_
2\t在\t_\tADP\t_\t_\t4\tcase\t_\t_
3\t那个\t_\tDET\t_\t_\t4\tdet\t_\t_
4\t柜子\t_\tNOUN\t_\t_\t4\troot\t_\t_
5\t里\t_\tNOUN\t_\t_\t2\tcase:loc\t_\t_
6\t找到\t_\tVERB\t_\t_\t4\troot\t_\t_
7\t了\t_\tPART\t_\t_\t6\taux\t_\t_
8\t钥匙\t_\tNOUN\t_\t_\t6\tobj\t_\t_
9\t。\t_\tPUNCT\t_\t_\t6\tpunct\t_\t_
"""


@pytest.fixture(scope="module")
def en_sentences():
    return read_conllu(_write_tmp(MINI_CONLLU_EN, "en.conllu"))


@pytest.fixture(scope="module")
def zh_sentences():
    return read_conllu(_write_tmp(MINI_CONLLU_ZH, "zh.conllu"))


def _write_tmp(content: str, name: str) -> Path:
    import tempfile

    d = tempfile.mkdtemp()
    p = Path(d) / name
    p.write_text(content, encoding="utf-8")
    return p


# --- conllu reading ------------------------------------------------------------------


class TestReadConllu:
    def test_tokens_and_text(self, en_sentences):
        sent = en_sentences["syn_en_ch01_p0001_s001#b001"]
        assert sent.text.startswith("Mr. Grunt saw")
        forms = [t.form for t in sent.tokens]
        # the 5-6 range line is skipped: 10 columns minus 1 range = 10 tokens
        assert forms == [
            "Mr.", "Grunt", "saw", "the", "cabinet", "door",
            "in", "the", "hallway", ".",
        ]
        assert sent.tokens[4].upos == "NOUN"
        assert "Number=Sing" in sent.tokens[4].feats

    def test_blocks_for_segment_orders_by_b_ordinal(self, en_sentences):
        assert blocks_for_segment(
            en_sentences, "syn_en_ch01_p0001_s001"
        ) == ["syn_en_ch01_p0001_s001#b001"]

    def test_side_layout_concatenation(self, en_sentences):
        layout = build_side_layout(["syn_en_ch01_p0001_s001"], en_sentences)
        assert len(layout.tokens) == 10
        assert layout.token_block == [0] * 10
        assert layout.block_texts[0].startswith("Mr. Grunt")
        assert eflomal_input_line(layout).startswith("Mr. Grunt saw the cabinet")

    def test_eflomal_line_one_field_per_token_despite_inner_spaces(self):
        from hp_corpus.deterministic_tuples import SideLayout

        # CoNLL-U forms with internal whitespace (the ZH "… 」" artifact)
        # must become exactly one whitespace field, or eflomal's indices
        # drift off the layout tokens.
        layout = SideLayout(
            block_ids=["b"],
            block_texts=["「… 」柜子"],
            token_block=[0, 0, 0],
            tokens=[
                Token(form="「", upos="PUNCT", feats=""),
                Token(form="… 」", upos="PUNCT", feats=""),
                Token(form="柜子", upos="NOUN", feats=""),
            ],
        )
        line = eflomal_input_line(layout)
        assert line.split() == ["「", "…□」", "柜子"]


# --- span recovery ---------------------------------------------------------------------


def _en_layout(en_sentences):
    return build_side_layout(["syn_en_ch01_p0001_s001"], en_sentences)


class TestSpanRecovery:
    def test_contiguous_links_recover_exact_substring(self, en_sentences):
        de = _en_layout(en_sentences)
        tgt = _en_layout(en_sentences)
        # link "Grunt" (1); preceding "Mr." is not a determiner
        fwd = {(2, 1)}
        rev = {(2, 1)}
        rec = recover_target_span(range(1, 3), de, tgt, fwd, rev, lang="en")
        assert rec.status == "aligned"
        assert rec.span_text == "Grunt"
        assert rec.tgt_block_id.endswith("#b001")

    def test_determiner_absorption_into_span(self, en_sentences):
        de = _en_layout(en_sentences)
        tgt = _en_layout(en_sentences)
        # link "hallway" only; preceding "the" must be absorbed
        fwd = {(7, 8)}
        rev = {(7, 8)}
        rec = recover_target_span(range(6, 8), de, tgt, fwd, rev, lang="en")
        assert rec.status == "aligned"
        assert rec.span_text == "the hallway"
        assert rec.absorbed_determiner is True

    def test_no_intersection_links_is_unresolved(self, en_sentences):
        de = _en_layout(en_sentences)
        tgt = _en_layout(en_sentences)
        fwd = {(7, 8)}
        rev = {(0, 9)}  # disjoint with fwd — no consensus link
        rec = recover_target_span(range(6, 8), de, tgt, fwd, rev, lang="en")
        assert rec.status == "unresolved"
        assert rec.reason == "no_intersection_links"
        assert rec.span_text == ""

    def test_links_across_blocks_is_unresolved(self, en_sentences):
        de = _en_layout(en_sentences)
        tgt = build_side_layout(
            ["syn_en_ch01_p0001_s001", "syn_en_ch01_p0001_s002"], en_sentences
        )
        # token 8 in block 0, token 11 ("empty") in block 1
        fwd = {(7, 8), (7, 11)}
        rev = {(7, 8), (7, 11)}
        rec = recover_target_span(range(6, 8), de, tgt, fwd, rev, lang="en")
        assert rec.status == "unresolved"
        assert rec.reason == "links_across_blocks"

    def test_discontinuous_hull_within_limit_recovered_verbatim(self, en_sentences):
        de = _en_layout(en_sentences)
        tgt = _en_layout(en_sentences)
        # links to "cabinet"(4) and "hallway"(8): hull spans the whole NP;
        # the "the" before "cabinet" is absorbed
        fwd = {(7, 4), (7, 8)}
        rev = {(7, 4), (7, 8)}
        rec = recover_target_span(range(6, 8), de, tgt, fwd, rev, lang="en")
        assert rec.status == "aligned"
        assert rec.span_text == "the cabinet door in the hallway"

    def test_oversized_hull_is_unresolved(self):
        from hp_corpus.deterministic_tuples import SideLayout

        n = 15
        tgt = SideLayout(
            block_ids=["b1"],
            block_texts=[" ".join(f"t{i}" for i in range(n))],
            token_block=[0] * n,
            tokens=[Token(form=f"t{i}", upos="NOUN", feats="") for i in range(n)],
        )
        de = SideLayout(
            block_ids=["b0"],
            block_texts=["pp"],
            token_block=[0],
            tokens=[Token(form="pp", upos="NOUN", feats="")],
        )
        fwd = {(0, 0), (0, 14)}
        rev = {(0, 0), (0, 14)}
        rec = recover_target_span(range(0, 1), de, tgt, fwd, rev, lang="en")
        assert rec.status == "unresolved"
        assert rec.reason == "span_hull_too_wide"
        assert MAX_SPAN_TOKENS == 12

    def test_invalid_de_range_is_unresolved(self, en_sentences):
        de = _en_layout(en_sentences)
        tgt = _en_layout(en_sentences)
        rec = recover_target_span(range(50, 60), de, tgt, set(), set(), lang="en")
        assert rec.status == "unresolved"
        assert rec.reason == "de_token_range_invalid"

    def test_zh_span_recovery_without_determiner(self, zh_sentences):
        de = build_side_layout(["syn_zh_ch01_p0001_s001"], zh_sentences)
        tgt = build_side_layout(["syn_zh_ch01_p0001_s001"], zh_sentences)
        # link 钥匙 (token 7); preceding 了 is not a demonstrative → no absorption
        fwd = {(6, 7)}
        rev = {(6, 7)}
        rec = recover_target_span(range(5, 7), de, tgt, fwd, rev, lang="zh")
        assert rec.status == "aligned"
        assert rec.span_text == "钥匙"

    def test_zh_demonstrative_absorption(self, zh_sentences):
        de = build_side_layout(["syn_zh_ch01_p0001_s001"], zh_sentences)
        tgt = build_side_layout(["syn_zh_ch01_p0001_s001"], zh_sentences)
        # link 柜子 (token 3); preceding 那个 (token 2) absorbed
        fwd = {(2, 3)}
        rev = {(2, 3)}
        rec = recover_target_span(range(2, 4), de, tgt, fwd, rev, lang="zh")
        assert rec.status == "aligned"
        assert rec.span_text == "那个柜子"
        assert rec.absorbed_determiner is True


# --- EN form rules -----------------------------------------------------------------------


def _tok(form: str, upos: str = "NOUN", feats: str = "Number=Sing") -> Token:
    return Token(form=form, upos=upos, feats=feats)


class TestEnglishFormRules:
    def test_definite(self):
        assert english_form([_tok("in", "ADP"), _tok("the", "DET"), _tok("house")]) == (
            "definite", ""
        )

    def test_demonstrative(self):
        assert english_form([_tok("that", "DET"), _tok("house")]) == ("demonstrative", "")

    def test_bare_singular(self):
        assert english_form([_tok("in", "ADP"), _tok("winter", feats="Number=Sing")]) == (
            "bare_singular", ""
        )

    def test_indefinite_detail(self):
        assert english_form([_tok("a", "DET"), _tok("house")]) == ("other", "indefinite")

    def test_possessive_detail(self):
        assert english_form([_tok("his", "DET"), _tok("house")]) == ("other", "possessive")

    def test_pronoun_detail(self):
        assert english_form([_tok("it", "PRON")]) == ("other", "pronoun")

    def test_proper_name_detail(self):
        assert english_form([_tok("Privet", "PROPN"), _tok("Drive", "PROPN")]) == (
            "other", "proper_name"
        )

    def test_plural_without_det_is_other(self):
        assert english_form([_tok("drills", feats="Number=Plur")]) == ("other", "")

    def test_leading_prep_and_trailing_punct_stripped(self):
        toks = [_tok("in", "ADP"), _tok("the", "DET"), _tok("hallway"), _tok(".", "PUNCT")]
        assert english_form(toks) == ("definite", "")

    def test_output_space_is_frozen_vocab(self):
        for toks in (
            [_tok("the", "DET"), _tok("x")],
            [_tok("a", "DET"), _tok("x")],
            [_tok("x")],
            [_tok("xs", feats="Number=Plur")],
        ):
            paper, detail = english_form(toks)
            assert paper in {"definite", "bare_singular", "demonstrative", "other"}
            assert detail in {"", "indefinite", "possessive", "pronoun", "proper_name"}


# --- ZH form rules --------------------------------------------------------------------------


class TestChineseFormRules:
    def test_bare(self):
        assert chinese_form([_tok("在", "ADP"), _tok("柜子")]) == ("bare", "")

    def test_demonstrative(self):
        assert chinese_form([_tok("那个", "DET"), _tok("柜子")]) == ("demonstrative", "")

    def test_demonstrative_this(self):
        assert chinese_form([_tok("这", "DET"), _tok("条", "CL"), _tok("路")]) == (
            "demonstrative", ""
        )

    def test_numeral_classifier_detail(self):
        assert chinese_form([_tok("一", "NUM"), _tok("个", "CL"), _tok("房子")]) == (
            "other", "numeral_classifier"
        )

    def test_pronoun_detail(self):
        assert chinese_form([_tok("它", "PRON")]) == ("other", "pronoun")

    def test_proper_name_detail(self):
        assert chinese_form([_tok("女贞路", "PROPN")]) == ("other", "proper_name")

    def test_possessive_detail(self):
        assert chinese_form([_tok("他", "PRON"), _tok("的", "PART"), _tok("房子")]) == (
            "other", "possessive"
        )

    def test_output_space_is_frozen_vocab(self):
        for toks in (
            [_tok("柜子")],
            [_tok("那个", "DET"), _tok("柜子")],
            [_tok("一", "NUM"), _tok("个", "CL"), _tok("房子")],
        ):
            paper, detail = chinese_form(toks)
            assert paper in {"bare", "demonstrative", "other"}
            assert detail in {"", "numeral_classifier", "possessive", "pronoun", "proper_name"}


# --- DE proposal -------------------------------------------------------------------------------


class TestDeValidProposal:
    def test_contracted_includes(self):
        assert de_valid_proposal("contracted", "-", "-") == ("include", "")

    def test_uncontracted_pass(self):
        assert de_valid_proposal("uncontracted", "ART", "det") == ("include", "")

    def test_uncontracted_relative_pronoun_excludes(self):
        assert de_valid_proposal("uncontracted", "PRELS", "rc") == (
            "exclude", "not_target_pp"
        )

    def test_missing_metadata_uncertain(self):
        assert de_valid_proposal("uncontracted", "-", "-") == ("uncertain", "")


# --- artifact row --------------------------------------------------------------------------------


def _master_row(dp: str = "dp1", **overrides) -> dict[str, str]:
    row = {
        "datapoint_id": dp,
        "chapter": "1",
        "de_form": "contracted",
        "de_pp_surface": "im Testhaus",
        "de_head_lemma": "Testhaus",
        "de_sentence_text": "Er wohnt im Testhaus.",
        "en_context_text": "He lives in the test house.",
        "en_context_provenance": "anchor_window",
        "zh_context_text": "他住在试验房子里。",
        "zh_context_provenance": "anchor_window",
        "de_det_xpos": "-",
        "de_det_deprel": "-",
        "source_row_sha256": "c" * 64,
    }
    row.update(overrides)
    return row


class TestBuildDeterministicRow:
    def test_full_row_schema(self):
        en = SpanRecovery("aligned", span_text="the test house", span_tokens=[
            _tok("the", "DET"), _tok("test"), _tok("house")
        ])
        zh = SpanRecovery("aligned", span_text="试验房子", span_tokens=[_tok("试验"), _tok("房子")])
        row = build_deterministic_row(
            _master_row(), en, zh, ("definite", ""), ("bare", ""),
            pack_stratum="routine", method_params_id="eflomal2.0-m3-s1-intersect-v1",
        )
        assert tuple(row) == MACHINE_TUPLE_COLUMNS
        assert row["machine_annotation_model"] == DETERMINISTIC_METHOD_ID
        assert row["machine_en_paper_form"] == "definite"
        assert row["machine_zh_paper_form"] == "bare"
        assert row["machine_de_valid_proposal"] == "include"
        # statuses come from the recovery, forms only when aligned
        assert row["machine_en_status"] == "aligned"

    def test_not_aligned_side_is_blank(self):
        row = build_deterministic_row(
            _master_row(), None, None, None, None,
            pack_stratum="routine", method_params_id="x",
        )
        assert row["machine_en_status"] == "not_aligned"
        assert row["machine_en_counterpart"] == ""
        assert row["machine_zh_status"] == "not_aligned"

    def test_unresolved_side_blank_forms(self):
        zh = SpanRecovery("unresolved", reason="no_intersection_links")
        row = build_deterministic_row(
            _master_row(), None, zh, None, None,
            pack_stratum="routine", method_params_id="x",
        )
        assert row["machine_zh_status"] == "unresolved"
        assert row["machine_zh_paper_form"] == ""


# --- links parsing + symmetrization ------------------------------------------------


class TestLinksParsing:
    def test_parse_links_file_one_line_per_sentence(self, tmp_path: Path):
        p = tmp_path / "fwd.links"
        p.write_text("0-0 1-2 2-3\n\n3-1\n", encoding="utf-8")
        links = parse_links_file(p)
        assert links == [{(0, 0), (1, 2), (2, 3)}, set(), {(3, 1)}]

    def test_intersection_symmetrization(self):
        # both files are source-first indexed (eflomal normalizes the
        # reverse model's output to src-trg pairs)
        fwd = {(0, 0), (1, 2), (2, 3)}
        rev = {(0, 0), (2, 3), (1, 1)}
        assert intersection_links(fwd, rev) == {(0, 0), (2, 3)}


# --- reproducibility -------------------------------------------------------------------


class TestReproducibility:
    def test_recovery_is_pure_function_of_inputs(self, en_sentences):
        de = _en_layout(en_sentences)
        tgt = _en_layout(en_sentences)
        fwd, rev = {(7, 8)}, {(7, 8)}
        first = recover_target_span(range(6, 8), de, tgt, fwd, rev, lang="en")
        results = [
            recover_target_span(range(6, 8), de, tgt, fwd, rev, lang="en")
            for _ in range(5)
        ]
        for r in results:
            assert (r.status, r.span_text, r.reason) == (
                first.status, first.span_text, first.reason)

    def test_form_rules_are_pure(self):
        toks = [_tok("the", "DET"), _tok("house")]
        assert [english_form(toks) for _ in range(3)] == [english_form(toks)] * 3


# --- production path untouched ---------------------------------------------------------


class TestProductionPathUntouched:
    def test_production_builder_leaves_annotator_columns_blank(self):
        build = _load_script("build_annotation_csv")
        rows = build.build_csv_rows([_master_row()])
        assert len(rows) == 1
        for col in ANNOTATOR_COLUMNS:
            assert rows[0][col] == ""

    def test_deterministic_artifact_shares_no_annotator_columns(self):
        assert set(MACHINE_TUPLE_COLUMNS).isdisjoint(ANNOTATOR_COLUMNS)

    def test_pyproject_declares_eflomal_optional_only(self):
        text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "eflomal" in text
        main_deps = text.split("dependencies = [", 1)[1].split("]", 1)[0]
        assert "eflomal" not in main_deps


# --- eflomal integration (opt-in) ------------------------------------------------------


@pytest.mark.integration
class TestEflomalIntegration:
    def test_toy_train_and_parse_roundtrip(self, tmp_path: Path):
        eflomal = pytest.importorskip("eflomal")
        src = ["das Haus ist klein", "der Hund im Garten"]
        trg = ["the house is small", "the dog in the garden"]
        fwd_path = tmp_path / "f.links"
        rev_path = tmp_path / "r.links"
        aligner = eflomal.Aligner(model=3, n_samplers=1, n_iterations=(2, 2, 2))
        aligner.align(
            iter(src), iter(trg),
            links_filename_fwd=str(fwd_path),
            links_filename_rev=str(rev_path),
            quiet=True,
        )
        fwd = parse_links_file(fwd_path)
        rev = parse_links_file(rev_path)
        assert len(fwd) == 2 and len(rev) == 2
        for links in (*fwd, *rev):
            for s, t in links:
                assert 0 <= s < 4 and 0 <= t < 5
