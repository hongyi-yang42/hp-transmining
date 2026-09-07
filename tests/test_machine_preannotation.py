"""Synthetic-fixture tests for the machine pre-annotation pilot.

All fixture text is invented; no novel text. Covers the frozen
codebook, the constrained-span rule, the omitted vs not_aligned
distinction, batch id integrity, the deterministic seeded sampler, the
reviewer sheet design, dev-pack strata invariants (known-difficult rows
never promoted to routine), and that the production annotation builder
is untouched by the new path.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from hp_corpus.annotation_csv import ANNOTATOR_COLUMNS, CSV_COLUMNS
from hp_corpus.llm_client import parse_json_object
from hp_corpus.machine_preannotation import (
    ALIGNMENT_STATUSES,
    ARTIFACT_STATUS_VALUES,
    DE_PROPOSALS,
    DEV_PACK_STRATA,
    EN_DETAIL_FORMS,
    EN_PAPER_FORMS,
    HUMAN_COLUMNS,
    MACHINE_TUPLE_COLUMNS,
    REVIEW_SHEET_COLUMNS,
    ZH_DETAIL_FORMS,
    ZH_PAPER_FORMS,
    aggregate_outcomes,
    build_machine_row,
    build_reviewer_sheet,
    build_user_prompt,
    project_source_cells,
    prompt_sha256,
    sample_validation_pack,
    select_dev_pack,
    validate_machine_response,
    validate_machine_responses,
    validate_reviewer_sheet,
)
from hp_corpus.machine_preannotation import (
    SYSTEM_PROMPT as SYSTEM_PROMPT_TEXT,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _master_row(dp: str, **overrides) -> dict[str, str]:
    row = {
        "datapoint_id": dp,
        "chapter": "3",
        "de_form": "contracted",
        "de_pp_surface": "im Testschrank",
        "de_head_lemma": "Testschrank",
        "de_sentence_text": "Er sah etwas im Testschrank.",
        "en_context_ids": "syn_en_s001|syn_en_s002",
        "en_context_text": "Prev line. He saw something in the test cabinet. Next line.",
        "en_context_provenance": "anchor_window",
        "zh_context_ids": "syn_zh_s001|syn_zh_s002",
        "zh_context_text": "上一句。他在试验柜里看到了什么东西。下一句。",
        "zh_context_provenance": "anchor_window",
        "en_alignment_confidence": "0.95",
        "zh_alignment_confidence": "0.91",
        "source_row_sha256": "b" * 64,
    }
    row.update(overrides)
    return row


def _response(dp: str, **overrides) -> dict:
    resp = {
        "id": dp,
        "de_valid_proposal": "include",
        "de_exclusion_reason": "",
        "en_counterpart": "the test cabinet",
        "en_paper_form": "definite",
        "en_detail_form": "",
        "en_status": "aligned",
        "zh_counterpart": "试验柜",
        "zh_paper_form": "bare",
        "zh_detail_form": "",
        "zh_status": "aligned",
    }
    resp.update(overrides)
    return resp


# --- frozen vocabularies --------------------------------------------------------


class TestFrozenCodebook:
    def test_en_paper_forms_frozen(self):
        assert EN_PAPER_FORMS == frozenset(
            {"definite", "bare_singular", "demonstrative", "other"}
        )

    def test_zh_paper_forms_frozen(self):
        assert ZH_PAPER_FORMS == frozenset({"bare", "demonstrative", "other"})

    def test_no_theoretical_labels_anywhere(self):
        banned = {"weak", "strong", "anaphoric", "bridging", "familiar", "salient"}
        for vocab in (EN_PAPER_FORMS, ZH_PAPER_FORMS, EN_DETAIL_FORMS, ZH_DETAIL_FORMS):
            assert not (vocab & banned)

    def test_paper_and_detail_vocabularies_are_disjoint(self):
        assert not (EN_PAPER_FORMS & EN_DETAIL_FORMS)
        assert not (ZH_PAPER_FORMS & ZH_DETAIL_FORMS)

    def test_statuses_and_proposals(self):
        assert ALIGNMENT_STATUSES == frozenset({"aligned", "not_aligned", "uncertain"})
        assert ARTIFACT_STATUS_VALUES == ALIGNMENT_STATUSES | {"invalid"}
        assert DE_PROPOSALS == frozenset({"include", "exclude", "uncertain"})

    def test_paper_form_may_not_carry_a_detail_value(self):
        row = _master_row("dp1")
        resp = _response("dp1", en_paper_form="possessive")
        assert "EN_PAPER_FORM_INVALID" in validate_machine_response(resp, row)

    def test_detail_form_may_not_carry_a_paper_value(self):
        row = _master_row("dp1")
        resp = _response("dp1", en_detail_form="definite")
        assert "EN_DETAIL_FORM_INVALID" in validate_machine_response(resp, row)

    def test_invented_paper_category_rejected(self):
        row = _master_row("dp1")
        resp = _response("dp1", zh_paper_form="classifier_bare")
        assert "ZH_PAPER_FORM_INVALID" in validate_machine_response(resp, row)


# --- constrained counterpart extraction ------------------------------------------


class TestConstrainedSpans:
    def test_exact_substring_span_accepted(self):
        row = _master_row("dp1")
        assert validate_machine_response(_response("dp1"), row) == []

    def test_hallucinated_span_rejected(self):
        row = _master_row("dp1")
        # a paraphrase the model invented instead of copying
        resp = _response("dp1", en_counterpart="the testing cupboard")
        assert "EN_SPAN_NOT_SUBSTRING" in validate_machine_response(resp, row)

    def test_normalized_span_rejected(self):
        row = _master_row("dp1")
        # whitespace normalization is already a rewrite — reject
        resp = _response("dp1", en_counterpart="the test  cabinet")
        assert "EN_SPAN_NOT_SUBSTRING" in validate_machine_response(resp, row)

    def test_zh_span_must_come_from_zh_context(self):
        row = _master_row("dp1")
        resp = _response("dp1", zh_counterpart="柜子")
        assert "ZH_SPAN_NOT_SUBSTRING" in validate_machine_response(resp, row)

    def test_span_with_blank_context_rejected(self):
        row = _master_row("dp1", zh_context_text="", zh_context_provenance="neighbor_fallback")
        resp = _response("dp1")
        assert "ZH_SPAN_NOT_SUBSTRING" in validate_machine_response(resp, row)

    def test_span_requires_paper_form(self):
        row = _master_row("dp1")
        resp = _response("dp1", en_paper_form="")
        assert "EN_SPAN_WITHOUT_PAPER_FORM" in validate_machine_response(resp, row)


# --- omitted vs not_aligned -------------------------------------------------------


class TestOmissionVsRetrievalFailure:
    def test_true_omission_is_aligned_with_other_and_omitted(self):
        row = _master_row("dp1")
        resp = _response(
            "dp1",
            zh_counterpart="",
            zh_paper_form="other",
            zh_detail_form="omitted",
            zh_status="aligned",
        )
        assert validate_machine_response(resp, row) == []

    def test_omission_with_span_rejected(self):
        row = _master_row("dp1")
        resp = _response("dp1", en_paper_form="other", en_detail_form="omitted")
        assert "EN_OMITTED_WITH_SPAN" in validate_machine_response(resp, row)

    def test_omission_must_map_to_paper_other(self):
        row = _master_row("dp1")
        resp = _response("dp1", zh_counterpart="", zh_detail_form="omitted", zh_status="aligned")
        assert "ZH_OMITTED_PAPER_NOT_OTHER" in validate_machine_response(resp, row)
        resp2 = _response(
            "dp1", zh_counterpart="", zh_paper_form="demonstrative",
            zh_detail_form="omitted", zh_status="aligned",
        )
        assert "ZH_OMITTED_PAPER_NOT_OTHER" in validate_machine_response(resp2, row)

    def test_not_aligned_must_be_fully_blank_on_that_side(self):
        row = _master_row("dp1")
        resp = _response(
            "dp1",
            zh_counterpart="",
            zh_paper_form="",
            zh_detail_form="",
            zh_status="not_aligned",
        )
        assert validate_machine_response(resp, row) == []
        resp_bad = _response(
            "dp1", zh_counterpart="", zh_paper_form="other", zh_status="not_aligned"
        )
        assert "ZH_NOT_ALIGNED_WITH_CONTENT" in validate_machine_response(resp_bad, row)

    def test_aligned_without_span_or_omission_rejected(self):
        row = _master_row("dp1")
        resp = _response("dp1", en_counterpart="", en_paper_form="", en_status="aligned")
        assert "EN_ALIGNED_WITHOUT_SPAN_OR_OMISSION" in validate_machine_response(resp, row)

    def test_uncertain_must_not_carry_forms(self):
        row = _master_row("dp1")
        resp = _response("dp1", zh_status="uncertain")
        assert "ZH_UNCERTAIN_WITH_FORM" in validate_machine_response(resp, row)


# --- batch integrity ---------------------------------------------------------------


class TestBatchIntegrity:
    def _sources(self):
        return {r["datapoint_id"]: r for r in (_master_row("dp1"), _master_row("dp2"))}

    def test_one_response_per_row_round_trip(self):
        per_row, batch = validate_machine_responses(
            [_response("dp1"), _response("dp2")], self._sources()
        )
        assert batch == []
        assert per_row == {"dp1": [], "dp2": []}

    def test_unknown_response_id_fails(self):
        sources = {"dp1": _master_row("dp1")}
        per_row, batch = validate_machine_responses([_response("dp2")], sources)
        assert "RESPONSE_ID_UNKNOWN" in batch
        assert per_row == {}

    def test_response_id_does_not_round_trip(self):
        row = _master_row("dp1")
        assert "ID_MISMATCH" in validate_machine_response(_response("dp2"), row)

    def test_duplicate_response_id_fails(self):
        per_row, batch = validate_machine_responses(
            [_response("dp1"), _response("dp1")], self._sources()
        )
        assert "DUPLICATE_RESPONSE_ID" in batch

    def test_missing_response_row_fails(self):
        per_row, batch = validate_machine_responses([_response("dp1")], self._sources())
        assert "RESPONSE_ROW_MISSING" in batch

    def test_unexpected_keys_rejected(self):
        row = _master_row("dp1")
        resp = _response("dp1", hallucinated_key="x")
        assert "UNEXPECTED_RESPONSE_KEY" in validate_machine_response(resp, row)

    def test_de_proposal_coupling(self):
        row = _master_row("dp1")
        assert "DE_PROPOSAL_INVALID" in validate_machine_response(
            _response("dp1", de_valid_proposal="maybe"), row
        )
        assert "DE_EXCLUSION_REASON_INVALID" in validate_machine_response(
            _response("dp1", de_valid_proposal="exclude"), row
        )
        assert "DE_EXCLUSION_REASON_INVALID" in validate_machine_response(
            _response("dp1", de_valid_proposal="exclude", de_exclusion_reason="vibes"), row
        )
        assert validate_machine_response(
            _response("dp1", de_valid_proposal="exclude", de_exclusion_reason="not_target_pp"),
            row,
        ) == []
        assert "DE_REASON_WITHOUT_EXCLUDE" in validate_machine_response(
            _response("dp1", de_exclusion_reason="duplicate"), row
        )


# --- artifact rows ------------------------------------------------------------------


class TestArtifactRows:
    def test_valid_response_row_carries_all_columns(self):
        row = build_machine_row(
            _master_row("dp1"), _response("dp1"), model="test-model",
            prompt_version="mpa-v1", pack_stratum="routine",
        )
        assert tuple(row) == MACHINE_TUPLE_COLUMNS
        assert row["machine_en_status"] == "aligned"
        assert row["machine_zh_paper_form"] == "bare"
        assert row["machine_annotation_model"] == "test-model"
        assert row["machine_annotation_prompt_version"] == "mpa-v1"
        assert row["pack_stratum"] == "routine"

    def test_failed_response_row_is_invalid_not_dropped(self):
        row = build_machine_row(
            _master_row("dp1"), None, model="test-model",
            prompt_version="mpa-v1", pack_stratum="routine",
        )
        assert row["machine_en_status"] == "invalid"
        assert row["machine_zh_status"] == "invalid"
        assert row["machine_de_valid_proposal"] == "uncertain"
        assert row["machine_en_counterpart"] == ""
        # the source cells stay bound to the master
        assert row["id"] == "dp1"
        assert row["row_hash"] == "b" * 64

    def test_source_projection_is_excel_safe_and_complete(self):
        row = project_source_cells(
            _master_row("dp1", de_sentence_text="-- dialog line")
        )
        assert row["german_sentence"].startswith(" ")
        assert row["german_sentence"].strip().startswith("--")
        assert set(row) == set(CSV_COLUMNS) - set(ANNOTATOR_COLUMNS)


# --- dev pack selection --------------------------------------------------------------


def _pack_master():
    rows = []
    # routine candidates: anchor_window both sides, high confidence
    for i in range(8):
        rows.append(_master_row(f"dp_r_{i:02d}", de_form="contracted"))
    for i in range(8):
        rows.append(_master_row(f"dp_u_{i:02d}", de_form="uncontracted"))
    # demonstrative-cue candidates
    for i in range(4):
        rows.append(
            _master_row(
                f"dp_c_{i:02d}",
                en_context_text="Look at that cabinet over there. He nodded.",
            )
        )
    for i in range(4):
        rows.append(
            _master_row(
                f"dp_z_{i:02d}",
                zh_context_text="他看着那个柜子，点了点头。随后一切归于平静。",
            )
        )
    # widened-context rows
    for i in range(4):
        rows.append(
            _master_row(
                f"dp_w_{i:02d}",
                en_context_provenance="merge_widened",
                zh_context_provenance="heuristic_widened",
            )
        )
    # companion-difficult rows: defer provenance or low confidence
    for i in range(3):
        rows.append(_master_row(f"dp_d_{i:02d}", zh_context_provenance="manual_review"))
    for i in range(2):
        rows.append(_master_row(f"dp_l_{i:02d}", zh_alignment_confidence="0.20"))
    # regression candidates
    for i in range(3):
        rows.append(_master_row(f"dp_g_{i:02d}", en_context_provenance="neighbor_fallback"))
    return rows


class TestDevPackSelection:
    def test_strata_and_priorities(self):
        rows = _pack_master()
        regression_ids = {"dp_g_00", "dp_g_01", "dp_g_02"}
        chosen, composition = select_dev_pack(
            rows, regression_ids=regression_ids,
            forced_regression_ids=("dp_g_00", "dp_g_01"),
        )
        assert set(composition) == set(DEV_PACK_STRATA)
        assert sum(composition.values()) == len(chosen)
        assert chosen["dp_g_00"] == "regression_case"
        assert chosen["dp_g_01"] == "regression_case"
        assert chosen["dp_g_02"] == "regression_case"
        assert composition["regression_case"] == 3
        # deferred rows land in companion_difficult, never routine
        assert chosen["dp_d_00"] == "companion_difficult"
        assert chosen["dp_l_00"] == "companion_difficult"
        # widened rows land in widened_context
        assert chosen["dp_w_00"] == "widened_context"
        # cue rows land in demonstrative_cue
        assert chosen["dp_c_00"] == "demonstrative_cue"
        assert chosen["dp_z_00"] == "demonstrative_cue"
        # routine rows balance the two German forms
        routine_rows = {dp for dp, s in chosen.items() if s == "routine"}
        assert 0 < len(routine_rows) <= 10
        by_form = {r["datapoint_id"]: r["de_form"] for r in rows}
        forms = {by_form[dp] for dp in routine_rows}
        assert forms == {"contracted", "uncontracted"}

    def test_deferred_rows_never_routine(self):
        rows = _pack_master()
        by_id = {r["datapoint_id"]: r for r in rows}
        chosen, _ = select_dev_pack(rows, regression_ids={"dp_g_00"})
        for dp, stratum in chosen.items():
            if stratum != "routine":
                continue
            r = by_id[dp]
            assert r["en_context_provenance"] == "anchor_window"
            assert r["zh_context_provenance"] == "anchor_window"
            assert float(r["en_alignment_confidence"]) >= 0.40
            assert float(r["zh_alignment_confidence"]) >= 0.40

    def test_selection_is_deterministic(self):
        rows = _pack_master()
        first = select_dev_pack(rows, regression_ids={"dp_g_00"})
        second = select_dev_pack(rows, regression_ids={"dp_g_00"})
        assert first == second

    def test_pack_size_default_target(self):
        # build enough synthetic rows to saturate every stratum
        rows = _pack_master() + [
            _master_row(f"dp_w_{i:02d}", en_context_provenance="merge_widened")
            for i in range(4, 12)
        ] + [
            _master_row(f"dp_x_{i:03d}", de_form="contracted") for i in range(20)
        ] + [
            _master_row(f"dp_y_{i:03d}", de_form="uncontracted") for i in range(20)
        ]
        chosen, composition = select_dev_pack(rows, regression_ids=set())
        total = sum(composition.values())
        assert 30 <= total <= 40
        assert composition["routine"] == 10
        assert composition["companion_difficult"] == 6
        assert composition["widened_context"] == 9


# --- seeded sampler --------------------------------------------------------------------


class TestSeededSampler:
    IDS = [f"dp_{i:03d}" for i in range(100)]

    def test_deterministic_for_same_seed_and_input(self):
        a = sample_validation_pack(self.IDS, n=50, seed=20260907, source_sha256="h" * 64)
        b = sample_validation_pack(
            list(reversed(self.IDS)), n=50, seed=20260907, source_sha256="h" * 64
        )
        assert a == b
        assert a["datapoint_ids"] == sorted(a["datapoint_ids"])
        assert len(set(a["datapoint_ids"])) == 50

    def test_different_seed_different_sample(self):
        a = sample_validation_pack(self.IDS, n=50, seed=1, source_sha256="h" * 64)
        b = sample_validation_pack(self.IDS, n=50, seed=2, source_sha256="h" * 64)
        assert a["datapoint_ids"] != b["datapoint_ids"]

    def test_manifest_records_seed_and_source_hash(self):
        m = sample_validation_pack(self.IDS, n=10, seed=7, source_sha256="k" * 64)
        assert m["seed"] == 7
        assert m["n"] == 10
        assert m["source_sha256"] == "k" * 64
        assert m["schema_version"] == "machine-tuple-validation-sample-v1"

    def test_out_of_range_fails_closed(self):
        with pytest.raises(ValueError):
            sample_validation_pack([], n=5, seed=1, source_sha256="k" * 64)
        with pytest.raises(ValueError):
            sample_validation_pack(self.IDS, n=0, seed=1, source_sha256="k" * 64)
        with pytest.raises(ValueError):
            sample_validation_pack(self.IDS, n=101, seed=1, source_sha256="k" * 64)


# --- reviewer sheet ----------------------------------------------------------------------


class TestReviewerSheet:
    def _machine_rows(self):
        return [
            build_machine_row(
                _master_row("dp1"), _response("dp1"), model="m",
                prompt_version="mpa-v1", pack_stratum="routine",
            )
        ]

    def test_machine_and_human_columns_separate_and_blank(self):
        sheet = build_reviewer_sheet(self._machine_rows())
        assert tuple(sheet[0]) == REVIEW_SHEET_COLUMNS
        assert set(HUMAN_COLUMNS).isdisjoint(MACHINE_TUPLE_COLUMNS)
        for col in HUMAN_COLUMNS:
            assert sheet[0][col] == ""
        # machine cells survive verbatim
        assert sheet[0]["machine_en_paper_form"] == "definite"

    def test_valid_checks_pass(self):
        sheet = build_reviewer_sheet(self._machine_rows())
        sheet[0].update(
            {
                "human_de_valid_check": "correct",
                "human_en_counterpart_check": "correct",
                "human_en_form_check": "correct",
                "human_zh_counterpart_check": "incorrect",
                "human_zh_counterpart_correction": "试验柜里",
                "human_zh_form_check": "uncertain",
            }
        )
        assert validate_reviewer_sheet(sheet) == []

    def test_invalid_check_value_rejected(self):
        sheet = build_reviewer_sheet(self._machine_rows())
        sheet[0]["human_de_valid_check"] = "mostly-fine"
        assert "HUMAN_CHECK_INVALID" in validate_reviewer_sheet(sheet)

    def test_incorrect_requires_correction(self):
        sheet = build_reviewer_sheet(self._machine_rows())
        sheet[0]["human_de_valid_check"] = "incorrect"
        assert "DE_CHECK_INCORRECT_WITHOUT_CORRECTION" in validate_reviewer_sheet(sheet)
        sheet[0]["human_de_valid_correction"] = "exclude"
        assert "DE_CHECK_INCORRECT_WITHOUT_CORRECTION" not in validate_reviewer_sheet(sheet)

    def test_correction_without_incorrect_rejected(self):
        sheet = build_reviewer_sheet(self._machine_rows())
        sheet[0]["human_en_counterpart_check"] = "correct"
        sheet[0]["human_en_counterpart_correction"] = "the test cabinet"
        errors = validate_reviewer_sheet(sheet)
        assert "EN_SPAN_CORRECTION_WITHOUT_INCORRECT" in errors

    def test_correction_span_must_be_substring(self):
        sheet = build_reviewer_sheet(self._machine_rows())
        sheet[0]["human_en_counterpart_check"] = "incorrect"
        sheet[0]["human_en_counterpart_correction"] = "a paraphrase not present"
        assert "EN_SPAN_CORRECTION_NOT_SUBSTRING" in validate_reviewer_sheet(sheet)

    def test_paper_correction_enum(self):
        sheet = build_reviewer_sheet(self._machine_rows())
        sheet[0]["human_zh_form_check"] = "incorrect"
        sheet[0]["human_zh_paper_form_correction"] = "weak"
        errors = validate_reviewer_sheet(sheet)
        assert "ZH_PAPER_CORRECTION_INVALID" in errors

    def test_duplicate_id_rejected(self):
        sheet = build_reviewer_sheet(self._machine_rows() + self._machine_rows())
        assert "DUPLICATE_ID" in validate_reviewer_sheet(sheet)


# --- prompt --------------------------------------------------------------------------------


class TestPrompt:
    def test_prompt_mentions_every_constraining_rule(self):
        # whitespace-normalized so prompt re-wrapping cannot weaken the check
        normalized = " ".join(SYSTEM_PROMPT_TEXT.split())
        for phrase in (
            "ONLY within the supplied context",
            "exact substring",
            "frozen vocabulary",
            '"other"',
            "detail_form",
            "not_aligned",
            '"omitted"',
            "weak/strong",
            "ONE JSON object",
        ):
            assert phrase in normalized

    def test_user_prompt_contains_row_and_schema(self):
        prompt = build_user_prompt(_master_row("dp1"))
        assert '"id": "dp1"' in prompt
        assert "german_pp" in prompt
        assert "en_paper_form" in prompt
        assert "im Testschrank" in prompt

    def test_prompt_version_and_hash_stable(self):
        import hp_corpus.machine_preannotation as m

        assert m.PROMPT_VERSION == "mpa-v1"
        h1 = prompt_sha256()
        assert len(h1) == 64 and h1 == m.prompt_sha256()


# --- JSON parsing ----------------------------------------------------------------------------


class TestParseJsonObject:
    def test_plain_object(self):
        assert parse_json_object('{"id": "dp1"}') == {"id": "dp1"}

    def test_fenced_object(self):
        text = '```json\n{"id": "dp1"}\n```'
        assert parse_json_object(text) == {"id": "dp1"}

    def test_array_rejected(self):
        assert parse_json_object('[{"id": "dp1"}]') is None

    def test_prose_rejected(self):
        assert parse_json_object("Here is my answer: ...") is None

    def test_empty_rejected(self):
        assert parse_json_object("") is None


# --- production path untouched ------------------------------------------------------------------


class TestProductionPathUntouched:
    def test_production_builder_leaves_annotator_columns_blank(self):
        build = _load_script("build_annotation_csv")
        rows = build.build_csv_rows([_master_row("dp1")])
        assert len(rows) == 1
        for col in ANNOTATOR_COLUMNS:
            assert rows[0][col] == ""
        assert tuple(rows[0]) == CSV_COLUMNS

    def test_machine_artifact_shares_no_annotator_columns(self):
        # the machine artifact can never be mistaken for (or merged
        # into) a human-filled annotation file
        assert set(MACHINE_TUPLE_COLUMNS).isdisjoint(ANNOTATOR_COLUMNS)

    def test_aggregates_are_counts_only(self):
        rows = [
            build_machine_row(
                _master_row("dp1"), _response("dp1"), model="m",
                prompt_version="mpa-v1", pack_stratum="routine",
            ),
            build_machine_row(
                _master_row("dp2"), None, model="m",
                prompt_version="mpa-v1", pack_stratum="routine",
            ),
        ]
        agg = aggregate_outcomes(rows)
        assert agg["rows"] == {"total": 2}
        assert agg["en_status"] == {"aligned": 1, "invalid": 1}
        assert agg["zh_status"] == {"aligned": 1, "invalid": 1}
        assert agg["en_paper_form"] == {"definite": 1, "(blank)": 1}
        for sub in agg.values():
            assert all(isinstance(v, int) for v in sub.values())


# --- script end-to-end (no network) ---------------------------------------------------------------


class TestScriptDevPack:
    def test_dev_pack_command(self, tmp_path: Path):
        script = _load_script("run_machine_preannotation")
        master_path = tmp_path / "master.tsv"
        import csv as csv_mod

        with open(master_path, "w", encoding="utf-8", newline="") as f:
            w = csv_mod.DictWriter(f, fieldnames=list(_master_row("dp1")), delimiter="\t")
            w.writeheader()
            rows = _pack_master()
            # include the two real forced regression ids as synthetic rows
            rows.append(_master_row(script.FORCED_REGRESSION_IDS[0]))
            rows.append(_master_row(script.FORCED_REGRESSION_IDS[1]))
            w.writerows(rows)

        regression_json = tmp_path / "reg.json"
        regression_json.write_text(
            json.dumps({"cases": {"dp_g_00": {"zh": ["x"]}}}), encoding="utf-8"
        )
        out = tmp_path / "pack_manifest.json"
        rc = script.main(
            [
                "dev-pack",
                "--master-tsv", str(master_path),
                "--regression-json", str(regression_json),
                "--output", str(out),
            ]
        )
        assert rc == 0
        manifest = json.loads(out.read_text(encoding="utf-8"))
        ids = [r["datapoint_id"] for r in manifest["rows"]]
        assert script.FORCED_REGRESSION_IDS[0] in ids
        assert script.FORCED_REGRESSION_IDS[1] in ids
        assert len(ids) == len(set(ids))
        assert sum(manifest["composition"].values()) == len(ids)
        assert manifest["schema_version"] == "machine-tuple-dev-pack-v1"
