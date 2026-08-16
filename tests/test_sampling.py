"""Synthetic-fixture tests for the formal German eligible-pool rule.

Covers the pure :func:`hp_corpus.sampling.build_eligible_pool` function
and the ``build_eligible_pool`` CLI (review-overlay contract, fail-closed
rules). All fixtures use invented lemma strings; no novel text.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from hp_corpus.sampling import (
    EARLY_CHAPTERS,
    FULL_NOVEL_CHAPTERS,
    LATE_CHAPTERS,
    POOL_REASONS,
    IncompleteReviewError,
    Occurrence,
    OccurrenceIdentityConflictError,
    StructuralMetadataMissingError,
    build_eligible_pool,
    canonical_preposition,
    collapse_occurrences,
    in_paired_inventory,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _occ(
    datapoint_id: str,
    *,
    chapter: int,
    form: str,
    canonical_prep: str,
    machine_lemma: str,
    corrected_lemma: str = "",
    decision: str = "include",
    inventory_eligible: bool = True,
    det_xpos: str = "ART",
    det_deprel: str = "det",
    source_segment_id: str | None = None,
    parse_block_id: str | None = None,
    pp_token_start: int = 1,
    pp_token_end: int = 2,
) -> Occurrence:
    """Build an Occurrence with a distinct identity by default
    (identity components derive from the datapoint id unless given)."""
    return Occurrence(
        datapoint_id=datapoint_id,
        chapter=chapter,
        form=form,
        canonical_prep=canonical_prep,
        machine_head_lemma=machine_lemma,
        decision=decision,
        corrected_head_lemma=corrected_lemma,
        inventory_eligible=inventory_eligible,
        source_hash="0" * 64,
        source_segment_id=source_segment_id or f"{datapoint_id}_seg",
        parse_block_id=parse_block_id or f"{datapoint_id}_blk",
        pp_token_start=pp_token_start,
        pp_token_end=pp_token_end,
        det_xpos=det_xpos,
        det_deprel=det_deprel,
    )


# --------------------------------------------------------------------- pair rule


def test_uncontracted_all_chapters_eligible_after_review_include() -> None:
    occs = [
        _occ("u1", chapter=1, form="uncontracted", canonical_prep="in", machine_lemma="alpha"),
        _occ("u2", chapter=8, form="uncontracted", canonical_prep="an", machine_lemma="beta"),
        _occ("u3", chapter=17, form="uncontracted", canonical_prep="auf", machine_lemma="gamma"),
    ]
    result = build_eligible_pool(occs)
    eligible = {r.occurrence.datapoint_id for r in result.rows if r.eligible}
    assert eligible == {"u1", "u2", "u3"}
    for r in result.rows:
        assert r.pool_reason == "uncontracted_all_chapters"
    ep = result.summary["eligible_pool"]
    assert ep["uncontracted_all_chapters"] == 3
    assert ep["eligible_total"] == 3


def test_contracted_ch1_3_eligible_after_review_include() -> None:
    occs = [
        _occ("c1", chapter=1, form="contracted", canonical_prep="in", machine_lemma="alpha"),
        _occ("c2", chapter=2, form="contracted", canonical_prep="an", machine_lemma="beta"),
        _occ("c3", chapter=3, form="contracted", canonical_prep="auf", machine_lemma="gamma"),
    ]
    result = build_eligible_pool(occs)
    for r in result.rows:
        assert r.eligible is True
        assert r.pool_reason == "contracted_ch1_3"
    assert result.summary["eligible_pool"]["contracted_ch1_3"] == 3


def test_exact_prep_noun_pair_makes_ch4_17_contracted_eligible() -> None:
    """Paper-literal pair rule: a Ch.5 contracted occurrence whose
    (canonical_prep, head_lemma) matches a reviewed-include uncontracted
    occurrence is eligible."""
    occs = [
        _occ("u1", chapter=1, form="uncontracted", canonical_prep="in", machine_lemma="alpha"),
        _occ("c_pair", chapter=5, form="contracted", canonical_prep="in", machine_lemma="alpha"),
        _occ("c_none", chapter=6, form="contracted", canonical_prep="an", machine_lemma="beta"),
    ]
    result = build_eligible_pool(occs)
    by_id = {r.occurrence.datapoint_id: r for r in result.rows}
    assert by_id["c_pair"].eligible is True
    assert by_id["c_pair"].pool_reason == "contracted_ch4_17_pair_matched"
    assert by_id["c_none"].eligible is False
    assert by_id["c_none"].pool_reason == "contracted_ch4_17_no_uncontracted_counterpart"


def test_same_noun_different_prep_not_eligible() -> None:
    occs = [
        _occ("u1", chapter=1, form="uncontracted", canonical_prep="in", machine_lemma="alpha"),
        _occ("c1", chapter=7, form="contracted", canonical_prep="an", machine_lemma="alpha"),
    ]
    result = build_eligible_pool(occs)
    by_id = {r.occurrence.datapoint_id: r for r in result.rows}
    assert by_id["c1"].eligible is False
    assert by_id["c1"].pool_reason == "contracted_ch4_17_no_uncontracted_counterpart"


def test_same_prep_different_noun_not_eligible() -> None:
    occs = [
        _occ("u1", chapter=1, form="uncontracted", canonical_prep="in", machine_lemma="alpha"),
        _occ("c1", chapter=7, form="contracted", canonical_prep="in", machine_lemma="delta"),
    ]
    result = build_eligible_pool(occs)
    by_id = {r.occurrence.datapoint_id: r for r in result.rows}
    assert by_id["c1"].eligible is False
    assert by_id["c1"].pool_reason == "contracted_ch4_17_no_uncontracted_counterpart"


def test_pair_matched_against_reviewed_include_only() -> None:
    """A pair that exists only among review-EXCLUDED uncontracted rows
    does not make the contracted row eligible."""
    occs = [
        _occ(
            "u_excl",
            chapter=1,
            form="uncontracted",
            canonical_prep="an",
            machine_lemma="alpha",
            decision="exclude",
        ),
        _occ("c1", chapter=9, form="contracted", canonical_prep="an", machine_lemma="alpha"),
    ]
    result = build_eligible_pool(occs)
    by_id = {r.occurrence.datapoint_id: r for r in result.rows}
    assert by_id["u_excl"].pool_reason == "excluded_by_german_review"
    assert by_id["c1"].eligible is False


# --------------------------------------------------------------------- corrected lemma


def test_corrected_lemma_takes_effect() -> None:
    """A reviewer correction changes the matching lemma in both
    directions: enabling a pair the machine lemma missed, and breaking
    one the machine lemma would have made."""
    occs = [
        # Machine lemma "delta" corrected to "alpha" → now pairs with U.
        _occ("u1", chapter=1, form="uncontracted", canonical_prep="in", machine_lemma="alpha"),
        _occ(
            "c_fix",
            chapter=5,
            form="contracted",
            canonical_prep="in",
            machine_lemma="delta",
            corrected_lemma="alpha",
        ),
        # Machine lemma "alpha" corrected to "beta" → pair broken.
        _occ(
            "c_break",
            chapter=6,
            form="contracted",
            canonical_prep="in",
            machine_lemma="alpha",
            corrected_lemma="beta",
        ),
    ]
    result = build_eligible_pool(occs)
    by_id = {r.occurrence.datapoint_id: r for r in result.rows}
    assert by_id["c_fix"].eligible is True
    assert by_id["c_break"].eligible is False


def test_blank_corrected_lemma_means_machine_lemma() -> None:
    """Blank correction is the explicit "machine lemma stands" default —
    it is never a correction of its own."""
    occs = [
        _occ("u1", chapter=1, form="uncontracted", canonical_prep="in", machine_lemma="alpha"),
        _occ(
            "c1",
            chapter=5,
            form="contracted",
            canonical_prep="in",
            machine_lemma="alpha",
            corrected_lemma="",
        ),
    ]
    result = build_eligible_pool(occs)
    by_id = {r.occurrence.datapoint_id: r for r in result.rows}
    assert by_id["c1"].eligible is True
    assert by_id["c1"].head_lemma == "alpha"


# --------------------------------------------------------------------- automatic gates


def test_outside_paired_inventory_excluded() -> None:
    occs = [
        _occ(
            "um1",
            chapter=5,
            form="contracted",
            canonical_prep="um",
            machine_lemma="x",
            inventory_eligible=False,
        ),
    ]
    result = build_eligible_pool(occs)
    r = result.rows[0]
    assert r.eligible is False
    assert r.pool_reason == "outside_paired_inventory"
    assert result.summary["automatically_excluded"]["outside_paired_inventory"] == 1


def test_structural_gate_wrong_xpos_excluded() -> None:
    occs = [
        _occ(
            "u1",
            chapter=4,
            form="uncontracted",
            canonical_prep="in",
            machine_lemma="alpha",
            det_xpos="PRELS",
        ),
    ]
    result = build_eligible_pool(occs)
    r = result.rows[0]
    assert r.eligible is False
    assert r.pool_reason == "failed_structural_gate"
    assert result.summary["automatically_excluded"]["failed_structural_gate"] == 1


def test_structural_gate_wrong_deprel_excluded() -> None:
    occs = [
        _occ(
            "u1",
            chapter=4,
            form="uncontracted",
            canonical_prep="in",
            machine_lemma="alpha",
            det_deprel="nsubj",
        ),
    ]
    result = build_eligible_pool(occs)
    assert result.rows[0].pool_reason == "failed_structural_gate"


@pytest.mark.parametrize("xpos,deprel", [("", "det"), ("ART", ""), ("-", "det"), ("ART", "-")])
def test_structural_gate_missing_metadata_fails_closed(xpos: str, deprel: str) -> None:
    """Absent structural metadata on an uncontracted row is a hard
    failure — the gate never passes by default."""
    occs = [
        _occ(
            "u1",
            chapter=4,
            form="uncontracted",
            canonical_prep="in",
            machine_lemma="alpha",
            det_xpos=xpos,
            det_deprel=deprel,
        ),
    ]
    with pytest.raises(StructuralMetadataMissingError):
        build_eligible_pool(occs)


def test_structural_gate_contracted_rows_exempt() -> None:
    """Contracted rows have no determiner — blank metadata never affects
    them."""
    occs = [
        _occ(
            "c1",
            chapter=2,
            form="contracted",
            canonical_prep="in",
            machine_lemma="alpha",
            det_xpos="",
            det_deprel="",
        ),
        _occ(
            "c_late",
            chapter=9,
            form="contracted",
            canonical_prep="in",
            machine_lemma="alpha",
            det_xpos="-",
            det_deprel="-",
        ),
        _occ("u1", chapter=1, form="uncontracted", canonical_prep="in", machine_lemma="alpha"),
    ]
    result = build_eligible_pool(occs)
    by_id = {r.occurrence.datapoint_id: r for r in result.rows}
    assert by_id["c1"].pool_reason == "contracted_ch1_3"
    assert by_id["c_late"].pool_reason == "contracted_ch4_17_pair_matched"


def test_review_exclude_not_eligible() -> None:
    occs = [
        _occ(
            "u1",
            chapter=1,
            form="uncontracted",
            canonical_prep="in",
            machine_lemma="alpha",
            decision="exclude",
        ),
    ]
    result = build_eligible_pool(occs)
    r = result.rows[0]
    assert r.eligible is False
    assert r.pool_reason == "excluded_by_german_review"
    assert result.summary["human_review"] == {"included": 0, "excluded": 1}


# --------------------------------------------------------------------- identity


def test_exact_duplicate_identity_collapses() -> None:
    dup = _occ("u1", chapter=1, form="uncontracted", canonical_prep="in", machine_lemma="alpha")
    occs = [
        dup,
        _occ("u2", chapter=2, form="uncontracted", canonical_prep="an", machine_lemma="beta"),
        _occ(
            "u1_copy",
            chapter=1,
            form="uncontracted",
            canonical_prep="in",
            machine_lemma="alpha",
            source_segment_id=dup.source_segment_id,
            parse_block_id=dup.parse_block_id,
            pp_token_start=dup.pp_token_start,
            pp_token_end=dup.pp_token_end,
        ),
    ]
    result = build_eligible_pool(occs)
    assert len(result.rows) == 2
    assert result.summary["duplicate_rows_collapsed"] == 1
    assert result.summary["extracted_total"] == 3
    assert result.summary["eligible_pool"]["eligible_total"] == 2


def test_identity_conflict_fails_closed() -> None:
    base = dict(
        chapter=1,
        form="uncontracted",
        canonical_prep="in",
        source_segment_id="seg_x",
        parse_block_id="seg_x#b001",
        pp_token_start=3,
        pp_token_end=5,
    )
    occs = [
        _occ("a", machine_lemma="alpha", **base),
        _occ("b", machine_lemma="delta", **base),  # same identity, different lemma
    ]
    with pytest.raises(OccurrenceIdentityConflictError):
        build_eligible_pool(occs)
    with pytest.raises(OccurrenceIdentityConflictError):
        collapse_occurrences(occs)


def test_collapse_preserves_first_occurrence_deterministically() -> None:
    base = dict(
        chapter=7,
        form="contracted",
        canonical_prep="in",
        source_segment_id="seg_y",
        parse_block_id="seg_y#b002",
        pp_token_start=1,
        pp_token_end=2,
    )
    first = _occ("first", machine_lemma="alpha", **base)
    second = _occ("second", machine_lemma="alpha", **base)
    collapsed, stats = collapse_occurrences([first, second])
    assert [o.datapoint_id for o in collapsed] == ["first"]
    assert stats == {"duplicate_rows_collapsed": 1, "unique_occurrences": 1}


# --------------------------------------------------------------------- summary hygiene


def test_summary_has_no_mode_projection_or_noun_only_keys() -> None:
    """The deleted concepts must not survive anywhere in the summary:
    no sampling modes, no projections, no noun-only counterfactual, no
    analysis-readiness flags."""
    occs = [
        _occ("u1", chapter=1, form="uncontracted", canonical_prep="in", machine_lemma="alpha"),
        _occ("c1", chapter=7, form="contracted", canonical_prep="an", machine_lemma="alpha"),
    ]
    summary = build_eligible_pool(occs).summary

    def _walk(d):
        for k, v in d.items():
            yield k
            if isinstance(v, dict):
                yield from _walk(v)

    keys = set(_walk(summary))
    banned_exact = {
        "mode_a",
        "mode_a_total",
        "mode_b",
        "mode_c",
        "projection",
        "projection_only",
        "all_include",
        "all_include_projection",
        "noun_only",
        "noun_only_counterfactual",
        "delta_vs_pair",
        "analysis_ready",
        "run_mode",
        "u",
        "c_early",
        "c_late",
        "c_late_pair_match",
        "u_lemma_count",
        "u_pair_count",
        "selected_total",
        "occurrence_total",
    }
    assert not (keys & banned_exact), keys & banned_exact
    # Stage abbreviations and mode/projection prefixes must not survive.
    for prefix in ("s0", "s1", "s2", "s3", "s5", "s6", "mode", "noun_only", "projection"):
        assert not any(k.startswith(prefix) for k in keys), prefix


def test_reason_vocabulary_is_plain() -> None:
    assert POOL_REASONS == frozenset(
        {
            "uncontracted_all_chapters",
            "contracted_ch1_3",
            "contracted_ch4_17_pair_matched",
            "contracted_ch4_17_no_uncontracted_counterpart",
            "outside_paired_inventory",
            "failed_structural_gate",
            "excluded_by_german_review",
        }
    )


def test_counts_conserve() -> None:
    occs = [
        _occ("u1", chapter=1, form="uncontracted", canonical_prep="in", machine_lemma="alpha"),
        _occ("u2", chapter=9, form="uncontracted", canonical_prep="an", machine_lemma="beta"),
        _occ(
            "um1",
            chapter=5,
            form="contracted",
            canonical_prep="um",
            machine_lemma="x",
            inventory_eligible=False,
        ),
        _occ(
            "bad",
            chapter=4,
            form="uncontracted",
            canonical_prep="auf",
            machine_lemma="gamma",
            det_xpos="PRELS",
        ),
        _occ(
            "excl",
            chapter=2,
            form="contracted",
            canonical_prep="in",
            machine_lemma="alpha",
            decision="exclude",
        ),
        _occ("early", chapter=3, form="contracted", canonical_prep="in", machine_lemma="alpha"),
        _occ("pair", chapter=8, form="contracted", canonical_prep="in", machine_lemma="alpha"),
        _occ("nopair", chapter=8, form="contracted", canonical_prep="auf", machine_lemma="alpha"),
    ]
    s = build_eligible_pool(occs).summary
    assert sum(s["by_reason"].values()) == s["extracted_total"] - s["duplicate_rows_collapsed"]
    ep = s["eligible_pool"]
    assert (
        ep["uncontracted_all_chapters"]
        + ep["contracted_ch1_3"]
        + ep["contracted_ch4_17_pair_matched"]
        == ep["eligible_total"]
    )
    # Whole-set conservation: every unique occurrence is accounted for.
    assert (
        ep["eligible_total"]
        + s["contracted_ch4_17_no_uncontracted_counterpart"]
        + s["automatically_excluded"]["outside_paired_inventory"]
        + s["automatically_excluded"]["failed_structural_gate"]
        + s["human_review"]["excluded"]
        == s["extracted_total"] - s["duplicate_rows_collapsed"]
    )
    assert s["human_review"]["included"] == 5  # u1, u2, early, pair, nopair


# --------------------------------------------------------------------- helpers / constants


def test_constants_chapter_ranges() -> None:
    assert list(EARLY_CHAPTERS) == [1, 2, 3]
    assert list(LATE_CHAPTERS) == list(range(4, 18))
    assert list(FULL_NOVEL_CHAPTERS) == list(range(1, 18))


def test_canonical_preposition_helper() -> None:
    assert canonical_preposition("im") == "in"
    assert canonical_preposition("in") == "in"
    assert canonical_preposition("zum") == "zu"
    assert canonical_preposition("xyz") == "xyz"


def test_in_paired_inventory_helper() -> None:
    assert in_paired_inventory("in")
    assert in_paired_inventory("an")
    assert not in_paired_inventory("um")
    assert not in_paired_inventory("vor")


# --------------------------------------------------------------------- CLI


def _write_extraction_tsv(path: Path, rows: list[dict[str, str]]) -> Path:
    from hp_corpus.german_extraction import TSV_FIELDS

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "-") for k in TSV_FIELDS}
            w.writerow(row)
    return path


def _ext_row(
    block_id: str,
    *,
    prep: str = "in",
    noun: str = "alpha",
    start: str = "1",
    end: str = "2",
    det_xpos: str = "ART",
    det_deprel: str = "det",
) -> dict[str, str]:
    return {
        "parse_block_id": block_id,
        "source_segment_id": block_id.split("#b")[0],
        "prep": prep,
        "det": "dem" if det_xpos != "-" else "-",
        "noun": noun,
        "prep_token_id": "1",
        "det_token_id": "2",
        "noun_token_id": "3",
        "pp_token_start": start,
        "pp_token_end": end,
        "pp_surface": "x",
        "det_xpos": det_xpos,
        "det_deprel": det_deprel,
    }


def _write_master_tsv(path: Path, datapoint_ids: list[str]) -> Path:
    from hp_corpus.step4 import ALL_TSV_COLUMNS

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ALL_TSV_COLUMNS), delimiter="\t", lineterminator="\n")
        w.writeheader()
        for dp in datapoint_ids:
            w.writerow(
                {
                    col: dp
                    if col == "datapoint_id"
                    else "a" * 64
                    if col == "source_row_sha256"
                    else ""
                    for col in ALL_TSV_COLUMNS
                }
            )
    return path


def _write_review_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    """A returned annotator CSV (full column set; the eligible-pool CLI
    reads the German columns + id + row_hash from it)."""
    from hp_corpus.annotation_csv import CSV_COLUMNS

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in CSV_COLUMNS})
    return path


_DPS = {
    "u1": "dp_ch01_s1#b001_t1-2",
    "c_pair": "dp_ch05_s2#b001_t1-2",
    "c_same_noun": "dp_ch05_s3#b001_t1-2",
    "c_early": "dp_ch02_s4#b001_t1-2",
}


def _synth_repo(tmp_path: Path) -> dict[str, Path]:
    """The COMPLETE Ch.1-17 extraction set (34 files — the formal CLI
    reads all of them, no subset). Rows live in Ch.1 uncontracted (pair
    base), Ch.2 contracted (early), Ch.5 contracted (one exact pair
    match, one same-noun-different-prep); every other chapter/form is
    header-only, matching the extractor's zero-hit contract."""
    extracted = tmp_path / "extracted"
    rows_by_file = {
        "hp1_de_ch01_uncontracted.tsv": [_ext_row("s1#b001", prep="in", noun="alpha")],
        "hp1_de_ch02_contracted.tsv": [_ext_row("s4#b001", prep="im", noun="alpha")],
        "hp1_de_ch05_contracted.tsv": [
            _ext_row("s2#b001", prep="im", noun="alpha"),  # in/alpha pair match
            _ext_row("s3#b001", prep="am", noun="alpha"),  # an/alpha — no counterpart
        ],
    }
    for ch in FULL_NOVEL_CHAPTERS:
        for kind in ("contracted", "uncontracted"):
            name = f"hp1_de_ch{ch:02d}_{kind}.tsv"
            _write_extraction_tsv(extracted / name, rows_by_file.get(name, []))
    master = _write_master_tsv(tmp_path / "master.tsv", list(_DPS.values()))
    return {"extraction": extracted, "master": master}


def _review_rows(**overrides) -> list[dict[str, str]]:
    rows = {
        dp: {
            "id": dp,
            "row_hash": "a" * 64,
            "de_valid": "include",
            "de_corrected_lemma": "",
            "de_exclusion_reason": "",
            "de_notes": "",
        }
        for dp in _DPS.values()
    }
    for dp, fields in overrides.items():
        rows[_DPS[dp]].update(fields)
    return list(rows.values())


def _run_cli(mod, tmp_path: Path, *, review_rows=None, argv_extra=()):
    review = _write_review_csv(
        tmp_path / "review.csv",
        review_rows if review_rows is not None else _review_rows(),
    )
    repo = _synth_repo(tmp_path)
    out_dir = tmp_path / "out"
    argv = [
        "--extraction-dir",
        str(repo["extraction"]),
        "--master-tsv",
        str(repo["master"]),
        "--review-csv",
        str(review),
        "--out-dir",
        str(out_dir),
        *argv_extra,
    ]
    return mod.main(argv), out_dir


def test_cli_builds_eligible_pool(tmp_path: Path) -> None:
    mod = _load_script("build_eligible_pool.py")
    rc, out_dir = _run_cli(mod, tmp_path)
    assert rc == 0
    summary = json.loads((out_dir / "eligible_pool_summary.json").read_text(encoding="utf-8"))
    ep = summary["eligible_pool"]
    assert ep["uncontracted_all_chapters"] == 1
    assert ep["contracted_ch1_3"] == 1
    assert ep["contracted_ch4_17_pair_matched"] == 1
    assert ep["eligible_total"] == 3
    assert summary["contracted_ch4_17_no_uncontracted_counterpart"] == 1
    assert summary["automatically_excluded"] == {
        "outside_paired_inventory": 0,
        "failed_structural_gate": 0,
    }
    assert summary["human_review"] == {"included": 4, "excluded": 0}
    assert set(summary["input_hashes"]) == {"extraction", "master", "review_csv"}
    # The pool TSV carries eligible rows only, with corrected lemma +
    # identity columns.
    with open(out_dir / "eligible_pool.tsv", encoding="utf-8") as f:
        pool_rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(pool_rows) == 3
    assert {r["pool_reason"] for r in pool_rows} == {
        "uncontracted_all_chapters",
        "contracted_ch1_3",
        "contracted_ch4_17_pair_matched",
    }


def test_cli_corrected_lemma_flows_through(tmp_path: Path) -> None:
    mod = _load_script("build_eligible_pool.py")
    rc, out_dir = _run_cli(
        mod,
        tmp_path,
        # Correct the pair-matched row's lemma away from alpha → it loses
        # its uncontracted counterpart.
        review_rows=_review_rows(c_pair={"de_corrected_lemma": "beta"}),
    )
    assert rc == 0
    summary = json.loads((out_dir / "eligible_pool_summary.json").read_text(encoding="utf-8"))
    assert summary["eligible_pool"]["contracted_ch4_17_pair_matched"] == 0
    assert summary["contracted_ch4_17_no_uncontracted_counterpart"] == 2
    with open(out_dir / "eligible_pool.tsv", encoding="utf-8") as f:
        pool_rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(pool_rows) == 2


@pytest.mark.parametrize(
    "decision_field",
    [
        {"de_valid": ""},
        {"de_valid": "uncertain"},
    ],
)
def test_cli_incomplete_review_fails_closed(tmp_path: Path, decision_field: dict) -> None:
    """Blank or uncertain decisions — or a master row missing from the
    overlay — mean the review is not finished: exit non-zero, and NO
    output files are written (no partial pool)."""
    mod = _load_script("build_eligible_pool.py")
    rc, out_dir = _run_cli(mod, tmp_path, review_rows=_review_rows(u1=decision_field))
    assert rc == 2
    assert not (out_dir / "eligible_pool.tsv").exists()
    assert not (out_dir / "eligible_pool_summary.json").exists()


def test_cli_master_row_not_reviewed_fails_closed(tmp_path: Path) -> None:
    mod = _load_script("build_eligible_pool.py")
    rows = _review_rows()
    rows = [r for r in rows if r["id"] != _DPS["c_early"]]  # drop one
    rc, out_dir = _run_cli(mod, tmp_path, review_rows=rows)
    assert rc == 2
    assert not (out_dir / "eligible_pool.tsv").exists()


def test_cli_duplicate_review_ids_fail_closed(tmp_path: Path) -> None:
    mod = _load_script("build_eligible_pool.py")
    rows = _review_rows()
    rows.append(dict(rows[0]))  # duplicate datapoint_id
    rc, out_dir = _run_cli(mod, tmp_path, review_rows=rows)
    assert rc == 2
    assert not (out_dir / "eligible_pool.tsv").exists()


def test_cli_source_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    mod = _load_script("build_eligible_pool.py")
    rc, out_dir = _run_cli(
        mod,
        tmp_path,
        review_rows=_review_rows(c_pair={"row_hash": "b" * 64}),
    )
    assert rc == 2
    assert not (out_dir / "eligible_pool.tsv").exists()


def test_cli_review_id_not_in_master_fails_closed(tmp_path: Path) -> None:
    mod = _load_script("build_eligible_pool.py")
    rows = _review_rows()
    rows.append(
        {
            "id": "dp_ch09_unknown#b001_t1-2",
            "row_hash": "a" * 64,
            "de_valid": "include",
        }
    )
    rc, out_dir = _run_cli(mod, tmp_path, review_rows=rows)
    assert rc == 2
    assert not (out_dir / "eligible_pool.tsv").exists()


def test_cli_missing_extraction_file_fails_closed(tmp_path: Path) -> None:
    """The full extraction input set is required — one missing
    chapter/form file fails the run."""
    mod = _load_script("build_eligible_pool.py")
    review = _write_review_csv(tmp_path / "review.csv", _review_rows())
    repo = _synth_repo(tmp_path)
    (repo["extraction"] / "hp1_de_ch05_contracted.tsv").unlink()
    rc = mod.main(
        [
            "--extraction-dir",
            str(repo["extraction"]),
            "--master-tsv",
            str(repo["master"]),
            "--review-csv",
            str(review),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2


def test_cli_extraction_schema_mismatch_fails_closed(tmp_path: Path) -> None:
    mod = _load_script("build_eligible_pool.py")
    review = _write_review_csv(tmp_path / "review.csv", _review_rows())
    repo = _synth_repo(tmp_path)
    # Old-schema header (no det_xpos / det_deprel).
    (repo["extraction"] / "hp1_de_ch01_uncontracted.tsv").write_text(
        "parse_block_id\tsource_segment_id\tprep\tnoun\tpp_token_start\tpp_token_end\n"
        "s1#b001\ts1\tin\talpha\t1\t2\n",
        encoding="utf-8",
    )
    rc = mod.main(
        [
            "--extraction-dir",
            str(repo["extraction"]),
            "--master-tsv",
            str(repo["master"]),
            "--review-csv",
            str(review),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2


def test_cli_identity_conflict_fails_closed(tmp_path: Path) -> None:
    mod = _load_script("build_eligible_pool.py")
    repo = _synth_repo(tmp_path)
    # Same identity twice with different lemmas.
    _write_extraction_tsv(
        repo["extraction"] / "hp1_de_ch01_uncontracted.tsv",
        [_ext_row("s1#b001", noun="alpha"), _ext_row("s1#b001", noun="delta")],
    )
    review = _write_review_csv(tmp_path / "review.csv", _review_rows())
    rc = mod.main(
        [
            "--extraction-dir",
            str(repo["extraction"]),
            "--master-tsv",
            str(repo["master"]),
            "--review-csv",
            str(review),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2


def test_cli_accepts_bomless_returned_csv(tmp_path: Path) -> None:
    """Excel round-trips sometimes drop the BOM — the returned file must
    be read with or without it."""
    mod = _load_script("build_eligible_pool.py")
    rows = _review_rows()
    p = tmp_path / "review_no_bom.csv"
    from hp_corpus.annotation_csv import CSV_COLUMNS

    with open(p, "w", encoding="utf-8", newline="") as f:  # no -sig
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    repo = _synth_repo(tmp_path)
    rc = mod.main(
        [
            "--extraction-dir", str(repo["extraction"]),
            "--master-tsv", str(repo["master"]),
            "--review-csv", str(p),
            "--out-dir", str(tmp_path / "out"),
        ]
    )
    assert rc == 0


def test_cli_refuses_overwrite_without_force(tmp_path: Path) -> None:
    mod = _load_script("build_eligible_pool.py")
    rc, out_dir = _run_cli(mod, tmp_path)
    assert rc == 0
    rc2, _ = _run_cli(mod, tmp_path)
    assert rc2 == 2


def test_cli_master_duplicate_id_fails_closed(tmp_path: Path) -> None:
    """A duplicated master datapoint_id must not silently overwrite."""
    mod = _load_script("build_eligible_pool.py")
    review = _write_review_csv(tmp_path / "review.csv", _review_rows())
    repo = _synth_repo(tmp_path)
    body = (repo["master"]).read_text(encoding="utf-8").splitlines(keepends=True)
    (repo["master"]).write_text("".join(body + [body[1]]), encoding="utf-8")
    rc = mod.main(
        [
            "--extraction-dir",
            str(repo["extraction"]),
            "--master-tsv",
            str(repo["master"]),
            "--review-csv",
            str(review),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    assert not (tmp_path / "out" / "eligible_pool.tsv").exists()


def test_cli_master_row_missing_from_extraction_fails_closed(tmp_path: Path) -> None:
    """File existence is not content: an extraction regression that
    drops a row must fail, not silently shrink the pool."""
    mod = _load_script("build_eligible_pool.py")
    repo = _synth_repo(tmp_path)
    # Drop one extraction row (ch05's no-counterpart occurrence) while
    # master + overlay still reference it.
    _write_extraction_tsv(
        repo["extraction"] / "hp1_de_ch05_contracted.tsv",
        [_ext_row("s2#b001", prep="im", noun="alpha")],
    )
    review = _write_review_csv(tmp_path / "review.csv", _review_rows())
    rc = mod.main(
        [
            "--extraction-dir",
            str(repo["extraction"]),
            "--master-tsv",
            str(repo["master"]),
            "--review-csv",
            str(review),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    assert not (tmp_path / "out" / "eligible_pool.tsv").exists()


def test_cli_extraction_row_missing_from_master_fails_closed(tmp_path: Path) -> None:
    mod = _load_script("build_eligible_pool.py")
    repo = _synth_repo(tmp_path)
    # Add an inventory-eligible extraction row the master doesn't know.
    _write_extraction_tsv(
        repo["extraction"] / "hp1_de_ch09_uncontracted.tsv",
        [_ext_row("s9#b001", prep="in", noun="alpha")],
    )
    review = _write_review_csv(tmp_path / "review.csv", _review_rows())
    rc = mod.main(
        [
            "--extraction-dir",
            str(repo["extraction"]),
            "--master-tsv",
            str(repo["master"]),
            "--review-csv",
            str(review),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    assert not (tmp_path / "out" / "eligible_pool.tsv").exists()


@pytest.mark.parametrize("decision", ["", "uncertain", "Includes", " "])
def test_core_incomplete_review_decision_fails_closed(decision: str) -> None:
    """The public core selector is itself fail-closed: any decision
    outside include/exclude raises before any row is classified — an
    unreviewed row can never enter the pool even without the CLI."""
    occs = [
        _occ(
            "u1",
            chapter=1,
            form="uncontracted",
            canonical_prep="in",
            machine_lemma="alpha",
            decision=decision,
        ),
    ]
    with pytest.raises(IncompleteReviewError):
        build_eligible_pool(occs)


def test_cli_stdout_privacy(tmp_path: Path, capsys) -> None:
    """Stdout carries aggregate counts only — never lemmas, datapoint or
    segment IDs, or surface text."""
    mod = _load_script("build_eligible_pool.py")
    rc, _ = _run_cli(mod, tmp_path)
    assert rc == 0
    out = capsys.readouterr().out
    for leaked in ("alpha", "beta", "s1#b001", "dp_ch", "_seg", "_blk"):
        assert leaked not in out
    assert "eligible_total=3" in out
