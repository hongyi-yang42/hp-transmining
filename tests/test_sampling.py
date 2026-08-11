"""Synthetic-fixture tests for the full-novel sampling rule.

Tests exercise the pure :func:`hp_corpus.sampling.select_sample` function
plus the CLI wrapper. All fixtures use invented lemma strings; no novel
text.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

from hp_corpus.sampling import (
    C_EARLY_CHAPTERS,
    C_LATE_CHAPTERS,
    SAMPLING_REASONS,
    Occurrence,
    canonical_preposition,
    is_inventory_eligible,
    select_sample,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
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
    reviewed_lemma: str = "",
    decision: str = "include",
    inventory_eligible: bool = True,
    manual_override: str = "",
) -> Occurrence:
    return Occurrence(
        datapoint_id=datapoint_id,
        chapter=chapter,
        form=form,
        canonical_prep=canonical_prep,
        machine_head_lemma=machine_lemma,
        reviewed_head_lemma=reviewed_lemma,
        german_candidate_decision=decision,
        inventory_eligible=inventory_eligible,
        source_hash="0" * 64,
        manual_lemma_override=manual_override,
    )


# --------------------------------------------------------------------- pure-function tests


def test_u_selects_every_eligible_uncontracted_in_ch1_17() -> None:
    occs = [
        _occ("u1", chapter=1, form="uncontracted", canonical_prep="in", machine_lemma="alpha"),
        _occ("u2", chapter=8, form="uncontracted", canonical_prep="an", machine_lemma="beta"),
        _occ("u3", chapter=17, form="uncontracted", canonical_prep="auf", machine_lemma="gamma"),
    ]
    result = select_sample(occs)
    selected = {r.occurrence.datapoint_id for r in result.ledger if r.sampling_selected}
    assert selected == {"u1", "u2", "u3"}
    for r in result.ledger:
        assert r.sampling_reason == "uncontracted_full_novel"
        assert r.sampling_status == "selected"


def test_c_early_selects_every_eligible_contracted_in_ch1_3() -> None:
    occs = [
        _occ("c1", chapter=1, form="contracted", canonical_prep="in", machine_lemma="alpha"),
        _occ("c2", chapter=2, form="contracted", canonical_prep="an", machine_lemma="beta"),
        _occ("c3", chapter=3, form="contracted", canonical_prep="auf", machine_lemma="gamma"),
    ]
    result = select_sample(occs)
    selected = {r.occurrence.datapoint_id for r in result.ledger if r.sampling_selected}
    assert selected == {"c1", "c2", "c3"}
    for r in result.ledger:
        assert r.sampling_reason == "contracted_ch1_3"


def test_c_late_noun_match_selects_and_back_fills_supports() -> None:
    """A Ch.5 contracted occurrence whose lemma matches a U lemma is
    selected, and the U row records it under supports_late_contracted_ids."""
    occs = [
        _occ("u1", chapter=1, form="uncontracted", canonical_prep="in", machine_lemma="alpha"),
        _occ("c_late_1", chapter=5, form="contracted", canonical_prep="in", machine_lemma="alpha"),
        # Different lemma → no match.
        _occ("c_late_2", chapter=6, form="contracted", canonical_prep="an", machine_lemma="beta"),
    ]
    result = select_sample(occs)
    by_id = {r.occurrence.datapoint_id: r for r in result.ledger}
    assert by_id["c_late_1"].sampling_selected is True
    assert by_id["c_late_1"].sampling_reason == "contracted_ch4_17_noun_match"
    assert by_id["c_late_2"].sampling_selected is False
    assert by_id["c_late_2"].sampling_reason == "contracted_ch4_17_no_noun_match"
    # U row records the supported C_late ID.
    assert "c_late_1" in by_id["u1"].supports_late_contracted_ids
    assert "c_late_2" not in by_id["u1"].supports_late_contracted_ids


def test_c_late_lemma_match_does_not_require_same_preposition() -> None:
    """The C_late expansion is keyed on head-noun lemma ALONE. A C_late
    occurrence with a different canonical preposition than the U row but
    the same lemma still selects."""
    occs = [
        _occ("u1", chapter=1, form="uncontracted", canonical_prep="in", machine_lemma="alpha"),
        _occ("c_late_1", chapter=7, form="contracted", canonical_prep="an", machine_lemma="alpha"),
    ]
    result = select_sample(occs)
    by_id = {r.occurrence.datapoint_id: r for r in result.ledger}
    assert by_id["c_late_1"].sampling_selected is True
    assert by_id["c_late_1"].sampling_reason == "contracted_ch4_17_noun_match"


def test_uncertain_decision_never_selects() -> None:
    occs = [
        _occ(
            "u_unc",
            chapter=1,
            form="uncontracted",
            canonical_prep="in",
            machine_lemma="alpha",
            decision="uncertain",
        ),
        _occ(
            "c_unc",
            chapter=2,
            form="contracted",
            canonical_prep="in",
            machine_lemma="alpha",
            decision="uncertain",
        ),
        _occ(
            "c_late_unc",
            chapter=5,
            form="contracted",
            canonical_prep="in",
            machine_lemma="alpha",
            decision="uncertain",
        ),
    ]
    result = select_sample(occs)
    for r in result.ledger:
        assert r.sampling_selected is False
        assert r.sampling_reason == "blocked_german_review"
        assert r.sampling_status == "blocked"


def test_blank_decision_blocks() -> None:
    occs = [
        _occ(
            "u_blank",
            chapter=1,
            form="uncontracted",
            canonical_prep="in",
            machine_lemma="alpha",
            decision="",
        )
    ]
    result = select_sample(occs)
    r = result.ledger[0]
    assert r.sampling_selected is False
    assert r.sampling_reason == "blocked_german_review"


def test_exclude_routes_to_excluded_by_german_review() -> None:
    occs = [
        _occ(
            "u_ex",
            chapter=1,
            form="uncontracted",
            canonical_prep="in",
            machine_lemma="alpha",
            decision="exclude",
        )
    ]
    result = select_sample(occs)
    r = result.ledger[0]
    assert r.sampling_selected is False
    assert r.sampling_reason == "excluded_by_german_review"
    assert r.sampling_status == "not_selected"


def test_outside_inventory_routes_correctly() -> None:
    """A row whose canonical_preposition is not in the paper's paired
    inventory (e.g. ``um``) goes to outside_author_inventory."""
    occs = [
        _occ(
            "u_um",
            chapter=1,
            form="uncontracted",
            canonical_prep="um",
            machine_lemma="alpha",
            inventory_eligible=False,
        )
    ]
    result = select_sample(occs)
    r = result.ledger[0]
    assert r.sampling_reason == "outside_author_inventory"
    assert r.sampling_status == "not_selected"


def test_missing_lemma_for_u_blocks() -> None:
    """A U row without any effective lemma cannot contribute to U's lemma
    set; block it under blocked_lemma_review."""
    occs = [
        _occ("u_no_lemma", chapter=1, form="uncontracted", canonical_prep="in", machine_lemma=""),
    ]
    result = select_sample(occs)
    r = result.ledger[0]
    assert r.sampling_reason == "blocked_lemma_review"
    assert r.sampling_status == "blocked"


def test_missing_lemma_for_c_late_blocks() -> None:
    """A C_late row without an effective lemma cannot be matched against
    U; block it under blocked_lemma_review."""
    occs = [
        _occ("u1", chapter=1, form="uncontracted", canonical_prep="in", machine_lemma="alpha"),
        _occ(
            "c_late_no_lemma",
            chapter=5,
            form="contracted",
            canonical_prep="in",
            machine_lemma="",
        ),
    ]
    result = select_sample(occs)
    by_id = {r.occurrence.datapoint_id: r for r in result.ledger}
    assert by_id["c_late_no_lemma"].sampling_reason == "blocked_lemma_review"
    assert by_id["c_late_no_lemma"].sampling_status == "blocked"


def test_c_early_without_lemma_still_selects() -> None:
    """C_early selection doesn't depend on the lemma (form + chapter is
    enough); a blank lemma is fine here."""
    occs = [
        _occ("c1", chapter=1, form="contracted", canonical_prep="in", machine_lemma=""),
    ]
    result = select_sample(occs)
    r = result.ledger[0]
    assert r.sampling_reason == "contracted_ch1_3"
    assert r.sampling_status == "selected"


def test_reviewed_lemma_preferred_over_machine() -> None:
    """When both machine and reviewed lemma are present and
    use_reviewed_lemma=True (default), the reviewed lemma drives the
    match."""
    occs = [
        _occ(
            "u1",
            chapter=1,
            form="uncontracted",
            canonical_prep="in",
            machine_lemma="wrong",
            reviewed_lemma="alpha",
        ),
        _occ(
            "c_late",
            chapter=5,
            form="contracted",
            canonical_prep="in",
            machine_lemma="wrong",
            reviewed_lemma="alpha",
        ),
    ]
    result = select_sample(occs, use_reviewed_lemma=True)
    by_id = {r.occurrence.datapoint_id: r for r in result.ledger}
    assert by_id["c_late"].sampling_selected is True


def test_machine_lemma_used_when_reviewed_absent() -> None:
    occs = [
        _occ(
            "u1",
            chapter=1,
            form="uncontracted",
            canonical_prep="in",
            machine_lemma="alpha",
            reviewed_lemma="",
        ),
        _occ(
            "c_late",
            chapter=5,
            form="contracted",
            canonical_prep="in",
            machine_lemma="alpha",
            reviewed_lemma="",
        ),
    ]
    result = select_sample(occs)
    by_id = {r.occurrence.datapoint_id: r for r in result.ledger}
    assert by_id["c_late"].sampling_selected is True


def test_machine_lemma_only_mode() -> None:
    """Provisional ledger: ignore reviewed lemma, use machine only."""
    occs = [
        _occ(
            "u1",
            chapter=1,
            form="uncontracted",
            canonical_prep="in",
            machine_lemma="alpha",
            reviewed_lemma="beta",  # would match C_late in default mode
        ),
        _occ(
            "c_late",
            chapter=5,
            form="contracted",
            canonical_prep="in",
            machine_lemma="alpha",
            reviewed_lemma="gamma",
        ),
    ]
    result = select_sample(occs, use_reviewed_lemma=False)
    by_id = {r.occurrence.datapoint_id: r for r in result.ledger}
    # Machine lemmas match → C_late selected.
    assert by_id["c_late"].sampling_selected is True


def test_manual_lemma_override_wins() -> None:
    """Manual override beats both machine and reviewed lemma."""
    occs = [
        _occ(
            "u1",
            chapter=1,
            form="uncontracted",
            canonical_prep="in",
            machine_lemma="wrong",
            reviewed_lemma="also_wrong",
            manual_override="alpha",
        ),
        _occ(
            "c_late",
            chapter=5,
            form="contracted",
            canonical_prep="in",
            machine_lemma="wrong2",
            reviewed_lemma="also_wrong2",
            manual_override="alpha",
        ),
    ]
    result = select_sample(occs)
    by_id = {r.occurrence.datapoint_id: r for r in result.ledger}
    assert by_id["c_late"].sampling_selected is True


def test_summary_counts_are_consistent() -> None:
    occs = [
        _occ("u1", chapter=1, form="uncontracted", canonical_prep="in", machine_lemma="alpha"),
        _occ("u2", chapter=2, form="uncontracted", canonical_prep="an", machine_lemma="beta"),
        _occ("c1", chapter=1, form="contracted", canonical_prep="in", machine_lemma="alpha"),
        _occ(
            "c_late_match",
            chapter=5,
            form="contracted",
            canonical_prep="in",
            machine_lemma="alpha",
        ),
        _occ(
            "c_late_nomatch",
            chapter=6,
            form="contracted",
            canonical_prep="an",
            machine_lemma="gamma",
        ),
        _occ(
            "u_ex",
            chapter=1,
            form="uncontracted",
            canonical_prep="in",
            machine_lemma="delta",
            decision="exclude",
        ),
        _occ(
            "u_unc",
            chapter=1,
            form="uncontracted",
            canonical_prep="in",
            machine_lemma="epsilon",
            decision="uncertain",
        ),
    ]
    result = select_sample(occs)
    s = result.summary
    assert s["occurrence_total"] == 7
    assert s["selected_total"] == 4  # u1, u2, c1, c_late_match
    assert s["by_reason"]["uncontracted_full_novel"] == 2
    assert s["by_reason"]["contracted_ch1_3"] == 1
    assert s["by_reason"]["contracted_ch4_17_noun_match"] == 1
    assert s["by_reason"]["contracted_ch4_17_no_noun_match"] == 1
    assert s["by_reason"]["excluded_by_german_review"] == 1
    assert s["by_reason"]["blocked_german_review"] == 1
    assert s["u_lemma_count"] == 2  # alpha, beta


def test_constants_cover_all_workpackage_reasons() -> None:
    expected = {
        "uncontracted_full_novel",
        "contracted_ch1_3",
        "contracted_ch4_17_noun_match",
        "contracted_ch4_17_no_noun_match",
        "outside_author_inventory",
        "excluded_by_german_review",
        "blocked_german_review",
        "blocked_lemma_review",
    }
    assert expected.issubset(SAMPLING_REASONS)


def test_constants_chapter_ranges() -> None:
    assert list(C_EARLY_CHAPTERS) == [1, 2, 3]
    assert list(C_LATE_CHAPTERS) == list(range(4, 18))


def test_canonical_preposition_helper() -> None:
    assert canonical_preposition("im") == "in"
    assert canonical_preposition("in") == "in"
    assert canonical_preposition("um") == "um"  # passes through


def test_is_inventory_eligible_helper() -> None:
    assert is_inventory_eligible("in") is True
    assert is_inventory_eligible("an") is True
    assert is_inventory_eligible("um") is False  # not in paired inventory
    assert is_inventory_eligible("vor") is False


# --------------------------------------------------------------------- CLI tests


def _write_extraction_tsv(path: Path, rows: list[dict[str, str]]) -> Path:
    fields = [
        "sentence_id",
        "prep",
        "det",
        "noun",
        "prep_token_id",
        "det_token_id",
        "noun_token_id",
        "pp_token_start",
        "pp_token_end",
        "pp_surface",
        "in_filter",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            row = {**r}
            if row.get("det") is None:
                row["det"] = "-"
            if row.get("det_token_id") is None:
                row["det_token_id"] = "-"
            w.writerow({k: row.get(k, "") for k in fields})
    return path


def _write_master_tsv(path: Path, decisions: dict[str, str]) -> Path:
    """Write a minimal master annotation TSV with the given
    ``datapoint_id → de_candidate_decision`` map. Only the columns the
    CLI reads are populated."""
    fields = ["datapoint_id", "de_candidate_decision", "source_row_sha256"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for dp, decision in decisions.items():
            row = {}
            for c in fields:
                if c == "datapoint_id":
                    row[c] = dp
                elif c == "de_candidate_decision":
                    row[c] = decision
                else:
                    row[c] = "0" * 64
            w.writerow(row)
    return path


def test_cli_writes_ledger_and_summary(tmp_path: Path) -> None:
    cli = _load_script("build_full_novel_sampling_ledger.py")
    extraction = tmp_path / "extracted"
    _write_extraction_tsv(
        extraction / "hp1_de_ch01_uncontracted.tsv",
        [
            {
                "sentence_id": "hp1_de_ch01_p0001_s001",
                "prep": "in",
                "det": "dem",
                "noun": "alpha",
                "prep_token_id": "1",
                "det_token_id": "2",
                "noun_token_id": "3",
                "pp_token_start": "1",
                "pp_token_end": "3",
                "pp_surface": "synth PP",
                "in_filter": "Y",
            }
        ],
    )
    _write_extraction_tsv(
        extraction / "hp1_de_ch01_contracted.tsv",
        [
            {
                "sentence_id": "hp1_de_ch01_p0002_s001",
                "prep": "im",
                "noun": "alpha",
                "prep_token_id": "1",
                "noun_token_id": "2",
                "pp_token_start": "1",
                "pp_token_end": "2",
                "pp_surface": "synth PP",
                "in_filter": "Y",
            }
        ],
    )
    master = _write_master_tsv(
        tmp_path / "master.tsv",
        {
            "dp_ch01_hp1_de_ch01_p0001_s001_t1-3": "include",
            "dp_ch01_hp1_de_ch01_p0002_s001_t1-2": "include",
        },
    )
    out_dir = tmp_path / "out"
    rc = cli.main([
        "--extraction-dir", str(extraction),
        "--master-tsv", str(master),
        "--chapters", "1",
        "--out-dir", str(out_dir),
    ])
    assert rc == 0
    ledger_path = out_dir / "full_novel_ledger.tsv"
    summary_path = out_dir / "full_novel_summary.json"
    assert ledger_path.exists()
    assert summary_path.exists()
    with open(ledger_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(rows) == 2
    by_id = {r["datapoint_id"]: r for r in rows}
    # The uncontracted U row + the contracted C_early row both select.
    u_id = "dp_ch01_hp1_de_ch01_p0001_s001_t1-3"
    c_id = "dp_ch01_hp1_de_ch01_p0002_s001_t1-2"
    assert by_id[u_id]["sampling_status"] == "selected"
    assert by_id[c_id]["sampling_status"] == "selected"


def test_cli_ch10_paths_work(tmp_path: Path) -> None:
    """Smoke test that the CLI requests hp1_de_ch10_*.tsv (zero-pad)."""
    cli = _load_script("build_full_novel_sampling_ledger.py")
    extraction = tmp_path / "extracted"
    # Ch.10 uncontracted TSV present.
    _write_extraction_tsv(
        extraction / "hp1_de_ch10_uncontracted.tsv",
        [
            {
                "sentence_id": "hp1_de_ch10_p0001_s001",
                "prep": "in",
                "det": "dem",
                "noun": "alpha",
                "prep_token_id": "1",
                "det_token_id": "2",
                "noun_token_id": "3",
                "pp_token_start": "1",
                "pp_token_end": "3",
                "pp_surface": "synth PP",
                "in_filter": "Y",
            }
        ],
    )
    rc = cli.main([
        "--extraction-dir", str(extraction),
        "--chapters", "10",
        "--out-dir", str(tmp_path / "out"),
    ])
    assert rc == 0


def test_cli_chapter_out_of_range(tmp_path: Path) -> None:
    cli = _load_script("build_full_novel_sampling_ledger.py")
    extraction = tmp_path / "extracted"
    extraction.mkdir()
    rc = cli.main([
        "--extraction-dir", str(extraction),
        "--chapters", "18",
        "--out-dir", str(tmp_path / "out"),
    ])
    assert rc == 2


def test_cli_no_occurrences(tmp_path: Path) -> None:
    cli = _load_script("build_full_novel_sampling_ledger.py")
    extraction = tmp_path / "extracted"
    extraction.mkdir()
    rc = cli.main([
        "--extraction-dir", str(extraction),
        "--chapters", "1",
        "--out-dir", str(tmp_path / "out"),
    ])
    assert rc == 2


def test_cli_refuses_overwrite_without_force(tmp_path: Path) -> None:
    cli = _load_script("build_full_novel_sampling_ledger.py")
    extraction = tmp_path / "extracted"
    _write_extraction_tsv(
        extraction / "hp1_de_ch01_uncontracted.tsv",
        [
            {
                "sentence_id": "hp1_de_ch01_p0001_s001",
                "prep": "in",
                "det": "dem",
                "noun": "alpha",
                "prep_token_id": "1",
                "det_token_id": "2",
                "noun_token_id": "3",
                "pp_token_start": "1",
                "pp_token_end": "3",
                "pp_surface": "synth PP",
                "in_filter": "Y",
            }
        ],
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "full_novel_ledger.tsv").write_text("placeholder", encoding="utf-8")
    rc = cli.main([
        "--extraction-dir", str(extraction),
        "--chapters", "1",
        "--out-dir", str(out_dir),
    ])
    assert rc == 2


def test_cli_stdout_privacy(tmp_path: Path, capsys) -> None:
    """Stdout must not carry lemmas, surface forms, or datapoint IDs."""
    cli = _load_script("build_full_novel_sampling_ledger.py")
    extraction = tmp_path / "extracted"
    _write_extraction_tsv(
        extraction / "hp1_de_ch01_uncontracted.tsv",
        [
            {
                "sentence_id": "hp1_de_ch01_p0001_s001",
                "prep": "in",
                "det": "dem",
                "noun": "alpha",
                "prep_token_id": "1",
                "det_token_id": "2",
                "noun_token_id": "3",
                "pp_token_start": "1",
                "pp_token_end": "3",
                "pp_surface": "synth PP",
                "in_filter": "Y",
            }
        ],
    )
    cli.main([
        "--extraction-dir", str(extraction),
        "--chapters", "1",
        "--out-dir", str(tmp_path / "out"),
    ])
    out = capsys.readouterr().out
    assert "alpha" not in out
    assert "synth PP" not in out
    assert "hp1_de_ch01_p0001_s001" not in out
