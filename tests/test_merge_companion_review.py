"""Synthetic-fixture tests for the split-pair rejoin merger.

Covers the deterministic happy path and every fail-closed rule of
``scripts/merge_companion_review.py``, plus the runbook guarantee that a
merged file with invalid/blank human annotation is rejected by the ordinary
full validator (never by the merger itself — it does not interpret human
cells). All fixture text is invented; no novel text.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_master(path: Path, ids: list[str], values: dict | None = None) -> Path:
    from hp_corpus.step4 import ALL_TSV_COLUMNS

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ALL_TSV_COLUMNS), delimiter="\t")
        w.writeheader()
        for dp in ids:
            row = {
                col: dp
                if col == "datapoint_id"
                else "a" * 64
                if col == "source_row_sha256"
                else "anchor_window"
                if col in ("en_context_provenance", "zh_context_provenance")
                else ""
                for col in ALL_TSV_COLUMNS
            }
            row.update((values or {}).get(dp, {}))
            w.writerow(row)
    return path


def _pair_row_values(
    dp: str,
    en_conf: str,
    zh_conf: str,
    *,
    en_prov: str = "anchor_window",
    zh_prov: str = "anchor_window",
    en_ctx: str = "He went into the house.",
    zh_ctx: str = "他走进了房子。",
    de_sentence: str = "Er ging ins Haus.",
) -> dict:
    return {
        dp: {
            "chapter": "3",
            "de_form": "contracted",
            "de_pp_surface": "im Haus",
            "de_head_lemma": "Haus",
            "de_sentence_text": de_sentence,
            "en_context_text": en_ctx,
            "zh_context_text": zh_ctx,
            "en_alignment_confidence": en_conf,
            "zh_alignment_confidence": zh_conf,
            "en_context_provenance": en_prov,
            "zh_context_provenance": zh_prov,
        }
    }


def _build_pair(tmp_path: Path, values_by_dp: dict):
    build = _load_script("build_annotation_csv.py")
    master = _write_master(tmp_path / "master.tsv", list(values_by_dp), values_by_dp)
    main_csv = tmp_path / "annotation_pairs.csv"
    low_csv = tmp_path / "low.csv"
    rc = build.main([
        "--master-tsv", str(master), "--output", str(main_csv),
        "--min-confidence", "0.4", "--low-confidence-output", str(low_csv),
    ])
    assert rc == 0
    return master, main_csv, low_csv


def _read_csv(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [{c: r.get(c, "") for c in columns} for r in csv.DictReader(f)]


# Master order is deliberately NOT sorted, and the two files own
# interleaved subsets, so order restoration is actually exercised.
_SCENARIO: dict = {}
_SCENARIO.update(_pair_row_values("dp3", "0.2", "0.9"))  # conf-deferred (en)
_SCENARIO.update(_pair_row_values("dp1", "0.9", "0.9"))  # kept
_SCENARIO.update(_pair_row_values(
    "dp4", "0.9", "0.9", zh_prov="neighbor_fallback"))  # provenance-deferred
_SCENARIO.update(_pair_row_values(
    "dp5", "0.8", "0.8", de_sentence="- «Er ist weg»."))  # kept; Excel-unsafe text
_SCENARIO.update(_pair_row_values("dp2", "0.9", "0.3"))  # conf-deferred (zh)
_MASTER_ORDER = ["dp3", "dp1", "dp4", "dp5", "dp2"]

_EXCLUDE_SPEC = {
    "de_valid": "exclude",
    "de_exclusion_reason": "not_target_pp",
    "en_counterpart": "",
    "en_form": "",
    "en_alignment_confidence": "",
    "zh_counterpart": "",
    "zh_form": "",
    "zh_alignment_confidence": "",
}


def _fill(rows: list[dict[str, str]], spec_by_dp: dict | None = None) -> list[dict[str, str]]:
    """Fill annotator cells validly (spec overrides per id)."""
    out = []
    for r in rows:
        fill = {
            "de_valid": "include",
            "de_corrected_lemma": "",
            "de_exclusion_reason": "",
            "de_notes": "",
            "en_counterpart": "in the house",
            "en_form": "definite",
            "en_alignment_confidence": "high",
            "en_notes": "",
            "zh_counterpart": "在房子里",
            "zh_form": "bare",
            "zh_alignment_confidence": "high",
            "zh_notes": "",
        }
        fill.update((spec_by_dp or {}).get(r["id"], {}))
        merged = dict(r)
        merged.update(fill)
        out.append(merged)
    return out


def _fill_pair(main_csv: Path, low_csv: Path, spec_by_dp: dict | None = None) -> None:
    from hp_corpus.annotation_csv import CSV_COLUMNS, LOW_CONF_COLUMNS

    main_rows = _fill(_read_csv(main_csv, CSV_COLUMNS), spec_by_dp)
    low_rows = _fill(_read_csv(low_csv, LOW_CONF_COLUMNS), spec_by_dp)
    with open(main_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS), lineterminator="\r\n")
        w.writeheader()
        w.writerows(main_rows)
    with open(low_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(LOW_CONF_COLUMNS), lineterminator="\r\n")
        w.writeheader()
        w.writerows(low_rows)


def _scenario_filled(tmp_path: Path, spec_by_dp: dict | None = None):
    master, main_csv, low_csv = _build_pair(tmp_path, _SCENARIO)
    _fill_pair(main_csv, low_csv, spec_by_dp)
    return master, main_csv, low_csv


def _run_merge(main_csv: Path, low_csv: Path, master: Path, out: Path,
               thr: str = "0.4", force: bool = True) -> int:
    mod = _load_script("merge_companion_review.py")
    argv = [
        "--review-csv", str(main_csv), "--companion-csv", str(low_csv),
        "--master-tsv", str(master), "--min-confidence", thr, "--output", str(out),
    ]
    if force:
        argv.append("--force-output")
    return mod.main(argv)


def _run_full_validate(merged: Path, master: Path) -> int:
    mod = _load_script("validate_annotation_csv.py")
    return mod.main([str(merged), "--master-tsv", str(master)])


# --------------------------------------------------------------- happy path


def test_valid_reviewed_pair_merges_and_passes_full_validator(tmp_path: Path, capsys) -> None:
    master, main_csv, low_csv = _scenario_filled(
        tmp_path,
        {"dp4": _EXCLUDE_SPEC, "dp3": {"de_notes": "trailing space kept "}},
    )
    merged = tmp_path / "merged.csv"
    assert _run_merge(main_csv, low_csv, master, merged) == 0
    capsys.readouterr()  # discard merger output

    from hp_corpus.annotation_csv import CSV_COLUMNS

    with open(merged, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    assert header == list(CSV_COLUMNS)
    assert "machine_low_conf_sides" not in header  # diagnostics dropped
    assert [r["id"] for r in rows] == _MASTER_ORDER
    by_id = {r["id"]: r for r in rows}

    # Human cells verbatim — including companion-origin and exact spacing.
    assert by_id["dp4"]["de_valid"] == "exclude"
    assert by_id["dp4"]["de_exclusion_reason"] == "not_target_pp"
    assert by_id["dp4"]["en_counterpart"] == ""
    assert by_id["dp3"]["de_notes"] == "trailing space kept "
    assert by_id["dp2"]["zh_counterpart"] == "在房子里"

    # Machine cells are the canonical master projection (Excel-safe space
    # re-applied, regardless of what the returned files carried).
    assert by_id["dp5"]["german_sentence"] == " - «Er ist weg»."
    assert by_id["dp1"]["row_hash"] == "a" * 64

    # The runbook's actual gate: the ordinary full validator accepts the
    # merged file with EVERY row human-filled — non-template state, so the
    # content rules (de_valid vocabulary, form⇒span coupling, confidence
    # coupling) genuinely ran over the companion-origin rows too.
    assert _run_full_validate(merged, master) == 0
    out = capsys.readouterr().out
    assert "template_state: False" in out
    assert "rows: 5" in out


def test_merge_is_deterministic(tmp_path: Path) -> None:
    master, main_csv, low_csv = _scenario_filled(tmp_path)
    out1 = tmp_path / "merged1.csv"
    out2 = tmp_path / "merged2.csv"
    assert _run_merge(main_csv, low_csv, master, out1) == 0
    assert _run_merge(main_csv, low_csv, master, out2) == 0
    h1 = hashlib.sha256(out1.read_bytes()).hexdigest()
    h2 = hashlib.sha256(out2.read_bytes()).hexdigest()
    assert h1 == h2

    # Row order of the returned files is irrelevant to the output.
    from hp_corpus.annotation_csv import CSV_COLUMNS

    shuffled = list(reversed(_read_csv(main_csv, CSV_COLUMNS)))
    with open(main_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS), lineterminator="\r\n")
        w.writeheader()
        w.writerows(shuffled)
    out3 = tmp_path / "merged3.csv"
    assert _run_merge(main_csv, low_csv, master, out3) == 0
    assert hashlib.sha256(out3.read_bytes()).hexdigest() == h1


def test_stdout_privacy_and_counts(tmp_path: Path, capsys) -> None:
    master, main_csv, low_csv = _scenario_filled(tmp_path)
    out = tmp_path / "merged.csv"
    assert _run_merge(main_csv, low_csv, master, out) == 0
    captured = capsys.readouterr()
    for leaked in ("house", "房子", "dp1", "Haus"):
        assert leaked not in captured.out
        assert leaked not in captured.err
    assert "from_main: 2" in captured.out
    assert "from_companion: 3" in captured.out
    assert "rows: 5" in captured.out


# ------------------------------------------------------------ fail-closed rules


def test_missing_companion_row_fails(tmp_path: Path, capsys) -> None:
    master, main_csv, low_csv = _scenario_filled(tmp_path)
    from hp_corpus.annotation_csv import LOW_CONF_COLUMNS

    rows = _read_csv(low_csv, LOW_CONF_COLUMNS)
    rows = [r for r in rows if r["id"] != "dp2"]
    with open(low_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(LOW_CONF_COLUMNS), lineterminator="\r\n")
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "merged.csv"
    assert _run_merge(main_csv, low_csv, master, out) == 2
    assert "COMPANION_EXPECTED_ROW_MISSING" in capsys.readouterr().out
    assert not out.exists()  # nothing written on failure


def test_extra_id_in_companion_fails(tmp_path: Path, capsys) -> None:
    master, main_csv, low_csv = _scenario_filled(tmp_path)
    from hp_corpus.annotation_csv import LOW_CONF_COLUMNS

    rows = _read_csv(low_csv, LOW_CONF_COLUMNS)
    extra = dict(rows[0])
    extra["id"] = "dp1"  # a kept id smuggled into the companion
    rows.append(extra)
    with open(low_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(LOW_CONF_COLUMNS), lineterminator="\r\n")
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "merged.csv"
    assert _run_merge(main_csv, low_csv, master, out) == 2
    out_text = capsys.readouterr().out
    assert "COMPANION_ID_NOT_EXPECTED_SET" in out_text


def test_row_moved_between_files_fails(tmp_path: Path, capsys) -> None:
    master, main_csv, low_csv = _scenario_filled(tmp_path)
    from hp_corpus.annotation_csv import CSV_COLUMNS, LOW_CONF_COLUMNS

    main_rows = _read_csv(main_csv, CSV_COLUMNS)
    low_rows = _read_csv(low_csv, LOW_CONF_COLUMNS)
    moved = next(r for r in low_rows if r["id"] == "dp2")
    main_rows.append({k: moved.get(k, "") for k in CSV_COLUMNS})
    low_rows = [r for r in low_rows if r["id"] != "dp2"]
    with open(main_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS), lineterminator="\r\n")
        w.writeheader()
        w.writerows(main_rows)
    with open(low_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(LOW_CONF_COLUMNS), lineterminator="\r\n")
        w.writeheader()
        w.writerows(low_rows)
    out = tmp_path / "merged.csv"
    assert _run_merge(main_csv, low_csv, master, out) == 2
    out_text = capsys.readouterr().out
    assert "MAIN_ID_NOT_EXPECTED_SET" in out_text
    assert "COMPANION_EXPECTED_ROW_MISSING" in out_text


def test_edited_machine_column_in_main_fails(tmp_path: Path, capsys) -> None:
    master, main_csv, low_csv = _scenario_filled(tmp_path)
    from hp_corpus.annotation_csv import CSV_COLUMNS

    rows = _read_csv(main_csv, CSV_COLUMNS)
    for r in rows:
        if r["id"] == "dp1":
            r["english_context"] = "He went into the EDITED house."
    with open(main_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS), lineterminator="\r\n")
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "merged.csv"
    assert _run_merge(main_csv, low_csv, master, out) == 2
    assert "MAIN_MACHINE_COLUMN_EDITED" in capsys.readouterr().out


def test_edited_machine_column_in_companion_fails(tmp_path: Path, capsys) -> None:
    master, main_csv, low_csv = _scenario_filled(tmp_path)
    from hp_corpus.annotation_csv import LOW_CONF_COLUMNS

    rows = _read_csv(low_csv, LOW_CONF_COLUMNS)
    for r in rows:
        if r["id"] == "dp2":
            r["german_sentence"] = "EDITED."
    with open(low_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(LOW_CONF_COLUMNS), lineterminator="\r\n")
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "merged.csv"
    assert _run_merge(main_csv, low_csv, master, out) == 2
    assert "COMPANION_MACHINE_COLUMN_EDITED" in capsys.readouterr().out


def test_edited_companion_diagnostic_fails(tmp_path: Path, capsys) -> None:
    master, main_csv, low_csv = _scenario_filled(tmp_path)
    from hp_corpus.annotation_csv import LOW_CONF_COLUMNS

    rows = _read_csv(low_csv, LOW_CONF_COLUMNS)
    for r in rows:
        if r["id"] == "dp2":
            r["machine_low_conf_sides"] = "en"  # truth: zh
    with open(low_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(LOW_CONF_COLUMNS), lineterminator="\r\n")
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "merged.csv"
    assert _run_merge(main_csv, low_csv, master, out) == 2
    assert "COMPANION_MACHINE_COLUMN_EDITED" in capsys.readouterr().out


def test_row_hash_mismatch_fails(tmp_path: Path, capsys) -> None:
    master, main_csv, low_csv = _scenario_filled(tmp_path)
    from hp_corpus.annotation_csv import CSV_COLUMNS

    rows = _read_csv(main_csv, CSV_COLUMNS)
    for r in rows:
        if r["id"] == "dp5":
            r["row_hash"] = "b" * 64
    with open(main_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS), lineterminator="\r\n")
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "merged.csv"
    assert _run_merge(main_csv, low_csv, master, out) == 2
    assert "MAIN_ROW_HASH_MISMATCH" in capsys.readouterr().out


def test_wrong_threshold_fails(tmp_path: Path) -> None:
    # A threshold that disagrees with the pair's build re-derives a
    # different split; the file pair no longer matches it.
    master, main_csv, low_csv = _scenario_filled(tmp_path)
    out = tmp_path / "merged.csv"
    assert _run_merge(main_csv, low_csv, master, out, thr="0.95") == 2


def test_header_mismatch_fails(tmp_path: Path, capsys) -> None:
    master, main_csv, low_csv = _scenario_filled(tmp_path)
    text = main_csv.read_text(encoding="utf-8-sig")
    main_csv.write_text(text.replace("row_hash", "rowhash", 1), encoding="utf-8-sig")
    out = tmp_path / "merged.csv"
    assert _run_merge(main_csv, low_csv, master, out) == 2
    assert "MAIN_HEADER_MISMATCH" in capsys.readouterr().out


def test_output_exists_requires_force(tmp_path: Path) -> None:
    master, main_csv, low_csv = _scenario_filled(tmp_path)
    out = tmp_path / "merged.csv"
    out.write_text("stale", encoding="utf-8")
    assert _run_merge(main_csv, low_csv, master, out, force=False) == 2
    assert out.read_text(encoding="utf-8") == "stale"  # untouched
    assert _run_merge(main_csv, low_csv, master, out, force=True) == 0


# --------------------------------------- human-content gate stays with the validator


def test_blank_companion_annotation_rejected_by_full_validator_not_merger(tmp_path: Path) -> None:
    # The merger does not interpret human cells: a pair whose companion is
    # still blank merges fine — the ordinary full validator on the merged
    # file is what fails (merged file is partially filled, not template).
    master, main_csv, low_csv = _build_pair(tmp_path, _SCENARIO)
    from hp_corpus.annotation_csv import CSV_COLUMNS

    main_rows = _fill(_read_csv(main_csv, CSV_COLUMNS))
    with open(main_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS), lineterminator="\r\n")
        w.writeheader()
        w.writerows(main_rows)
    # low_csv stays template-blank.

    merged = tmp_path / "merged.csv"
    assert _run_merge(main_csv, low_csv, master, merged) == 0
    assert _run_full_validate(merged, master) == 2


def test_invalid_companion_annotation_rejected_by_full_validator(tmp_path: Path) -> None:
    # A de_valid typo (or vocabulary violation) on a companion-origin row
    # passes the merger verbatim and is caught by the full validator —
    # this is the gap the rejoin path closes.
    master, main_csv, low_csv = _scenario_filled(tmp_path, {"dp2": {"de_valid": "exlude"}})
    merged = tmp_path / "merged.csv"
    assert _run_merge(main_csv, low_csv, master, merged) == 0
    with open(merged, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_id = {r["id"]: r for r in rows}
    assert by_id["dp2"]["de_valid"] == "exlude"  # preserved verbatim
    assert _run_full_validate(merged, master) == 2
