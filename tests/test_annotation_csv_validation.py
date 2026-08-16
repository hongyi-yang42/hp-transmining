"""Synthetic-fixture tests for the returned-annotation-CSV validator.

One test per fail-closed rule plus the template-state allowance and the
BOM/BOM-less tolerance. All fixture text is invented; no novel text.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_master(path: Path, ids: list[str]) -> Path:
    from hp_corpus.step4 import ALL_TSV_COLUMNS

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ALL_TSV_COLUMNS), delimiter="\t")
        w.writeheader()
        for dp in ids:
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


def _write_csv(path: Path, rows: list[dict[str, str]], *, bom: bool = True) -> Path:
    from hp_corpus.annotation_csv import CSV_COLUMNS

    enc = "utf-8-sig" if bom else "utf-8"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=enc, newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in CSV_COLUMNS})
    return path


def _filled_row(dp: str, **overrides) -> dict[str, str]:
    row = {
        "id": dp,
        "row_hash": "a" * 64,
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
    row.update(overrides)
    return row


def _run_validate(csv_path: Path, master_path: Path):
    mod = _load_script("validate_annotation_csv.py")
    return mod.main([str(csv_path), "--master-tsv", str(master_path)])


def test_template_state_passes_structure_checks(tmp_path: Path) -> None:
    master = _write_master(tmp_path / "master.tsv", ["dp1", "dp2"])
    csv_path = _write_csv(
        tmp_path / "returned.csv",
        [{"id": "dp1", "row_hash": "a" * 64}, {"id": "dp2", "row_hash": "a" * 64}],
    )
    assert _run_validate(csv_path, master) == 0


def test_filled_valid_file_passes(tmp_path: Path) -> None:
    master = _write_master(tmp_path / "master.tsv", ["dp1"])
    csv_path = _write_csv(tmp_path / "returned.csv", [_filled_row("dp1")])
    assert _run_validate(csv_path, master) == 0


def test_bomless_returned_file_accepted(tmp_path: Path) -> None:
    master = _write_master(tmp_path / "master.tsv", ["dp1"])
    csv_path = _write_csv(
        tmp_path / "returned.csv", [_filled_row("dp1")], bom=False
    )
    assert _run_validate(csv_path, master) == 0


def test_header_mismatch_fails(tmp_path: Path) -> None:
    master = _write_master(tmp_path / "master.tsv", ["dp1"])
    p = tmp_path / "bad.csv"
    p.write_text("id,row_hash\nx,y\n", encoding="utf-8-sig")
    assert _run_validate(p, master) == 2


def test_duplicate_ids_fail(tmp_path: Path) -> None:
    master = _write_master(tmp_path / "master.tsv", ["dp1", "dp2"])
    csv_path = _write_csv(
        tmp_path / "returned.csv",
        [{"id": "dp1", "row_hash": "a" * 64}, {"id": "dp1", "row_hash": "a" * 64}],
    )
    assert _run_validate(csv_path, master) == 2


def test_extra_id_not_in_master_fails(tmp_path: Path) -> None:
    master = _write_master(tmp_path / "master.tsv", ["dp1"])
    csv_path = _write_csv(
        tmp_path / "returned.csv",
        [{"id": "dp1", "row_hash": "a" * 64}, _filled_row("dpX")],
    )
    assert _run_validate(csv_path, master) == 2


def test_master_row_missing_fails(tmp_path: Path) -> None:
    master = _write_master(tmp_path / "master.tsv", ["dp1", "dp2"])
    csv_path = _write_csv(
        tmp_path / "returned.csv", [{"id": "dp1", "row_hash": "a" * 64}]
    )
    assert _run_validate(csv_path, master) == 2


def test_row_hash_mismatch_fails(tmp_path: Path) -> None:
    master = _write_master(tmp_path / "master.tsv", ["dp1"])
    csv_path = _write_csv(tmp_path / "returned.csv", [_filled_row("dp1", row_hash="b" * 64)])
    assert _run_validate(csv_path, master) == 2


def test_invalid_de_valid_fails(tmp_path: Path) -> None:
    master = _write_master(tmp_path / "master.tsv", ["dp1"])
    csv_path = _write_csv(tmp_path / "returned.csv", [_filled_row("dp1", de_valid="uncertain")])
    assert _run_validate(csv_path, master) == 2


def test_exclude_without_reason_fails(tmp_path: Path) -> None:
    master = _write_master(tmp_path / "master.tsv", ["dp1"])
    csv_path = _write_csv(
        tmp_path / "returned.csv",
        [_filled_row("dp1", de_valid="exclude", de_exclusion_reason="")],
    )
    assert _run_validate(csv_path, master) == 2


def test_exclude_with_valid_reason_passes(tmp_path: Path) -> None:
    master = _write_master(tmp_path / "master.tsv", ["dp1"])
    csv_path = _write_csv(
        tmp_path / "returned.csv",
        [
            _filled_row(
                "dp1",
                de_valid="exclude",
                de_exclusion_reason="not_target_pp",
                en_counterpart="",
                en_form="",
                en_alignment_confidence="",
                zh_counterpart="",
                zh_form="",
                zh_alignment_confidence="",
            )
        ],
    )
    assert _run_validate(csv_path, master) == 0


def test_include_with_reason_fails(tmp_path: Path) -> None:
    master = _write_master(tmp_path / "master.tsv", ["dp1"])
    csv_path = _write_csv(
        tmp_path / "returned.csv",
        [_filled_row("dp1", de_exclusion_reason="other")],
    )
    assert _run_validate(csv_path, master) == 2


def test_invalid_form_fails(tmp_path: Path) -> None:
    master = _write_master(tmp_path / "master.tsv", ["dp1"])
    csv_path = _write_csv(tmp_path / "returned.csv", [_filled_row("dp1", en_form="wat")])
    assert _run_validate(csv_path, master) == 2


def test_span_without_form_fails(tmp_path: Path) -> None:
    master = _write_master(tmp_path / "master.tsv", ["dp1"])
    csv_path = _write_csv(
        tmp_path / "returned.csv", [_filled_row("dp1", en_form="")]
    )
    assert _run_validate(csv_path, master) == 2


def test_form_without_span_fails(tmp_path: Path) -> None:
    master = _write_master(tmp_path / "master.tsv", ["dp1"])
    csv_path = _write_csv(
        tmp_path / "returned.csv", [_filled_row("dp1", en_counterpart="")]
    )
    assert _run_validate(csv_path, master) == 2


def test_omitted_form_requires_blank_span(tmp_path: Path) -> None:
    master = _write_master(tmp_path / "master.tsv", ["dp1"])
    csv_path = _write_csv(
        tmp_path / "returned.csv",
        [_filled_row("dp1", en_form="omitted")],  # span still filled → error
    )
    assert _run_validate(csv_path, master) == 2
    csv_ok = _write_csv(
        tmp_path / "returned2.csv",
        [
            _filled_row(
                "dp1",
                en_form="omitted",
                en_counterpart="",
                en_alignment_confidence="high",
            )
        ],
    )
    assert _run_validate(csv_ok, master) == 0


def test_uncertain_form_requires_notes(tmp_path: Path) -> None:
    master = _write_master(tmp_path / "master.tsv", ["dp1"])
    bad = _write_csv(
        tmp_path / "returned.csv",
        [_filled_row("dp1", en_form="uncertain", en_counterpart="")],
    )
    assert _run_validate(bad, master) == 2
    good = _write_csv(
        tmp_path / "returned2.csv",
        [_filled_row("dp1", en_form="uncertain", en_counterpart="", en_notes="unclear span")],
    )
    assert _run_validate(good, master) == 0


def test_invalid_alignment_confidence_fails(tmp_path: Path) -> None:
    master = _write_master(tmp_path / "master.tsv", ["dp1"])
    csv_path = _write_csv(
        tmp_path / "returned.csv", [_filled_row("dp1", zh_alignment_confidence="0.93")]
    )
    assert _run_validate(csv_path, master) == 2


def test_stdout_privacy(tmp_path: Path, capsys) -> None:
    master = _write_master(tmp_path / "master.tsv", ["dp1"])
    csv_path = _write_csv(tmp_path / "returned.csv", [_filled_row("dp1")])
    assert _run_validate(csv_path, master) == 0
    out = capsys.readouterr().out
    for leaked in ("house", "房子", "dp1"):
        assert leaked not in out
    assert "template_state: False" in out
