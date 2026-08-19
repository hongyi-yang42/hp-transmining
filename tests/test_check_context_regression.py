"""Synthetic-fixture tests for the retrieval-context regression gate."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_script():
    path = REPO_ROOT / "scripts" / "check_context_regression.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_manifest(path: Path, cases: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    return path


def _write_master(path: Path, rows: list[dict[str, str]]) -> Path:
    from hp_corpus.step4 import ALL_TSV_COLUMNS

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ALL_TSV_COLUMNS), delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in ALL_TSV_COLUMNS})
    return path


def _row(dp: str, **overrides) -> dict[str, str]:
    row = {
        "datapoint_id": dp,
        "en_context_ids": '["hp1_en_ch01_p0002_s001"]',
        "en_context_provenance": "anchor_window",
        "zh_context_ids": '["hp1_zh_ch01_p0002_s001"]',
        "zh_context_provenance": "neighbor_fallback",
    }
    row.update(overrides)
    return row


def test_locus_in_context_passes(tmp_path: Path, capsys) -> None:
    mod = _load_script()
    manifest = _write_manifest(
        tmp_path / "reg.json", {"dp1": {"en": ["hp1_en_ch01_p0002_s001"]}}
    )
    master = _write_master(tmp_path / "master.tsv", [_row("dp1")])
    assert mod.main(["--manifest", str(manifest), "--master-tsv", str(master)]) == 0
    out = capsys.readouterr().out
    assert "in-context: 1" in out
    assert "violations: 0" in out


def test_locus_outside_without_manual_review_fails(tmp_path: Path, capsys) -> None:
    mod = _load_script()
    manifest = _write_manifest(
        tmp_path / "reg.json", {"dp1": {"zh": ["hp1_zh_ch01_p0009_s001"]}}
    )
    master = _write_master(tmp_path / "master.tsv", [_row("dp1")])
    assert mod.main(["--manifest", str(manifest), "--master-tsv", str(master)]) == 2
    assert "LOCUS_NOT_COVERED" in capsys.readouterr().out


def test_locus_outside_with_manual_review_passes(tmp_path: Path, capsys) -> None:
    mod = _load_script()
    manifest = _write_manifest(
        tmp_path / "reg.json", {"dp1": {"zh": ["hp1_zh_ch01_p0009_s001"]}}
    )
    master = _write_master(
        tmp_path / "master.tsv", [_row("dp1", zh_context_provenance="manual_review")]
    )
    assert mod.main(["--manifest", str(manifest), "--master-tsv", str(master)]) == 0
    assert "manual_review: 1" in capsys.readouterr().out


def test_case_missing_from_master_fails(tmp_path: Path) -> None:
    mod = _load_script()
    manifest = _write_manifest(tmp_path / "reg.json", {"dpX": {"en": ["hp1_en_x"]}})
    master = _write_master(tmp_path / "master.tsv", [_row("dp1")])
    assert mod.main(["--manifest", str(manifest), "--master-tsv", str(master)]) == 2


def test_missing_files_fail_closed(tmp_path: Path) -> None:
    mod = _load_script()
    manifest = _write_manifest(tmp_path / "reg.json", {"dp1": {"en": ["hp1_en_x"]}})
    assert mod.main(["--manifest", str(manifest), "--master-tsv", str(tmp_path / "no.tsv")]) == 2
    no_json = tmp_path / "no.json"
    assert mod.main(["--manifest", str(no_json), "--master-tsv", str(tmp_path / "no.tsv")]) == 2
