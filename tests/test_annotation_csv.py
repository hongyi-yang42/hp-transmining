"""Synthetic-fixture tests for the annotator-facing pair CSV builder.

Covers the schema, encoding (BOM), quoting round-trip with awkward
characters, blank annotator columns, fail-closed rules, and stdout
privacy. All fixture text is invented; no novel text.
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


def _master_row(dp: str, **overrides) -> dict[str, str]:
    row = {
        "datapoint_id": dp,
        "chapter": "3",
        "de_form": "contracted",
        "de_pp_surface": 'im, "großen" Haus',
        "de_head_lemma": "Haus",
        "de_sentence_text": 'Er ging, wie er sagte, ins "große" Haus.',
        "en_aligned_text": 'He went, as he said, into the "big" house.',
        "en_context_ids": "hp1_en_ch03_p0002_s004|hp1_en_ch03_p0002_s005",
        "en_context_text": (
            'Prev line. He went, as he said, into the "big" house. Next line. 后续中文，逗号。'
        ),
        "zh_aligned_text": "他说着，走进了那幢大房子。",
        "zh_context_ids": "hp1_zh_ch03_p0002_s004|hp1_zh_ch03_p0002_s005",
        "zh_context_text": "上一句。他说着，走进了那幢大房子。下一句。",
        "source_row_sha256": "a" * 64,
    }
    row.update(overrides)
    return row


def _write_master(path: Path, rows: list[dict[str, str]]) -> Path:
    from hp_corpus.step4 import ALL_TSV_COLUMNS

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ALL_TSV_COLUMNS), delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in ALL_TSV_COLUMNS})
    return path


def _build(tmp_path: Path, rows: list[dict[str, str]]):
    mod = _load_script("build_annotation_csv.py")
    master = _write_master(tmp_path / "master.tsv", rows)
    out = tmp_path / "annotation_pairs.csv"
    rc = mod.main(["--master-tsv", str(master), "--output", str(out)])
    return mod, out, rc


def test_builds_csv_with_schema_blanks_and_bom(tmp_path: Path) -> None:
    from hp_corpus.annotation_csv import ANNOTATOR_COLUMNS, CSV_COLUMNS, MACHINE_COLUMNS

    mod, out, rc = _build(
        tmp_path,
        [_master_row("dp1"), _master_row("dp2", de_form="uncontracted")],
    )
    assert rc == 0
    raw = out.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # Excel-friendly BOM
    with open(out, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == list(CSV_COLUMNS)
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["id"] == "dp1"
    assert rows[0]["german_pp"] == 'im, "großen" Haus'
    # The CSV carries the retrieval view (anchor ± window), not the bare
    # aligned anchor text.
    assert rows[0]["english_context"].startswith("Prev line.")
    assert rows[0]["english_context"].endswith("后续中文，逗号。")
    assert "他说着，走进了那幢大房子。" in rows[0]["chinese_context"]
    assert rows[0]["chinese_context"].startswith("上一句。")
    assert rows[0]["row_hash"] == "a" * 64
    # Annotator columns all blank.
    for r in rows:
        for col in ANNOTATOR_COLUMNS:
            assert r[col] == ""
    # Machine columns all filled.
    for r in rows:
        for col in MACHINE_COLUMNS:
            assert r[col] != ""
    # Commas/quotes survive the round-trip inside quoted fields.
    assert ', "großen" Haus' in rows[0]["german_pp"]


def test_stdout_privacy(tmp_path: Path, capsys) -> None:
    _build(tmp_path, [_master_row("dp1")])
    out = capsys.readouterr().out
    for leaked in ("Haus", "房子", "dp1", "großen"):
        assert leaked not in out
    assert "rows: 1" in out
    assert "columns:" in out


def test_fail_closed_on_missing_master(tmp_path: Path) -> None:
    mod = _load_script("build_annotation_csv.py")
    rc = mod.main(
        ["--master-tsv", str(tmp_path / "absent.tsv"), "--output", str(tmp_path / "o.csv")]
    )
    assert rc == 2


def test_fail_closed_on_duplicate_master_ids(tmp_path: Path) -> None:
    mod, out, rc = _build(tmp_path, [_master_row("dp1"), _master_row("dp1")])
    assert rc == 2
    assert not out.exists()


def test_fail_closed_on_missing_master_columns(tmp_path: Path) -> None:
    master = tmp_path / "master.tsv"
    master.parent.mkdir(parents=True, exist_ok=True)
    master.write_text("datapoint_id\tchapter\nx\t1\n", encoding="utf-8")
    mod = _load_script("build_annotation_csv.py")
    rc = mod.main(["--master-tsv", str(master), "--output", str(tmp_path / "o.csv")])
    assert rc == 2


def test_fail_closed_on_blank_machine_cell(tmp_path: Path) -> None:
    mod, out, rc = _build(tmp_path, [_master_row("dp1", de_sentence_text="")])
    assert rc == 2
    assert not out.exists()


def test_blank_context_allowed_for_anchorless_rows(tmp_path: Path) -> None:
    """An empty context column is legitimate when the DE sentence has no
    aligned anchor on that side (annotator marks it not_aligned) — the
    builder must not fail-closed on it, but other machine cells stay
    strict."""
    from hp_corpus.annotation_csv import MACHINE_COLUMNS

    mod, out, rc = _build(tmp_path, [_master_row("dp1", en_context_text="", zh_context_text="")])
    assert rc == 0
    with open(out, encoding="utf-8-sig", newline="") as f:
        row = next(csv.DictReader(f))
    assert row["english_context"] == "" and row["chinese_context"] == ""
    # Every other machine column still non-blank.
    strict = [c for c in MACHINE_COLUMNS if c not in ("english_context", "chinese_context")]
    assert all(row[c] != "" for c in strict)

    # A blank non-context machine cell still fails closed.
    mod, out2, rc2 = _build(tmp_path / "b", [_master_row("dp1", de_pp_surface="")])
    assert rc2 == 2
    assert not out2.exists()


def test_refuses_overwrite_without_force(tmp_path: Path) -> None:
    mod, out, rc = _build(tmp_path, [_master_row("dp1")])
    assert rc == 0
    master = tmp_path / "master.tsv"
    rc2 = mod.main(["--master-tsv", str(master), "--output", str(out)])
    assert rc2 == 2
