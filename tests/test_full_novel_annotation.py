"""Synthetic-fixture tests for the full-novel annotation master builder.

Covers ``scripts/build_full_novel_annotation.py`` and the ``zero_hits_ok``
extension to ``hp_corpus.step4.build_candidates``:

* Manifest gate: missing manifest, non-ready statuses (``missing_input``,
  no entry) fail closed without writing a master.
* A ``zero_hits_ok`` (chapter, form) pair with a header-only TSV is
  accepted; a zero-byte file never is; a header-only TSV WITHOUT the
  manifest blessing still fails (Ch.1-3 behaviour unchanged).
* Master rows carry the machine source columns + ``source_row_sha256``
  and blank human-editable columns (except the builder QC default).
* Stdout privacy: no fixture surface forms or segment IDs on stdout.

All fixtures are synthetic; no novel text.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_full_novel_annotation.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_full_novel_annotation", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_extraction_tsv(path: Path, rows: list[dict[str, object]] | None) -> None:
    """rows=None → header-only TSV (the zero_hits_ok shape)."""
    fields = [
        "parse_block_id", "source_segment_id", "prep", "det", "noun",
        "prep_token_id", "det_token_id", "noun_token_id", "pp_token_start",
        "pp_token_end", "pp_surface", "in_filter",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows or []:
            row = {**r}
            if row.get("det") is None:
                row["det"] = "-"
            if row.get("det_token_id") is None:
                row["det_token_id"] = "-"
            w.writerow({k: row.get(k, "") for k in fields})


def _write_segments(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, sid in enumerate(ids, 1):
            lang = sid.split("_")[1]
            f.write(json.dumps({
                "id": sid,
                "chapter": int(sid.split("_ch")[1][:2]),
                "paragraph": 1,
                "sentence": i,
                "text": f"synth {lang.upper()} {i}",
                "source_pages": [1],
            }, ensure_ascii=False) + "\n")


def _write_alignment(path: Path, de_ids: list[str], other_lang: str, other_ids: list[str]) -> None:
    """One 1:1 record per sentence pair. Both sides are written under the
    legacy ``en``/``zh`` keys as ID lists; step4 identifies the real
    language by inspecting the segment IDs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, (d, o) in enumerate(zip(de_ids, other_ids, strict=True), 1):
            f.write(json.dumps({
                "align_id": f"a{i:03d}",
                "en": [d] if other_lang == "en" else [o],
                "zh": [o] if other_lang == "en" else [d],
                "type": "1:1",
                "confidence": 0.9,
                "method": "synthetic",
                "validated": False,
            }, ensure_ascii=False) + "\n")


def _write_manifest(path: Path, entries: list[tuple[int, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([
            {
                "chapter": ch, "form": form, "status": status,
                "row_count": 0, "parsed_path_sha256": "x" * 64,
                "parsed_path_size": 1, "parsed_path_sentence_count": 1,
                "extractor_version": "synthetic",
            }
            for ch, form, status in entries
        ], f)


def _build_synth_repo(
    tmp_path: Path,
    *,
    ch2_uncontracted_rows: list[dict[str, object]] | None = None,
    ch2_uncontracted_status: str = "zero_hits_ok",
) -> dict[str, Path]:
    """Two synthetic chapters. Ch.2 uncontracted defaults to a legitimately
    header-only form (manifest zero_hits_ok)."""
    extraction = tmp_path / "extracted"
    segmented = tmp_path / "segmented"
    aligned = tmp_path / "aligned"

    _write_extraction_tsv(
        extraction / "hp1_de_ch01_contracted.tsv",
        [{"parse_block_id": "hp1_de_ch01_p0001_s001#b001",
          "source_segment_id": "hp1_de_ch01_p0001_s001", "prep": "im", "noun": "Haus",
          "prep_token_id": "1", "noun_token_id": "2", "pp_token_start": "1",
          "pp_token_end": "2", "pp_surface": "im Haus", "in_filter": "Y"}],
    )
    _write_extraction_tsv(
        extraction / "hp1_de_ch01_uncontracted.tsv",
        [{"parse_block_id": "hp1_de_ch01_p0002_s001#b001",
          "source_segment_id": "hp1_de_ch01_p0002_s001", "prep": "in", "det": "dem",
          "noun": "Wald", "prep_token_id": "1", "det_token_id": "2",
          "noun_token_id": "3", "pp_token_start": "1", "pp_token_end": "3",
          "pp_surface": "in dem Wald", "in_filter": "Y"}],
    )
    _write_extraction_tsv(
        extraction / "hp1_de_ch02_contracted.tsv",
        [{"parse_block_id": "hp1_de_ch02_p0001_s001#b001",
          "source_segment_id": "hp1_de_ch02_p0001_s001", "prep": "am", "noun": "Tag",
          "prep_token_id": "1", "noun_token_id": "2", "pp_token_start": "1",
          "pp_token_end": "2", "pp_surface": "am Tag", "in_filter": "N"}],
    )
    _write_extraction_tsv(extraction / "hp1_de_ch02_uncontracted.tsv", ch2_uncontracted_rows)

    ch1_de = ["hp1_de_ch01_p0001_s001", "hp1_de_ch01_p0002_s001"]
    ch2_de = ["hp1_de_ch02_p0001_s001"]
    _write_segments(segmented / "hp1_de_ch01.jsonl", ch1_de)
    _write_segments(segmented / "hp1_de_ch02.jsonl", ch2_de)
    _write_segments(
        segmented / "hp1_en_ch01.jsonl", ["hp1_en_ch01_p0001_s001", "hp1_en_ch01_p0002_s001"]
    )
    _write_segments(segmented / "hp1_en_ch02.jsonl", ["hp1_en_ch02_p0001_s001"])
    _write_segments(
        segmented / "hp1_zh_ch01.jsonl", ["hp1_zh_ch01_p0001_s001", "hp1_zh_ch01_p0002_s001"]
    )
    _write_segments(segmented / "hp1_zh_ch02.jsonl", ["hp1_zh_ch02_p0001_s001"])

    _write_alignment(aligned / "hp1_de_en_ch01.jsonl", ch1_de, "en",
                     ["hp1_en_ch01_p0001_s001", "hp1_en_ch01_p0002_s001"])
    _write_alignment(aligned / "hp1_de_zh_ch01.jsonl", ch1_de, "zh",
                     ["hp1_zh_ch01_p0001_s001", "hp1_zh_ch01_p0002_s001"])
    _write_alignment(aligned / "hp1_de_en_ch02.jsonl", ch2_de, "en", ["hp1_en_ch02_p0001_s001"])
    _write_alignment(aligned / "hp1_de_zh_ch02.jsonl", ch2_de, "zh", ["hp1_zh_ch02_p0001_s001"])

    _write_manifest(extraction / "manifest.json", [
        (1, "contracted", "ok"),
        (1, "uncontracted", "ok"),
        (2, "contracted", "ok"),
        (2, "uncontracted", ch2_uncontracted_status),
    ])
    return {"extraction": extraction, "segmented": segmented, "aligned": aligned}


def _run(script, tmp_path: Path, paths: dict[str, Path], chapters: list[int]):
    return script.main([
        "--extraction-dir", str(paths["extraction"]),
        "--manifest", str(paths["extraction"] / "manifest.json"),
        "--segmented-dir", str(paths["segmented"]),
        "--aligned-dir", str(paths["aligned"]),
        "--output-dir", str(tmp_path / "out"),
        "--chapters", *[str(c) for c in chapters],
    ])


def _read_master(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# --------------------------------------------------------------------- tests


def test_missing_manifest_fails_closed(tmp_path: Path) -> None:
    script = _load_script()
    rc = script.main([
        "--extraction-dir", str(tmp_path / "nowhere"),
        "--segmented-dir", str(tmp_path / "s"),
        "--aligned-dir", str(tmp_path / "a"),
        "--output-dir", str(tmp_path / "out"),
        "--chapters", "1",
    ])
    assert rc == 2


def test_missing_input_status_fails_closed(tmp_path: Path, capsys) -> None:
    script = _load_script()
    paths = _build_synth_repo(tmp_path)
    # degrade ch2 contracted to missing_input in the manifest
    _write_manifest(paths["extraction"] / "manifest.json", [
        (1, "contracted", "ok"),
        (1, "uncontracted", "ok"),
        (2, "contracted", "missing_input"),
        (2, "uncontracted", "zero_hits_ok"),
    ])
    rc = _run(script, tmp_path, paths, [1, 2])
    assert rc == 2
    assert "EXTRACTION_NOT_READY" in capsys.readouterr().err
    assert not (tmp_path / "out" / "full_novel_annotation_master.tsv").exists()


def test_zero_hits_form_header_only_accepted(tmp_path: Path, capsys) -> None:
    script = _load_script()
    paths = _build_synth_repo(tmp_path)  # ch2 uncontracted header-only + blessed
    rc = _run(script, tmp_path, paths, [1, 2])
    assert rc == 0
    out = capsys.readouterr().out
    master = tmp_path / "out" / "full_novel_annotation_master.tsv"
    rows = _read_master(master)
    # ch1 im/in + ch2 am → am→an is in the shared inventory, so 3 rows
    assert len(rows) == 3
    assert {r["chapter"] for r in rows} == {"1", "2"}
    assert "zero_hits_ok (chapter, form) pairs: 1" in out


def test_header_only_without_manifest_blessing_fails(tmp_path: Path) -> None:
    script = _load_script()
    # header-only TSV but manifest claims "ok" → must fail closed
    paths = _build_synth_repo(tmp_path, ch2_uncontracted_rows=None, ch2_uncontracted_status="ok")
    rc = _run(script, tmp_path, paths, [1, 2])
    assert rc == 2


def test_zero_byte_tsv_fails_even_when_blessed(tmp_path: Path) -> None:
    script = _load_script()
    paths = _build_synth_repo(tmp_path)
    (paths["extraction"] / "hp1_de_ch02_uncontracted.tsv").write_text("", encoding="utf-8")
    rc = _run(script, tmp_path, paths, [1, 2])
    assert rc == 2


def test_master_source_hashed_editable_blank(tmp_path: Path) -> None:
    script = _load_script()
    paths = _build_synth_repo(tmp_path)
    rc = _run(script, tmp_path, paths, [1, 2])
    assert rc == 0
    rows = _read_master(tmp_path / "out" / "full_novel_annotation_master.tsv")
    from hp_corpus.step4 import BUILDER_DEFAULT_EDITABLE, EDITABLE_COLUMNS
    for r in rows:
        assert r["dataset_scope"] == "full_novel_machine_candidate_pool"
        assert r["de_candidate_decision"] == ""
        assert r["en_form"] == ""
        assert r["zh_form"] == ""
        assert r["annotator"] == ""
        assert len(r["source_row_sha256"]) == 64
        for col in EDITABLE_COLUMNS:
            expected = BUILDER_DEFAULT_EDITABLE.get(col, "")
            assert r[col] == expected, col


def test_stdout_privacy_no_fixture_text(tmp_path: Path, capsys) -> None:
    script = _load_script()
    paths = _build_synth_repo(tmp_path)
    rc = _run(script, tmp_path, paths, [1, 2])
    assert rc == 0
    out = capsys.readouterr().out
    for leak in ("im Haus", "in dem Wald", "am Tag", "Haus", "Wald",
                 "hp1_de_ch01_p0001_s001", "dp_ch01"):
        assert leak not in out, leak


def test_existing_output_requires_force(tmp_path: Path) -> None:
    script = _load_script()
    paths = _build_synth_repo(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "full_novel_annotation_master.tsv").write_text("sentinel", encoding="utf-8")
    rc = _run(script, tmp_path, paths, [1, 2])
    assert rc == 2
    assert (out_dir / "full_novel_annotation_master.tsv").read_text(encoding="utf-8") == "sentinel"
