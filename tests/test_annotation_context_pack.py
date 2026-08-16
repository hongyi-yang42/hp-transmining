"""Synthetic-fixture tests for the generic annotation context-pack builder.

Every fixture here uses invented, non-novel text — no Harry Potter
content. The tests mirror the style of ``tests/test_step4.py``: tiny
synthetic corpora written to ``tmp_path`` that exercise each documented
behaviour of the builder.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

from hp_corpus.annotation_workflow import (
    OUTPUT_COLUMNS,
    REQUIRED_INPUT_COLUMNS,
    build_index,
    context_around,
    join_text,
    load_segments_chapter,
)
from hp_corpus.schema import Segment


def _mk_segs(lang: str, texts: list[str]) -> list[Segment]:
    """Build a list of Segment objects for testing. Sentence N gets
    page 1, sentence N, paragraph 1."""
    return [
        Segment(
            id=f"hp1_{lang}_ch01_p0001_s{n:03d}",
            chapter=1,
            paragraph=1,
            sentence=n,
            text=t,
            source_pages=[1],
        )
        for n, t in enumerate(texts, start=1)
    ]

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "build_annotation_context_pack.py"
)


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_annotation_context_pack", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------- helpers


def _write_segments(path: Path, segments: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in segments:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def _seg(
    lang: str, chapter: int, page: int, sentence: int, text: str
) -> dict[str, Any]:
    """Build a minimal Segment dict matching the schema pattern."""
    sid = f"hp1_{lang}_ch{chapter:02d}_p{page:04d}_s{sentence:03d}"
    return {
        "id": sid,
        "chapter": chapter,
        "paragraph": 1,
        "sentence": sentence,
        "text": text,
        "source_pages": [page],
    }


def _write_input_tsv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    drop_columns: tuple[str, ...] = (),
) -> None:
    """Write an input TSV with the required columns. ``drop_columns``
    lets tests simulate a missing-column failure."""
    cols = [c for c in REQUIRED_INPUT_COLUMNS if c not in drop_columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            row_out = {c: r.get(c, "") for c in cols}
            # Serialize list-valued fields as JSON.
            for list_col in ("en_sentence_ids", "zh_sentence_ids"):
                if list_col in row_out and isinstance(row_out[list_col], list):
                    row_out[list_col] = json.dumps(row_out[list_col], ensure_ascii=False)
            w.writerow(row_out)


def _build_synth_repo(tmp_path: Path) -> dict[str, Path]:
    """Tiny Ch.1-only synthetic corpus with 5 sentences per language."""
    segmented = tmp_path / "segmented"
    for lang in ("de", "en", "zh"):
        segs = [_seg(lang, 1, 1, n, f"synth {lang.upper()} {n}") for n in range(1, 6)]
        _write_segments(segmented / f"hp1_{lang}_ch01.jsonl", segs)
    return {"segmented_dir": segmented}


def _row(
    *,
    datapoint_id: str | None = None,
    chapter: int = 1,
    de_source_segment_id: str = "hp1_de_ch01_p0001_s001",
    de_parse_block_id: str | None = None,
    de_form: str = "contracted",
    de_token_start: int = 1,
    de_token_end: int = 2,
    de_sentence_text: str = "synth DE one",
    en_sentence_ids: list[str] | None = None,
    en_aligned_text: str = "synth EN one",
    zh_sentence_ids: list[str] | None = None,
    zh_aligned_text: str = "synth ZH one",
    en_alignment_cardinality: str = "1:1",
    en_alignment_confidence: str = "0.88",
    zh_alignment_cardinality: str = "1:1",
    zh_alignment_confidence: str = "0.82",
    source_row_sha256: str = "deadbeef",
) -> dict[str, Any]:
    """Build an input row for the context-pack builder. The parse block
    defaults to the first block (``#b001``) of the source segment, and the
    datapoint_id defaults to the migrated id embedding that block."""
    if de_parse_block_id is None:
        de_parse_block_id = f"{de_source_segment_id}#b001"
    if datapoint_id is None:
        datapoint_id = (
            f"dp_ch{chapter:02d}_{de_parse_block_id}_t{de_token_start}-{de_token_end}"
        )
    return {
        "datapoint_id": datapoint_id,
        "chapter": chapter,
        "de_parse_block_id": de_parse_block_id,
        "de_source_segment_id": de_source_segment_id,
        "de_form": de_form,
        "de_token_start": de_token_start,
        "de_token_end": de_token_end,
        "de_sentence_text": de_sentence_text,
        "en_sentence_ids": en_sentence_ids or ["hp1_en_ch01_p0001_s001"],
        "en_aligned_text": en_aligned_text,
        "zh_sentence_ids": zh_sentence_ids or ["hp1_zh_ch01_p0001_s001"],
        "zh_aligned_text": zh_aligned_text,
        "en_alignment_cardinality": en_alignment_cardinality,
        "en_alignment_confidence": en_alignment_confidence,
        "zh_alignment_cardinality": zh_alignment_cardinality,
        "zh_alignment_confidence": zh_alignment_confidence,
        "source_row_sha256": source_row_sha256,
    }


# --------------------------------------------------------------------- pure-function tests


def test_load_segments_chapter_returns_empty_when_missing(tmp_path: Path) -> None:
    out = load_segments_chapter("de", 1, tmp_path / "missing")
    assert out == []


def test_load_segments_chapter_reads_synthetic(tmp_path: Path) -> None:
    d = tmp_path / "segmented"
    _write_segments(
        d / "hp1_de_ch01.jsonl",
        [_seg("de", 1, 1, 1, "synth DE one"), _seg("de", 1, 1, 2, "synth DE two")],
    )
    segs = load_segments_chapter("de", 1, d)
    assert [s.id for s in segs] == [
        "hp1_de_ch01_p0001_s001",
        "hp1_de_ch01_p0001_s002",
    ]


def test_build_index_positions_match_file_order() -> None:
    segs = _mk_segs("de", ["t1", "t2", "t3"])
    idx = build_index(segs)
    assert idx["hp1_de_ch01_p0001_s001"] == 0
    assert idx["hp1_de_ch01_p0001_s002"] == 1
    assert idx["hp1_de_ch01_p0001_s003"] == 2


def test_join_text_joins_with_slash() -> None:
    segs = _mk_segs("en", ["EN1", "EN2", "EN3"])
    out = join_text(segs, ["hp1_en_ch01_p0001_s001", "hp1_en_ch01_p0001_s003"])
    assert out == "EN1 / EN3"


def test_context_around_default_size_one() -> None:
    segs = _mk_segs("en", ["EN1", "EN2", "EN3", "EN4", "EN5"])
    prev, curr, nxt = context_around(segs, ["hp1_en_ch01_p0001_s003"], 1)
    assert prev == "EN2"
    assert curr == "EN3"
    assert nxt == "EN4"


def test_context_around_size_two_fetches_two_each_side() -> None:
    segs = _mk_segs("en", ["EN1", "EN2", "EN3", "EN4", "EN5", "EN6"])
    prev, curr, nxt = context_around(segs, ["hp1_en_ch01_p0001_s004"], 2)
    assert prev == "EN2 / EN3"
    assert curr == "EN4"
    assert nxt == "EN5 / EN6"


def test_context_around_stops_at_chapter_start() -> None:
    """No cross-chapter fetch — the earliest sentence has no predecessor."""
    segs = _mk_segs("en", ["EN1", "EN2"])
    prev, curr, nxt = context_around(segs, ["hp1_en_ch01_p0001_s001"], 2)
    assert prev == ""
    assert curr == "EN1"
    assert nxt == "EN2"


def test_context_around_stops_at_chapter_end() -> None:
    segs = _mk_segs("en", ["EN1", "EN2"])
    prev, curr, nxt = context_around(segs, ["hp1_en_ch01_p0001_s002"], 2)
    assert prev == "EN1"
    assert curr == "EN2"
    assert nxt == ""


def test_context_around_multi_sentence_uses_min_max_positions() -> None:
    """When ids are non-contiguous, prev is before the earliest and
    next is after the latest."""
    segs = _mk_segs("en", ["EN1", "EN2", "EN3", "EN4", "EN5"])
    prev, curr, nxt = context_around(
        segs, ["hp1_en_ch01_p0001_s002", "hp1_en_ch01_p0001_s004"], 1
    )
    assert prev == "EN1"
    assert curr == "EN2 / EN4"
    assert nxt == "EN5"


# --------------------------------------------------------------------- CLI tests


def _run(builder_module, argv: list[str]) -> int:
    return builder_module.main(argv)


def test_cli_happy_path_two_rows(tmp_path: Path, capsys) -> None:
    builder = _load_builder()
    repo = _build_synth_repo(tmp_path)
    input_tsv = tmp_path / "in.tsv"
    out = tmp_path / "out" / "pack.tsv"
    _write_input_tsv(
        input_tsv,
        [
            _row(de_source_segment_id="hp1_de_ch01_p0001_s002",
                 en_sentence_ids=["hp1_en_ch01_p0001_s002"],
                 zh_sentence_ids=["hp1_zh_ch01_p0001_s002"]),
            _row(datapoint_id="dp_ch01_hp1_de_ch01_p0001_s003#b001_t1-2",
                 de_source_segment_id="hp1_de_ch01_p0001_s003",
                 de_form="uncontracted",
                 en_sentence_ids=["hp1_en_ch01_p0001_s003"],
                 en_aligned_text="synth EN three",
                 zh_sentence_ids=["hp1_zh_ch01_p0001_s003"],
                 zh_aligned_text="synth ZH three"),
        ],
    )
    rc = _run(
        builder,
        [
            "--input-tsv", str(input_tsv),
            "--segmented-dir", str(repo["segmented_dir"]),
            "--output", str(out),
        ],
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert out.exists()
    rows = list(csv.DictReader(out.open(encoding="utf-8"), delimiter="\t"))
    assert len(rows) == 2
    # All expected output columns present.
    for col in OUTPUT_COLUMNS:
        assert col in rows[0]
    # Context populated correctly for row 1 (s002 with size 1).
    r0 = rows[0]
    assert r0["de_text"] == "«synth DE 2»"
    assert r0["de_context_prev"] == "synth DE 1"
    assert r0["de_context_next"] == "synth DE 3"
    assert r0["en_text"] == "synth EN 2"
    assert r0["en_context_prev"] == "synth EN 1"
    assert r0["en_context_next"] == "synth EN 3"
    assert r0["zh_text"] == "synth ZH 2"
    assert r0["zh_context_prev"] == "synth ZH 1"
    assert r0["zh_context_next"] == "synth ZH 3"
    # The 6 trailing human-check columns are blank.
    for col in (
        "en_scene_match", "zh_scene_match",
        "en_counterpart_locatable", "zh_counterpart_locatable",
        "alignment_issue", "review_notes",
    ):
        assert r0[col] == ""
    # Stdout aggregate lines.
    assert "rows: 2" in captured.out
    assert "by_chapter:" in captured.out
    assert "by_form:" in captured.out
    assert f"output: {out}" in captured.out
    assert "6 blank human-check columns" in captured.out


def test_cli_missing_required_column_exits_2(tmp_path: Path, capsys) -> None:
    builder = _load_builder()
    repo = _build_synth_repo(tmp_path)
    input_tsv = tmp_path / "in.tsv"
    out = tmp_path / "out.tsv"
    _write_input_tsv(
        input_tsv, [_row()], drop_columns=("source_row_sha256",)
    )
    rc = _run(
        builder,
        [
            "--input-tsv", str(input_tsv),
            "--segmented-dir", str(repo["segmented_dir"]),
            "--output", str(out),
        ],
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "MISSING_REQUIRED_COLUMN" in captured.err
    assert "source_row_sha256" in captured.err
    # Output must not be created on failure.
    assert not out.exists()


def test_cli_malformed_de_sentence_id_exits_2(tmp_path: Path, capsys) -> None:
    builder = _load_builder()
    repo = _build_synth_repo(tmp_path)
    input_tsv = tmp_path / "in.tsv"
    out = tmp_path / "out.tsv"
    _write_input_tsv(
        input_tsv,
        [_row(de_source_segment_id="not-a-real-segment-id")],
    )
    rc = _run(
        builder,
        [
            "--input-tsv", str(input_tsv),
            "--segmented-dir", str(repo["segmented_dir"]),
            "--output", str(out),
        ],
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "MALFORMED_DE_SENTENCE_ID" in captured.err
    # row index printed (0-indexed)
    assert "row=0" in captured.err


def test_cli_malformed_de_parse_block_id_exits_2(tmp_path: Path, capsys) -> None:
    """A de_parse_block_id without the #bNNN block marker must fail
    closed — non-empty is not good enough."""
    builder = _load_builder()
    repo = _build_synth_repo(tmp_path)
    input_tsv = tmp_path / "in.tsv"
    out = tmp_path / "out.tsv"
    _write_input_tsv(
        input_tsv,
        [_row(de_parse_block_id="hp1_de_ch01_p0001_s001")],
    )
    rc = _run(
        builder,
        [
            "--input-tsv", str(input_tsv),
            "--segmented-dir", str(repo["segmented_dir"]),
            "--output", str(out),
        ],
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "MALFORMED_DE_PARSE_BLOCK_ID" in captured.err
    assert "row=0" in captured.err


def test_cli_de_provenance_mismatch_exits_2(tmp_path: Path, capsys) -> None:
    """A well-formed block id that extends a DIFFERENT segment than the
    row's de_source_segment_id must fail closed (silent mis-join risk)."""
    builder = _load_builder()
    repo = _build_synth_repo(tmp_path)
    input_tsv = tmp_path / "in.tsv"
    out = tmp_path / "out.tsv"
    _write_input_tsv(
        input_tsv,
        [
            _row(
                de_source_segment_id="hp1_de_ch01_p0001_s002",
                de_parse_block_id="hp1_de_ch01_p0001_s003#b001",
            )
        ],
    )
    rc = _run(
        builder,
        [
            "--input-tsv", str(input_tsv),
            "--segmented-dir", str(repo["segmented_dir"]),
            "--output", str(out),
        ],
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "DE_PROVENANCE_MISMATCH" in captured.err
    assert "row=0" in captured.err


def test_cli_unresolved_segment_id_exits_2(tmp_path: Path, capsys) -> None:
    builder = _load_builder()
    repo = _build_synth_repo(tmp_path)
    input_tsv = tmp_path / "in.tsv"
    out = tmp_path / "out.tsv"
    _write_input_tsv(
        input_tsv,
        [
            _row(
                de_source_segment_id="hp1_de_ch01_p0001_s002",
                # EN id not present in synth JSONL (which has s001..s005)
                # — but use a malformed-pattern id to be safe.
                en_sentence_ids=["hp1_en_ch01_p9999_s999"],
            )
        ],
    )
    rc = _run(
        builder,
        [
            "--input-tsv", str(input_tsv),
            "--segmented-dir", str(repo["segmented_dir"]),
            "--output", str(out),
        ],
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "UNRESOLVED_SEGMENT_ID" in captured.err
    # Never print the offending segment id.
    assert "hp1_en_ch01_p9999_s999" not in captured.err
    assert "hp1_en_ch01_p9999_s999" not in captured.out


def test_cli_malformed_en_sentence_ids_json_exits_2(tmp_path: Path, capsys) -> None:
    builder = _load_builder()
    repo = _build_synth_repo(tmp_path)
    input_tsv = tmp_path / "in.tsv"
    out = tmp_path / "out.tsv"
    # Build a row, then surgically break the en_sentence_ids cell.
    _write_input_tsv(input_tsv, [_row()])
    lines = input_tsv.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    body = lines[1].split("\t")
    en_idx = header.index("en_sentence_ids")
    body[en_idx] = "[not, valid, json"  # malformed
    input_tsv.write_text(
        lines[0] + "\n" + "\t".join(body) + "\n", encoding="utf-8"
    )
    rc = _run(
        builder,
        [
            "--input-tsv", str(input_tsv),
            "--segmented-dir", str(repo["segmented_dir"]),
            "--output", str(out),
        ],
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "MALFORMED_SENTENCE_IDS_JSON" in captured.err


def test_cli_context_size_two_fetches_two_sentences_each_side(
    tmp_path: Path, capsys
) -> None:
    builder = _load_builder()
    repo = _build_synth_repo(tmp_path)
    input_tsv = tmp_path / "in.tsv"
    out = tmp_path / "out.tsv"
    _write_input_tsv(
        input_tsv,
        [
            _row(
                de_source_segment_id="hp1_de_ch01_p0001_s003",
                en_sentence_ids=["hp1_en_ch01_p0001_s003"],
                zh_sentence_ids=["hp1_zh_ch01_p0001_s003"],
            )
        ],
    )
    rc = _run(
        builder,
        [
            "--input-tsv", str(input_tsv),
            "--segmented-dir", str(repo["segmented_dir"]),
            "--output", str(out),
            "--context-size", "2",
        ],
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    rows = list(csv.DictReader(out.open(encoding="utf-8"), delimiter="\t"))
    r = rows[0]
    assert r["de_context_prev"] == "synth DE 1 / synth DE 2"
    assert r["de_context_next"] == "synth DE 4 / synth DE 5"
    assert r["en_context_prev"] == "synth EN 1 / synth EN 2"
    assert r["en_context_next"] == "synth EN 4 / synth EN 5"
    assert r["zh_context_prev"] == "synth ZH 1 / synth ZH 2"
    assert r["zh_context_next"] == "synth ZH 4 / synth ZH 5"


def test_cli_no_cross_chapter_context_at_end(tmp_path: Path, capsys) -> None:
    """Asking for more context than the chapter has must not crash and
    must not pull from outside the file."""
    builder = _load_builder()
    repo = _build_synth_repo(tmp_path)
    input_tsv = tmp_path / "in.tsv"
    out = tmp_path / "out.tsv"
    _write_input_tsv(
        input_tsv,
        [
            _row(
                de_source_segment_id="hp1_de_ch01_p0001_s005",
                en_sentence_ids=["hp1_en_ch01_p0001_s005"],
                zh_sentence_ids=["hp1_zh_ch01_p0001_s005"],
            )
        ],
    )
    rc = _run(
        builder,
        [
            "--input-tsv", str(input_tsv),
            "--segmented-dir", str(repo["segmented_dir"]),
            "--output", str(out),
            "--context-size", "5",  # larger than the file
        ],
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    rows = list(csv.DictReader(out.open(encoding="utf-8"), delimiter="\t"))
    r = rows[0]
    # prev contains all 4 preceding sentences; next is empty.
    assert r["de_context_prev"] == " / ".join(f"synth DE {n}" for n in range(1, 5))
    assert r["de_context_next"] == ""
    assert r["en_context_next"] == ""
    assert r["zh_context_next"] == ""


def test_cli_multi_sentence_alignment_joined_correctly(
    tmp_path: Path, capsys
) -> None:
    builder = _load_builder()
    repo = _build_synth_repo(tmp_path)
    input_tsv = tmp_path / "in.tsv"
    out = tmp_path / "out.tsv"
    _write_input_tsv(
        input_tsv,
        [
            _row(
                de_source_segment_id="hp1_de_ch01_p0001_s002",
                en_sentence_ids=[
                    "hp1_en_ch01_p0001_s002",
                    "hp1_en_ch01_p0001_s003",
                ],
                en_aligned_text="synth EN two synth EN three",
                en_alignment_cardinality="1:2",
                zh_sentence_ids=["hp1_zh_ch01_p0001_s002"],
            )
        ],
    )
    rc = _run(
        builder,
        [
            "--input-tsv", str(input_tsv),
            "--segmented-dir", str(repo["segmented_dir"]),
            "--output", str(out),
        ],
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    rows = list(csv.DictReader(out.open(encoding="utf-8"), delimiter="\t"))
    r = rows[0]
    assert r["en_text"] == "synth EN 2 / synth EN 3"
    assert r["en_cardinality"] == "1:2"
    # DE curr still gets the «» marker, even though EN doesn't.
    assert r["de_text"] == "«synth DE 2»"
    # EN context: prev = sentence before s002 = s001;
    #             next = sentence after s003 = s004
    assert r["en_context_prev"] == "synth EN 1"
    assert r["en_context_next"] == "synth EN 4"


def test_cli_refuses_overwrite_without_force(tmp_path: Path, capsys) -> None:
    builder = _load_builder()
    repo = _build_synth_repo(tmp_path)
    input_tsv = tmp_path / "in.tsv"
    out = tmp_path / "out.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("PREEXISTING\n", encoding="utf-8")
    _write_input_tsv(input_tsv, [_row()])
    rc = _run(
        builder,
        [
            "--input-tsv", str(input_tsv),
            "--segmented-dir", str(repo["segmented_dir"]),
            "--output", str(out),
        ],
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "OUTPUT_EXISTS" in captured.err
    # Pre-existing content untouched.
    assert out.read_text(encoding="utf-8") == "PREEXISTING\n"

    # With --force-output, the build proceeds.
    rc2 = _run(
        builder,
        [
            "--input-tsv", str(input_tsv),
            "--segmented-dir", str(repo["segmented_dir"]),
            "--output", str(out),
            "--force-output",
        ],
    )
    assert rc2 == 0
    assert "PREEXISTING" not in out.read_text(encoding="utf-8")


def test_cli_stdout_carries_no_sentence_text(tmp_path: Path, capsys) -> None:
    """The builder must NEVER print fixture sentence text on stdout.
    This guard test catches any future regression that leaks text via
    an error message, a debug print, or an aggregate line."""
    builder = _load_builder()
    repo = _build_synth_repo(tmp_path)
    input_tsv = tmp_path / "in.tsv"
    out = tmp_path / "out.tsv"
    _write_input_tsv(input_tsv, [_row()])
    rc = _run(
        builder,
        [
            "--input-tsv", str(input_tsv),
            "--segmented-dir", str(repo["segmented_dir"]),
            "--output", str(out),
        ],
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    forbidden = [
        "synth DE",
        "synth EN",
        "synth ZH",
        "synth_de",
        "synth_en",
        "synth_zh",
    ]
    for token in forbidden:
        assert token not in captured.out, f"stdout leaked: {token!r}"
        assert token not in captured.err, f"stderr leaked: {token!r}"


def test_cli_missing_input_tsv_exits_2(tmp_path: Path, capsys) -> None:
    builder = _load_builder()
    rc = _run(
        builder,
        [
            "--input-tsv", str(tmp_path / "nope.tsv"),
            "--segmented-dir", str(tmp_path / "segmented"),
            "--output", str(tmp_path / "out.tsv"),
        ],
    )
    captured = capsys.readouterr()
    assert rc == 2
    assert "INPUT_TSV_MISSING" in captured.err


def test_cli_output_directory_created(tmp_path: Path, capsys) -> None:
    """A nested non-existent output parent dir must be created."""
    builder = _load_builder()
    repo = _build_synth_repo(tmp_path)
    input_tsv = tmp_path / "in.tsv"
    out = tmp_path / "deep" / "nested" / "out.tsv"
    _write_input_tsv(input_tsv, [_row()])
    rc = _run(
        builder,
        [
            "--input-tsv", str(input_tsv),
            "--segmented-dir", str(repo["segmented_dir"]),
            "--output", str(out),
        ],
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert out.exists()
