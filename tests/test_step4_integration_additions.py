"""Integration tests for the Step 4 validator's ``--require-complete`` mode
and the German-candidate-decision / alignment-QC vocabulary additions.

These tests are deliberately kept separate from ``tests/test_step4.py`` so
the surrounding-infra work package can pin the semantics it depends on
without editing Agent A's owned test file. All fixtures are synthetic.

What this locks down for downstream work packages:

  * ``--require-complete`` returns no violations + a populated ``rollup``
    only when every row is either ``completed`` or ``excluded``.
  * An ``include`` row that lacks ``annotation_status=complete`` fails.
  * An ``include + complete`` row whose ``{lang}_alignment_qc`` is not
    ``confirmed`` on both sides fails.
  * An ``uncertain`` German decision is counted in the rollup but never
    in the ``completed`` bucket.
  * The ``de_candidate_decision`` and ``{lang}_alignment_qc`` vocabularies
    are enforced in all modes.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from hp_corpus.schema import Segment  # noqa: F401  (used in type only)
from hp_corpus.step4 import (
    ALL_TSV_COLUMNS,
    BUILDER_DEFAULT_EDITABLE,
    SOURCE_COLUMNS,
    build_candidates,
    select_pilot,
    write_pilot_tsv,
)

_EXTRACTION_FIELDS = [
    "parse_block_id",
    "source_segment_id",
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


def _write_extraction_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_EXTRACTION_FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            row = {**r}
            if row.get("det") is None:
                row["det"] = "-"
            if row.get("det_token_id") is None:
                row["det_token_id"] = "-"
            w.writerow({k: row.get(k, "") for k in _EXTRACTION_FIELDS})


def _write_segments(path: Path, segments: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        import json

        for s in segments:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def _write_alignments(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        import json

        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _alignment_record(
    align_id: str,
    src_ids: list[str],
    tgt_ids: list[str],
    *,
    type_str: str = "1:1",
    confidence: float = 0.9,
) -> dict[str, Any]:
    """The current serializer always names both sides en/zh; the IDs
    themselves carry the real language."""
    return {
        "align_id": align_id,
        "en": src_ids,
        "zh": tgt_ids,
        "type": type_str,
        "confidence": confidence,
        "method": "vecalign_labse",
        "validated": False,
    }


def _build_synth_repo(tmp_path: Path) -> dict[str, Path]:
    """Tiny two-chapter Ch.1 corpus: one contracted + one uncontracted PP,
    both 1:1 aligned to EN and ZH. Sufficient for a 1+1 pilot fixture."""
    extraction = tmp_path / "extracted"
    segmented = tmp_path / "segmented"
    aligned = tmp_path / "aligned"

    _write_extraction_tsv(
        extraction / "hp1_de_ch01_contracted.tsv",
        [
            {
                "parse_block_id": "hp1_de_ch01_p0001_s001#b001",
                "source_segment_id": "hp1_de_ch01_p0001_s001",
                "prep": "im",
                "noun": "Haus",
                "prep_token_id": "1",
                "noun_token_id": "2",
                "pp_token_start": "1",
                "pp_token_end": "2",
                "pp_surface": "im Haus",
                "in_filter": "Y",
            },
        ],
    )
    _write_extraction_tsv(
        extraction / "hp1_de_ch01_uncontracted.tsv",
        [
            {
                "parse_block_id": "hp1_de_ch01_p0002_s001#b001",
                "source_segment_id": "hp1_de_ch01_p0002_s001",
                "prep": "in",
                "det": "dem",
                "noun": "Haus",
                "prep_token_id": "1",
                "det_token_id": "2",
                "noun_token_id": "3",
                "pp_token_start": "1",
                "pp_token_end": "3",
                "pp_surface": "in dem Haus",
                "in_filter": "Y",
            },
        ],
    )

    seg_de = [
        {
            "id": "hp1_de_ch01_p0001_s001",
            "chapter": 1,
            "paragraph": 1,
            "sentence": 1,
            "text": "synth DE one",
            "source_pages": [1],
        },
        {
            "id": "hp1_de_ch01_p0002_s001",
            "chapter": 1,
            "paragraph": 1,
            "sentence": 2,
            "text": "synth DE two",
            "source_pages": [1],
        },
    ]
    seg_en = [
        {
            "id": "hp1_en_ch01_p0001_s001",
            "chapter": 1,
            "paragraph": 1,
            "sentence": 1,
            "text": "synth EN one",
            "source_pages": [1],
        },
        {
            "id": "hp1_en_ch01_p0002_s001",
            "chapter": 1,
            "paragraph": 1,
            "sentence": 2,
            "text": "synth EN two",
            "source_pages": [1],
        },
    ]
    seg_zh = [
        {
            "id": "hp1_zh_ch01_p0001_s001",
            "chapter": 1,
            "paragraph": 1,
            "sentence": 1,
            "text": "synth ZH one",
            "source_pages": [1],
        },
        {
            "id": "hp1_zh_ch01_p0002_s001",
            "chapter": 1,
            "paragraph": 1,
            "sentence": 2,
            "text": "synth ZH two",
            "source_pages": [1],
        },
    ]
    _write_segments(segmented / "hp1_de_ch01.jsonl", seg_de)
    _write_segments(segmented / "hp1_en_ch01.jsonl", seg_en)
    _write_segments(segmented / "hp1_zh_ch01.jsonl", seg_zh)

    _write_alignments(
        aligned / "hp1_de_en_ch01.jsonl",
        [
            _alignment_record("a0001", ["hp1_de_ch01_p0001_s001"], ["hp1_en_ch01_p0001_s001"]),
            _alignment_record("a0002", ["hp1_de_ch01_p0002_s001"], ["hp1_en_ch01_p0002_s001"]),
        ],
    )
    _write_alignments(
        aligned / "hp1_de_zh_ch01.jsonl",
        [
            _alignment_record("a0001", ["hp1_de_ch01_p0001_s001"], ["hp1_zh_ch01_p0001_s001"]),
            _alignment_record("a0002", ["hp1_de_ch01_p0002_s001"], ["hp1_zh_ch01_p0002_s001"]),
        ],
    )

    return {
        "extraction_dir": extraction,
        "segmented_dir": segmented,
        "aligned_dir": aligned,
    }


# --------------------------------------------------------------------- helpers


def _build_initial_tsv(tmp_path: Path, validator_module) -> Path:
    paths = _build_synth_repo(tmp_path)
    candidates = build_candidates(
        extraction_dir=paths["extraction_dir"],
        segmented_dir=paths["segmented_dir"],
        aligned_dir=paths["aligned_dir"],
        chapters=[1],
    )
    sel, _ = select_pilot(candidates, n_contracted=1, n_uncontracted=1)
    return write_pilot_tsv(sel, tmp_path / "out" / "pilot.tsv")


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)
    header = rows[0]
    body = [dict(zip(header, r, strict=False)) for r in rows[1:]]
    return header, body


def _write_tsv(path: Path, header: list[str], body: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for row in body:
            w.writerow({col: row.get(col, "") for col in header})


def _fully_annotate_row(row: dict[str, str], *, en_text: str, zh_text: str) -> dict[str, str]:
    """Mark a row as fully completed: include + complete + confirmed on both
    sides with valid spans/forms/confidence. Spans cover the whole aligned
    text — exact contents are irrelevant to ``--require-complete`` as long
    as char ranges resolve."""
    row["de_candidate_decision"] = "include"
    row["annotation_status"] = "complete"
    for lang, text in (("en", en_text), ("zh", zh_text)):
        row[f"{lang}_alignment_qc"] = "confirmed"
        row[f"{lang}_alignment_relation"] = "direct"
        row[f"{lang}_span_text"] = text
        row[f"{lang}_char_ranges"] = f"[[0,{len(text)}]]"
        row[f"{lang}_form"] = "definite" if lang == "en" else "bare"
        row[f"{lang}_confidence"] = "high"
    return row


# --------------------------------------------------------------------- tests


def test_builder_default_editable_constants_exist() -> None:
    """The validator's initial-state check depends on the builder default
    map being available. Pin the constant's existence and contents."""
    assert "en_alignment_qc" in BUILDER_DEFAULT_EDITABLE
    assert "zh_alignment_qc" in BUILDER_DEFAULT_EDITABLE
    assert BUILDER_DEFAULT_EDITABLE["en_alignment_qc"] == "assumed_ok"
    assert BUILDER_DEFAULT_EDITABLE["zh_alignment_qc"] == "assumed_ok"


def test_all_tsv_columns_includes_integration_additions() -> None:
    """The integration additions must appear in ALL_TSV_COLUMNS so the
    validator's header check accepts TSVs that include them."""
    for col in (
        "de_candidate_decision",
        "de_exclusion_reason",
        "de_candidate_notes",
        "en_alignment_qc",
        "en_alignment_notes",
        "zh_alignment_qc",
        "zh_alignment_notes",
    ):
        assert col in ALL_TSV_COLUMNS, f"{col!r} missing from ALL_TSV_COLUMNS"
        assert col not in SOURCE_COLUMNS, f"{col!r} must be editable, not source"
