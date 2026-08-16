"""Synthetic-fixture tests for src/hp_corpus/step4.py and the Step 4 validator.

Every fixture here uses invented, non-novel text — no Harry Potter content,
no filter-list lemmas. Tests cover the methodology boundaries enumerated
in docs/METHODS.md.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from hp_corpus.step4 import (
    CONTRACTED_PREP_NORMALIZATION,
    EDITABLE_COLUMNS,
    PAPER_SHARED_PREPOSITIONS,
    SOURCE_COLUMNS,
    InsufficientCandidatesError,
    build_candidates,
    compute_source_row_sha256,
    lang_from_segment_id,
    normalize_contracted_prep,
    select_paper_sample,
    select_pilot,
    summarize_candidates,
    write_candidates_jsonl,
    write_pilot_tsv,
)


def _write_extraction_tsv(path: Path, kind: str, rows: list[dict[str, Any]]) -> None:
    """Write an extraction TSV in the same format as the migrated extractor."""
    fields = [
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


def _write_segments(path: Path, segments: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in segments:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def _write_alignments(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _alignment_record(
    align_id: str,
    src_ids: list[str],
    tgt_ids: list[str],
    *,
    type_str: str = "1:1",
    confidence: float = 0.9,
    src_key: str = "en",  # legacy serializer always wrote "en" + "zh"
    tgt_key: str = "zh",
) -> dict[str, Any]:
    """Mimic the production alignment serializer: both fields always named
    en/zh, but the IDs themselves carry the real language."""
    return (
        {
            "align_id": align_id,
            "en": src_ids,
            "zh": tgt_ids,
            "type": type_str,
            "confidence": confidence,
            "method": "vecalign_labse",
            "validated": False,
        }
        if (src_key == "en" and tgt_key == "zh")
        else {
            "align_id": align_id,
            src_key: src_ids,
            tgt_key: tgt_ids,
            "type": type_str,
            "confidence": confidence,
            "method": "vecalign_labse",
            "validated": False,
        }
    )


def _build_synth_repo(tmp_path: Path) -> dict[str, Path]:
    """Set up a tiny Ch.1-only synthetic corpus that exercises the
    important edge cases. Returns a dict of useful paths."""
    extraction = tmp_path / "extracted"
    segmented = tmp_path / "segmented"
    aligned = tmp_path / "aligned"

    # German extraction — two sentences, both forms, one minimal-pair group
    # ("in|Haus"), plus a same-(prep,noun) duplicate in s001 to test that
    # occurrences stay distinguishable. The parser split s001 into two
    # parse blocks (#b001 / #b002), one PP per block.
    _write_extraction_tsv(
        extraction / "hp1_de_ch01_contracted.tsv",
        "contracted",
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
            {
                # Same (prep, noun) again in the segment's second parse
                # block — must NOT be deduped.
                "parse_block_id": "hp1_de_ch01_p0001_s001#b002",
                "source_segment_id": "hp1_de_ch01_p0001_s001",
                "prep": "im",
                "noun": "Haus",
                "prep_token_id": "5",
                "noun_token_id": "6",
                "pp_token_start": "5",
                "pp_token_end": "6",
                "pp_surface": "im Haus",
                "in_filter": "Y",
            },
            {
                "parse_block_id": "hp1_de_ch01_p0002_s001#b001",
                "source_segment_id": "hp1_de_ch01_p0002_s001",
                "prep": "im",
                "noun": "Wald",
                "prep_token_id": "1",
                "noun_token_id": "2",
                "pp_token_start": "1",
                "pp_token_end": "2",
                "pp_surface": "im Wald",
                "in_filter": "N",
            },
        ],
    )
    _write_extraction_tsv(
        extraction / "hp1_de_ch01_uncontracted.tsv",
        "uncontracted",
        [
            {
                "parse_block_id": "hp1_de_ch01_p0003_s001#b001",
                "source_segment_id": "hp1_de_ch01_p0003_s001",
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
            {
                # Different group; will be eligible only via stable-fill.
                "parse_block_id": "hp1_de_ch01_p0004_s001#b001",
                "source_segment_id": "hp1_de_ch01_p0004_s001",
                "prep": "an",
                "det": "dem",
                "noun": "Baum",
                "prep_token_id": "1",
                "det_token_id": "2",
                "noun_token_id": "3",
                "pp_token_start": "1",
                "pp_token_end": "3",
                "pp_surface": "an dem Baum",
                "in_filter": "Y",
            },
        ],
    )

    # Segments — three sentences each in DE/EN/ZH, all 1:1 mapped.
    _write_segments(
        segmented / "hp1_de_ch01.jsonl",
        [
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
            {
                "id": "hp1_de_ch01_p0003_s001",
                "chapter": 1,
                "paragraph": 1,
                "sentence": 3,
                "text": "synth DE three",
                "source_pages": [1],
            },
            {
                "id": "hp1_de_ch01_p0004_s001",
                "chapter": 1,
                "paragraph": 1,
                "sentence": 4,
                "text": "synth DE four",
                "source_pages": [1],
            },
            {
                "id": "hp1_de_ch01_p0005_s001",
                "chapter": 1,
                "paragraph": 1,
                "sentence": 5,
                "text": "synth DE five",
                "source_pages": [1],
            },
        ],
    )
    _write_segments(
        segmented / "hp1_en_ch01.jsonl",
        [
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
            {
                "id": "hp1_en_ch01_p0003_s001",
                "chapter": 1,
                "paragraph": 1,
                "sentence": 3,
                "text": "synth EN three",
                "source_pages": [1],
            },
            {
                "id": "hp1_en_ch01_p0004_s001",
                "chapter": 1,
                "paragraph": 1,
                "sentence": 4,
                "text": "synth EN four",
                "source_pages": [1],
            },
            {
                "id": "hp1_en_ch01_p0005_s001",
                "chapter": 1,
                "paragraph": 1,
                "sentence": 5,
                "text": "synth EN five",
                "source_pages": [1],
            },
        ],
    )
    _write_segments(
        segmented / "hp1_zh_ch01.jsonl",
        [
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
            {
                "id": "hp1_zh_ch01_p0003_s001",
                "chapter": 1,
                "paragraph": 1,
                "sentence": 3,
                "text": "synth ZH three",
                "source_pages": [1],
            },
            {
                "id": "hp1_zh_ch01_p0004_s001",
                "chapter": 1,
                "paragraph": 1,
                "sentence": 4,
                "text": "synth ZH four",
                "source_pages": [1],
            },
            {
                "id": "hp1_zh_ch01_p0005_s001",
                "chapter": 1,
                "paragraph": 1,
                "sentence": 5,
                "text": "synth ZH five",
                "source_pages": [1],
            },
        ],
    )

    # Alignments — DE↔EN: 1:1 mapping for s001..s004, s005 missing from EN side.
    de_en = [
        _alignment_record(
            "a0001", ["hp1_de_ch01_p0001_s001"], ["hp1_en_ch01_p0001_s001"], type_str="1:1"
        ),
        _alignment_record(
            "a0002", ["hp1_de_ch01_p0002_s001"], ["hp1_en_ch01_p0002_s001"], type_str="1:1"
        ),
        _alignment_record(
            "a0003", ["hp1_de_ch01_p0003_s001"], ["hp1_en_ch01_p0003_s001"], type_str="1:1"
        ),
        _alignment_record(
            "a0004", ["hp1_de_ch01_p0004_s001"], ["hp1_en_ch01_p0004_s001"], type_str="1:1"
        ),
        # s005 deliberately unaligned to EN — exercises 'missing' status.
    ]
    _write_alignments(aligned / "hp1_de_en_ch01.jsonl", de_en)

    # DE↔ZH: 1:2 record for s001 to exercise multi-sentence alignment.
    de_zh = [
        _alignment_record(
            "a0001",
            ["hp1_de_ch01_p0001_s001"],
            ["hp1_zh_ch01_p0001_s001", "hp1_zh_ch01_p0002_s001"],
            type_str="1:2",
        ),
        _alignment_record(
            "a0002", ["hp1_de_ch01_p0002_s001"], ["hp1_zh_ch01_p0003_s001"], type_str="1:1"
        ),
        _alignment_record(
            "a0003", ["hp1_de_ch01_p0003_s001"], ["hp1_zh_ch01_p0004_s001"], type_str="1:1"
        ),
        _alignment_record(
            "a0004", ["hp1_de_ch01_p0004_s001"], ["hp1_zh_ch01_p0005_s001"], type_str="1:1"
        ),
    ]
    _write_alignments(aligned / "hp1_de_zh_ch01.jsonl", de_zh)

    return {
        "extraction_dir": extraction,
        "segmented_dir": segmented,
        "aligned_dir": aligned,
    }


# --------------------------------------------------------------------- builder


def test_lang_from_segment_id() -> None:
    assert lang_from_segment_id("hp1_de_ch01_p0001_s001") == "de"
    assert lang_from_segment_id("hp1_en_ch01_p0001_s001") == "en"
    assert lang_from_segment_id("hp1_zh_ch01_p0001_s001") == "zh"
    assert lang_from_segment_id("garbage") is None


def test_normalize_contracted_prep() -> None:
    assert normalize_contracted_prep("im") == "in"
    assert normalize_contracted_prep("ins") == "in"
    assert normalize_contracted_prep("am") == "an"
    assert normalize_contracted_prep("ans") == "an"
    assert normalize_contracted_prep("zum") == "zu"
    assert normalize_contracted_prep("zur") == "zu"
    assert normalize_contracted_prep("vom") == "von"
    assert normalize_contracted_prep("beim") == "bei"
    assert normalize_contracted_prep("in") == "in"  # pass-through


def test_two_occurrences_same_prep_noun_distinguishable(tmp_path: Path) -> None:
    paths = _build_synth_repo(tmp_path)
    candidates = build_candidates(
        extraction_dir=paths["extraction_dir"],
        segmented_dir=paths["segmented_dir"],
        aligned_dir=paths["aligned_dir"],
        chapters=[1],
    )
    # Filter to s001 contracted candidates — there are 2 with same (prep, noun).
    s001 = [
        c
        for c in candidates
        if c["de_source_segment_id"] == "hp1_de_ch01_p0001_s001" and c["de_form"] == "contracted"
    ]
    assert len(s001) == 2
    assert s001[0]["de_head_lemma"] == s001[1]["de_head_lemma"] == "Haus"
    # Distinct datapoint_ids and token coords.
    assert s001[0]["datapoint_id"] != s001[1]["datapoint_id"]
    assert {c["de_token_start"] for c in s001} == {1, 5}
    assert {c["de_token_end"] for c in s001} == {2, 6}


def test_contracted_uncontracted_form_same_minimal_pair_group(tmp_path: Path) -> None:
    paths = _build_synth_repo(tmp_path)
    candidates = build_candidates(
        extraction_dir=paths["extraction_dir"],
        segmented_dir=paths["segmented_dir"],
        aligned_dir=paths["aligned_dir"],
        chapters=[1],
    )
    by_form = {"contracted": [], "uncontracted": []}
    for c in candidates:
        by_form[c["de_form"]].append(c)
    # im Haus (contracted) and in dem Haus (uncontracted) both normalize to in|Haus.
    contracted_in_haus = [
        c
        for c in by_form["contracted"]
        if c["de_prep_normalized"] == "in" and c["de_head_lemma"] == "Haus"
    ]
    uncontracted_in_haus = [
        c
        for c in by_form["uncontracted"]
        if c["de_prep_normalized"] == "in" and c["de_head_lemma"] == "Haus"
    ]
    assert contracted_in_haus and uncontracted_in_haus
    assert contracted_in_haus[0]["minimal_pair_group"] == "in|Haus"
    assert uncontracted_in_haus[0]["minimal_pair_group"] == "in|Haus"


def test_one_to_one_alignment_carried_through(tmp_path: Path) -> None:
    paths = _build_synth_repo(tmp_path)
    candidates = build_candidates(
        extraction_dir=paths["extraction_dir"],
        segmented_dir=paths["segmented_dir"],
        aligned_dir=paths["aligned_dir"],
        chapters=[1],
    )
    # p0003_s001 uncontracted "in dem Haus" is 1:1 in both EN and ZH.
    cand = next(
        c
        for c in candidates
        if c["de_source_segment_id"] == "hp1_de_ch01_p0003_s001" and c["de_form"] == "uncontracted"
    )
    assert cand["en_alignment_cardinality"] == "1:1"
    assert cand["zh_alignment_cardinality"] == "1:1"
    assert cand["en_alignment_status"] == "aligned"
    assert cand["zh_alignment_status"] == "aligned"
    assert cand["en_aligned_text"] == "synth EN three"
    assert cand["zh_aligned_text"] == "synth ZH four"


def test_one_to_two_alignment_preserved(tmp_path: Path) -> None:
    paths = _build_synth_repo(tmp_path)
    candidates = build_candidates(
        extraction_dir=paths["extraction_dir"],
        segmented_dir=paths["segmented_dir"],
        aligned_dir=paths["aligned_dir"],
        chapters=[1],
    )
    # s001 ZH side is 1:2 — both ZH sentences must appear in the candidate.
    cand = next(
        c
        for c in candidates
        if c["de_source_segment_id"] == "hp1_de_ch01_p0001_s001" and c["de_form"] == "contracted"
    )
    assert cand["zh_alignment_cardinality"] == "1:2"
    assert cand["zh_sentence_ids"] == [
        "hp1_zh_ch01_p0001_s001",
        "hp1_zh_ch01_p0002_s001",
    ]
    # Both target sentences joined with single space.
    assert cand["zh_aligned_text"] == "synth ZH one synth ZH two"


def test_legacy_en_zh_keys_decoded_by_segment_id(tmp_path: Path) -> None:
    """DE↔EN file has 'en' field with DE IDs and 'zh' field with EN IDs —
    the language must be detected from the segment ID, not the JSON key."""
    paths = _build_synth_repo(tmp_path)
    candidates = build_candidates(
        extraction_dir=paths["extraction_dir"],
        segmented_dir=paths["segmented_dir"],
        aligned_dir=paths["aligned_dir"],
        chapters=[1],
    )
    # Sanity: every candidate has its DE sentence id, and at least one EN/ZH
    # target sentence id with the right language prefix (when status=aligned).
    for c in candidates:
        assert c["de_source_segment_id"].startswith("hp1_de_")
        if c["en_alignment_status"] == "aligned":
            for sid in c["en_sentence_ids"]:
                assert sid.startswith("hp1_en_"), sid
        if c["zh_alignment_status"] == "aligned":
            for sid in c["zh_sentence_ids"]:
                assert sid.startswith("hp1_zh_"), sid


def test_missing_alignment_status_flagged(tmp_path: Path) -> None:
    paths = _build_synth_repo(tmp_path)
    candidates = build_candidates(
        extraction_dir=paths["extraction_dir"],
        segmented_dir=paths["segmented_dir"],
        aligned_dir=paths["aligned_dir"],
        chapters=[1],
    )
    # p0004_s001 uncontracted "an dem Baum" — its DE sentence is aligned, but
    # we deliberately set up the corpus so p0005_s001 has NO EN alignment.
    # Add a candidate pointing at p0005 to test 'missing' status.
    # (Use a synthetic extraction row pointing at p0005 to exercise missing.)
    extraction = paths["extraction_dir"]
    _write_extraction_tsv(
        extraction / "hp1_de_ch01_contracted.tsv",
        "contracted",
        [
            {
                "parse_block_id": "hp1_de_ch01_p0005_s001#b001",
                "source_segment_id": "hp1_de_ch01_p0005_s001",
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
    candidates = build_candidates(
        extraction_dir=paths["extraction_dir"],
        segmented_dir=paths["segmented_dir"],
        aligned_dir=paths["aligned_dir"],
        chapters=[1],
    )
    missing = [c for c in candidates if c["de_source_segment_id"] == "hp1_de_ch01_p0005_s001"]
    assert missing, "expected one candidate at p0005_s001"
    assert missing[0]["en_alignment_status"] == "missing"
    # ZH side also doesn't have p0005 in our fixture.
    assert missing[0]["zh_alignment_status"] == "missing"


def test_duplicate_de_alignment_record_raises(tmp_path: Path) -> None:
    paths = _build_synth_repo(tmp_path)
    # Append a second record pointing at the same DE sentence id.
    de_en = (paths["aligned_dir"] / "hp1_de_en_ch01.jsonl").read_text(encoding="utf-8")
    duplicate = _alignment_record(
        "a9999",
        ["hp1_de_ch01_p0001_s001"],
        ["hp1_en_ch01_p0001_s001"],
        type_str="1:1",
    )
    with open(paths["aligned_dir"] / "hp1_de_en_ch01.jsonl", "w", encoding="utf-8") as f:
        f.write(de_en)
        f.write(json.dumps(duplicate, ensure_ascii=False) + "\n")
    with pytest.raises(ValueError, match="appears in two alignment records"):
        build_candidates(
            extraction_dir=paths["extraction_dir"],
            segmented_dir=paths["segmented_dir"],
            aligned_dir=paths["aligned_dir"],
            chapters=[1],
        )


def test_deterministic_pilot_selection(tmp_path: Path) -> None:
    paths = _build_synth_repo(tmp_path)
    candidates = build_candidates(
        extraction_dir=paths["extraction_dir"],
        segmented_dir=paths["segmented_dir"],
        aligned_dir=paths["aligned_dir"],
        chapters=[1],
    )
    sel1, _ = select_pilot(candidates, n_contracted=2, n_uncontracted=2)
    sel2, _ = select_pilot(candidates, n_contracted=2, n_uncontracted=2)
    ids1 = [c["datapoint_id"] for c in sel1]
    ids2 = [c["datapoint_id"] for c in sel2]
    assert ids1 == ids2


def test_pilot_priority_minimal_pair_first(tmp_path: Path) -> None:
    """The 'in|Haus' minimal pair should be selected before author_resource_match
    fills from unrelated groups."""
    paths = _build_synth_repo(tmp_path)
    candidates = build_candidates(
        extraction_dir=paths["extraction_dir"],
        segmented_dir=paths["segmented_dir"],
        aligned_dir=paths["aligned_dir"],
        chapters=[1],
    )
    sel, summary = select_pilot(candidates, n_contracted=1, n_uncontracted=1)
    # The chosen contracted should be 'im Haus' from group in|Haus.
    assert sel[0]["minimal_pair_group"] == "in|Haus"
    assert sel[0]["de_form"] == "contracted"
    assert sel[1]["minimal_pair_group"] == "in|Haus"
    assert sel[1]["de_form"] == "uncontracted"
    assert summary["by_reason"]["minimal_pair"] == 2


def test_pilot_insufficient_raises(tmp_path: Path) -> None:
    paths = _build_synth_repo(tmp_path)
    candidates = build_candidates(
        extraction_dir=paths["extraction_dir"],
        segmented_dir=paths["segmented_dir"],
        aligned_dir=paths["aligned_dir"],
        chapters=[1],
    )
    with pytest.raises(InsufficientCandidatesError):
        select_pilot(candidates, n_contracted=99, n_uncontracted=99)


def test_pilot_excludes_unaligned_candidates(tmp_path: Path) -> None:
    """Candidates without both EN and ZH alignments must not enter the pilot."""
    paths = _build_synth_repo(tmp_path)
    # Rewrite contracted TSV to point one occurrence at p0005 (missing both).
    _write_extraction_tsv(
        paths["extraction_dir"] / "hp1_de_ch01_contracted.tsv",
        "contracted",
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
            {
                "parse_block_id": "hp1_de_ch01_p0005_s001#b001",  # missing alignment
                "source_segment_id": "hp1_de_ch01_p0005_s001",
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
    candidates = build_candidates(
        extraction_dir=paths["extraction_dir"],
        segmented_dir=paths["segmented_dir"],
        aligned_dir=paths["aligned_dir"],
        chapters=[1],
    )
    sel, summary = select_pilot(candidates, n_contracted=1, n_uncontracted=1)
    # The p0005 candidate (missing alignment) must NOT appear.
    assert all(c["de_source_segment_id"] != "hp1_de_ch01_p0005_s001" for c in sel)


def test_source_row_sha256_changes_when_source_edited(tmp_path: Path) -> None:
    paths = _build_synth_repo(tmp_path)
    candidates = build_candidates(
        extraction_dir=paths["extraction_dir"],
        segmented_dir=paths["segmented_dir"],
        aligned_dir=paths["aligned_dir"],
        chapters=[1],
    )
    cand = candidates[0]
    h1 = compute_source_row_sha256(cand)
    # Edit a source column.
    cand_edited = dict(cand)
    cand_edited["de_head_lemma"] = "EDITED"
    h2 = compute_source_row_sha256(cand_edited)
    assert h1 != h2


def test_candidates_jsonl_roundtrip(tmp_path: Path) -> None:
    paths = _build_synth_repo(tmp_path)
    candidates = build_candidates(
        extraction_dir=paths["extraction_dir"],
        segmented_dir=paths["segmented_dir"],
        aligned_dir=paths["aligned_dir"],
        chapters=[1],
    )
    out = write_candidates_jsonl(candidates, tmp_path / "out" / "c.jsonl")
    reloaded = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(reloaded) == len(candidates)
    # datapoint_ids match
    original_ids = {c["datapoint_id"] for c in candidates}
    reloaded_ids = {c["datapoint_id"] for c in reloaded}
    assert original_ids == reloaded_ids


def test_pilot_tsv_has_full_column_set_and_initial_blanks(tmp_path: Path) -> None:
    paths = _build_synth_repo(tmp_path)
    candidates = build_candidates(
        extraction_dir=paths["extraction_dir"],
        segmented_dir=paths["segmented_dir"],
        aligned_dir=paths["aligned_dir"],
        chapters=[1],
    )
    sel, _ = select_pilot(candidates, n_contracted=1, n_uncontracted=1)
    out = write_pilot_tsv(sel, tmp_path / "out" / "pilot.tsv")
    rows = list(csv.DictReader(out.open(encoding="utf-8"), delimiter="\t"))
    assert len(rows) == 2
    # All SOURCE_COLUMNS present in header
    for col in SOURCE_COLUMNS:
        assert col in rows[0]
    # All EDITABLE_COLUMNS present. Initial-state semantics: every editable
    # column is at its builder default (blank for most, "assumed_ok" for the
    # two alignment_qc columns).
    from hp_corpus.step4 import BUILDER_DEFAULT_EDITABLE

    for col in EDITABLE_COLUMNS:
        assert col in rows[0]
        assert rows[0][col] == BUILDER_DEFAULT_EDITABLE.get(col, ""), col


def test_summary_no_token_text(tmp_path: Path) -> None:
    paths = _build_synth_repo(tmp_path)
    candidates = build_candidates(
        extraction_dir=paths["extraction_dir"],
        segmented_dir=paths["segmented_dir"],
        aligned_dir=paths["aligned_dir"],
        chapters=[1],
    )
    sel, pilot_summary = select_pilot(candidates, n_contracted=1, n_uncontracted=1)
    summary = summarize_candidates(candidates, sel, pilot_summary)
    blob = json.dumps(summary)
    # No surface forms, lemmas, or sentence text from the corpus.
    assert "synth DE" not in blob
    assert "synth EN" not in blob
    assert "synth ZH" not in blob
    assert "Haus" not in blob and "Wald" not in blob


# --------------------------------------------------------------------- validator


def _write_initial_pilot_tsv(tmp_path: Path, validator_module) -> Path:
    """Build a tiny pilot TSV with one contracted + one uncontracted row,
    all editable cells blank. Used for validator structural tests."""
    # The validator's PILOT_DEFAULT_N_CONTRACTED/UNCONTRACTED is 10, but we
    # can use the production writer to make a 2-row TSV, then temporarily
    # patch the validator's expected count to 1+1 to keep tests small.
    paths = _build_synth_repo(tmp_path)
    candidates = build_candidates(
        extraction_dir=paths["extraction_dir"],
        segmented_dir=paths["segmented_dir"],
        aligned_dir=paths["aligned_dir"],
        chapters=[1],
    )
    sel, _ = select_pilot(candidates, n_contracted=1, n_uncontracted=1)
    out = write_pilot_tsv(sel, tmp_path / "out" / "pilot.tsv")
    return out


def test_paper_shared_prepositions_constant() -> None:
    """The filter set is exactly the canonical prepositions that have
    both a contracted form and an uncontracted form in the dataset."""
    # 13 canonical prepositions survive: an, auf, aus, bei, durch, für,
    # gegen, hinter, in, über, unter, von, zu. Excluded: um/vor (not in
    # the uncontracted PREPOSITIONS list).
    assert PAPER_SHARED_PREPOSITIONS == frozenset(
        {"an", "auf", "aus", "bei", "durch", "für", "gegen",
         "hinter", "in", "über", "unter", "von", "zu"}
    )


def _cand(
    prep_norm: str, form: str, *, chapter: int = 1, group: str | None = None
) -> dict[str, Any]:
    """Build a minimal candidate dict that select_paper_sample will accept."""
    if group is None:
        group = f"{prep_norm}|Synth"
    return {
        "de_prep_normalized": prep_norm,
        "de_form": form,
        "chapter": chapter,
        "minimal_pair_group": group,
    }


def test_select_paper_sample_keeps_shared_drops_rest() -> None:
    """Survivors must have canonical prep in PAPER_SHARED_PREPOSITIONS."""
    candidates = [
        # Shared — kept.
        _cand("in", "contracted"),
        _cand("in", "uncontracted"),
        _cand("zu", "contracted"),
        _cand("an", "uncontracted"),
        _cand("aus", "uncontracted"),  # aus IS shared (ausm → aus)
        # Not shared — dropped.
        _cand("um", "contracted"),    # um not in uncontracted list
        _cand("vor", "contracted"),   # vor not in uncontracted list
    ]
    selected, summary = select_paper_sample(candidates)
    kept_norms = sorted(c["de_prep_normalized"] for c in selected)
    assert kept_norms == ["an", "aus", "in", "in", "zu"]
    assert summary["selected_total"] == 5
    assert summary["dropped_total"] == 2
    assert summary["by_form"] == {"contracted": 2, "uncontracted": 3}
    assert summary["dropped_by_form"] == {"contracted": 2, "uncontracted": 0}


def test_select_paper_sample_does_not_mutate_input() -> None:
    """The input list and its dicts must be untouched after the call."""
    candidates = [_cand("in", "contracted"), _cand("aus", "uncontracted")]
    snapshot = [dict(c) for c in candidates]
    select_paper_sample(candidates)
    assert [dict(c) for c in candidates] == snapshot
    assert len(candidates) == 2  # no items appended or removed


def test_select_paper_sample_summary_fields() -> None:
    """Summary carries every field the build script's stdout depends on."""
    candidates = [
        _cand("in", "contracted", chapter=1, group="in|Haus"),
        _cand("in", "uncontracted", chapter=1, group="in|Haus"),  # both forms → group with both
        _cand("an", "uncontracted", chapter=2, group="an|Baum"),
        _cand("um", "contracted", chapter=1, group="um|X"),  # dropped (um not shared)
    ]
    _, summary = select_paper_sample(candidates)
    # New canonical keys + backward-compat aliases.
    assert set(summary.keys()) == {
        "candidate_total",
        "pool_total",
        "ineligible_total",
        "selected_total",  # backward-compat alias for pool_total
        "dropped_total",  # backward-compat alias for ineligible_total
        "by_form",
        "dropped_by_form",
        "by_chapter",
        "shared_prepositions",
        "minimal_pair_groups_in_sample",
        "minimal_pair_groups_with_both_forms",
    }
    assert summary["candidate_total"] == 4
    assert summary["pool_total"] == 3
    assert summary["ineligible_total"] == 1
    assert summary["by_chapter"] == {"1": 2, "2": 1}
    assert summary["minimal_pair_groups_in_sample"] == 2  # in|Haus, an|Baum
    assert summary["minimal_pair_groups_with_both_forms"] == 1  # only in|Haus


def test_select_paper_sample_tolerates_unknown_de_form() -> None:
    """An unknown de_form value must not raise KeyError — it lands in
    by_form / dropped_by_form as a new key with the canonical defaults
    still present."""
    candidates = [
        _cand("in", "contracted"),
        _cand("um", "unknown_form"),  # would have raised KeyError before fix
    ]
    selected, summary = select_paper_sample(candidates)
    assert len(selected) == 1  # only "in" survives the prep filter
    # by_form has the canonical keys (kept counts) plus the unknown form
    # does not appear (the unknown_form candidate was dropped by prep, not
    # by form). The drop is recorded under the unknown key.
    assert summary["by_form"] == {"contracted": 1, "uncontracted": 0}
    assert summary["dropped_by_form"] == {"contracted": 0, "uncontracted": 0,
                                          "unknown_form": 1}


def test_vorm_normalizes_to_vor() -> None:
    """vorm → vor (previously the table had vorn → vor, which is wrong:
    the author's CONTRACTED list pins vorm, not vorn)."""
    assert normalize_contracted_prep("vorm") == "vor"


def test_untern_normalizes_to_unter() -> None:
    """untern → unter (was missing from the table entirely)."""
    assert normalize_contracted_prep("untern") == "unter"


def test_untern_survives_eligibility_filter() -> None:
    """A PP with surface form ``untern`` would normalize to ``unter``,
    which IS in the 13-item paired inventory — so the row would survive
    the eligibility filter (it is not in real Ch.1–3 data, but the
    grammar must allow it)."""
    assert normalize_contracted_prep("untern") in PAPER_SHARED_PREPOSITIONS


def test_vorn_no_longer_in_table() -> None:
    """The buggy ``vorn`` entry must be gone (the author list has
    ``vorm``, not ``vorn``)."""
    assert "vorn" not in CONTRACTED_PREP_NORMALIZATION
    assert "vorm" in CONTRACTED_PREP_NORMALIZATION
    assert "untern" in CONTRACTED_PREP_NORMALIZATION


def test_inventory_remains_thirteen_prepositions() -> None:
    """The eligibility filter still has exactly the 13 canonical paired
    prepositions — not expanded."""
    assert PAPER_SHARED_PREPOSITIONS == frozenset(
        {"an", "auf", "aus", "bei", "durch", "für", "gegen",
         "hinter", "in", "über", "unter", "von", "zu"}
    )
    assert len(PAPER_SHARED_PREPOSITIONS) == 13


def test_contracted_table_parity_with_published_inventory() -> None:
    """Every form in the pinned author CONTRACTED list normalizes via
    :func:`normalize_contracted_prep` to its intended canonical base.

    Uses the tracked fixture ``tests/fixtures/author_contracted.txt``
    (a 26-form unique list mirroring the author's published inventory at
    the pinned commit). Each line is ``form|expected_canonical`` so the
    test verifies the exact mapping, not just prefix containment (some
    contractions like ``am → an`` do not share a prefix with their base).
    Runs without the vendor clone present.
    """
    fixture = Path(__file__).resolve().parent / "fixtures" / "author_contracted.txt"
    assert fixture.exists(), f"fixture missing: {fixture}"
    pairs: list[tuple[str, str]] = []
    for raw in fixture.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        form, sep, expected = line.partition("|")
        assert sep, f"malformed fixture line: {raw!r}"
        pairs.append((form.strip(), expected.strip()))
    assert pairs, "fixture is empty"
    for form, expected in pairs:
        assert form in CONTRACTED_PREP_NORMALIZATION, f"missing from table: {form}"
        actual = normalize_contracted_prep(form)
        assert actual == expected, (
            f"{form} normalizes to {actual!r}, fixture says {expected!r}"
        )


def test_contracted_table_parity_with_vendor() -> None:
    """Stronger parity test: imports the vendored author data module and
    asserts every form in ``CONTRACTED`` is in our table.

    Skipped when the vendor clone is absent (CI on a fresh checkout
    runs the fixture-based test above instead). The fixture test above
    covers the exact ``form → canonical`` mapping; this test only needs
    to confirm the table's key set is a superset of the author's.
    """
    pytest.importorskip("conll_extractor.prepositions.data")
    from conll_extractor.prepositions.data import CONTRACTED

    for form in CONTRACTED:
        assert form in CONTRACTED_PREP_NORMALIZATION, (
            f"author contracted form {form!r} missing from our table"
        )


def test_vendor_commit_matches_pin() -> None:
    """When the vendor clone is present, its HEAD must match the pin in
    ``vendor/conll-extractor.commit``. Skipped when the vendor is absent."""
    import subprocess

    vendor_dir = Path(__file__).resolve().parent.parent / "vendor" / "conll-extractor"
    pin_file = Path(__file__).resolve().parent.parent / "vendor" / "conll-extractor.commit"
    if not vendor_dir.exists() or not (vendor_dir / ".git").exists():
        pytest.skip("vendor/conll-extractor not cloned")
    actual = subprocess.run(
        ["git", "-C", str(vendor_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    expected = pin_file.read_text(encoding="utf-8").strip()
    assert actual == expected, (
        f"vendor HEAD ({actual}) does not match pin ({expected}); "
        f"the vendored checkout has drifted from the validated revision"
    )


def test_build_candidates_fails_on_missing_extraction_tsv(tmp_path: Path) -> None:
    """Removing one extraction TSV must raise MissingInputsError."""
    from hp_corpus.step4 import MissingInputsError

    paths = _build_synth_repo(tmp_path)
    (paths["extraction_dir"] / "hp1_de_ch01_contracted.tsv").unlink()
    with pytest.raises(MissingInputsError, match="missing inputs"):
        build_candidates(
            extraction_dir=paths["extraction_dir"],
            segmented_dir=paths["segmented_dir"],
            aligned_dir=paths["aligned_dir"],
            chapters=[1],
        )


def test_build_candidates_fails_on_missing_segmented_file(tmp_path: Path) -> None:
    from hp_corpus.step4 import MissingInputsError

    paths = _build_synth_repo(tmp_path)
    (paths["segmented_dir"] / "hp1_zh_ch01.jsonl").unlink()
    with pytest.raises(MissingInputsError, match="missing inputs"):
        build_candidates(
            extraction_dir=paths["extraction_dir"],
            segmented_dir=paths["segmented_dir"],
            aligned_dir=paths["aligned_dir"],
            chapters=[1],
        )


def test_build_candidates_fails_on_missing_alignment_file(tmp_path: Path) -> None:
    from hp_corpus.step4 import MissingInputsError

    paths = _build_synth_repo(tmp_path)
    (paths["aligned_dir"] / "hp1_de_zh_ch01.jsonl").unlink()
    with pytest.raises(MissingInputsError, match="missing inputs"):
        build_candidates(
            extraction_dir=paths["extraction_dir"],
            segmented_dir=paths["segmented_dir"],
            aligned_dir=paths["aligned_dir"],
            chapters=[1],
        )


def test_build_candidates_fails_on_header_only_tsv(tmp_path: Path) -> None:
    """A header-only extraction TSV must be rejected — for Ch.1–3 every
    form is known to be populated, so header-only signals an upstream
    failure rather than a legitimately-empty chapter."""
    from hp_corpus.step4 import MissingInputsError

    paths = _build_synth_repo(tmp_path)
    # Overwrite with a header-only file.
    from hp_corpus.step4 import SOURCE_COLUMNS as _SC  # noqa: F401 (sanity)
    fields = [
        "parse_block_id", "source_segment_id", "prep", "det", "noun",
        "prep_token_id", "det_token_id", "noun_token_id",
        "pp_token_start", "pp_token_end", "pp_surface", "in_filter",
    ]
    with open(paths["extraction_dir"] / "hp1_de_ch01_contracted.tsv", "w", encoding="utf-8") as f:
        f.write("\t".join(fields) + "\n")
    with pytest.raises(MissingInputsError, match="malformed"):
        build_candidates(
            extraction_dir=paths["extraction_dir"],
            segmented_dir=paths["segmented_dir"],
            aligned_dir=paths["aligned_dir"],
            chapters=[1],
        )


def test_build_candidates_fails_on_empty_tsv(tmp_path: Path) -> None:
    from hp_corpus.step4 import MissingInputsError

    paths = _build_synth_repo(tmp_path)
    (paths["extraction_dir"] / "hp1_de_ch01_uncontracted.tsv").write_text("", encoding="utf-8")
    with pytest.raises(MissingInputsError, match="malformed"):
        build_candidates(
            extraction_dir=paths["extraction_dir"],
            segmented_dir=paths["segmented_dir"],
            aligned_dir=paths["aligned_dir"],
            chapters=[1],
        )


def test_build_candidates_fails_on_unresolvable_de_segment_id(tmp_path: Path) -> None:
    """An extraction TSV row pointing at a sentence id the segmented
    JSONL doesn't have must raise UnresolvedSegmentIdError rather than
    silently emitting an empty sentence-text row."""
    from hp_corpus.step4 import UnresolvedSegmentIdError

    paths = _build_synth_repo(tmp_path)
    # Inject a row pointing at a sentence id that doesn't exist.
    _write_extraction_tsv(
        paths["extraction_dir"] / "hp1_de_ch01_contracted.tsv",
        "contracted",
        [
            {
                "parse_block_id": "hp1_de_ch01_p9999_s999#b001",
                "source_segment_id": "hp1_de_ch01_p9999_s999",  # not in segments
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
    with pytest.raises(UnresolvedSegmentIdError, match="hp1_de_ch01_p9999_s999"):
        build_candidates(
            extraction_dir=paths["extraction_dir"],
            segmented_dir=paths["segmented_dir"],
            aligned_dir=paths["aligned_dir"],
            chapters=[1],
        )


def test_builder_rejects_chapters_outside_1_2_3(tmp_path: Path) -> None:
    """``--chapters`` other than [1, 2, 3] is rejected at the CLI layer."""

    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "build_ch1_3_full_annotation.py"
    spec = importlib.util.spec_from_file_location("build_ch1_3_full_annotation", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # parser.error() raises SystemExit in argparse.
    with pytest.raises(SystemExit):
        mod.main(["--chapters", "1"])


def _annotate_row_fully(
    cols: list[str], row: list[str], *, en_text: str
) -> list[str]:
    """Helper: set every field needed for --require-complete on one row."""
    def _set(name: str, val: str) -> None:
        row[cols.index(name)] = val

    # DE candidate
    _set("de_candidate_decision", "include")
    # EN: confirmed + direct + valid span + form + confidence.
    _set("en_alignment_qc", "confirmed")
    _set("en_alignment_relation", "direct")
    # Span = whole en_aligned_text.
    _set("en_span_text", en_text)
    _set("en_char_ranges", f"[[0,{len(en_text)}]]")
    _set("en_form", "definite")
    _set("en_confidence", "high")
    # ZH: confirmed + omitted (no counterpart).
    _set("zh_alignment_qc", "confirmed")
    _set("zh_alignment_relation", "omitted")
    _set("zh_form", "omitted")
    _set("zh_confidence", "high")
    # Row status.
    _set("annotation_status", "complete")
    return row


def test_crosslingual_map_docstring_states_pp_shape_limitation() -> None:
    """The module docstring must explicitly disclaim NP/pronoun/paraphrase/
    omitted coverage so callers don't treat the output as exhaustive."""
    from hp_corpus import crosslingual_map

    doc = crosslingual_map.__doc__ or ""
    assert "PP-shaped candidates only" in doc
    assert "pronominal" in doc
    assert "omitted" in doc
    assert "never auto-populate" in doc.lower() or "auto-populate" in doc.lower()
