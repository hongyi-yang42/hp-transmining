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

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "validate_step4_annotations.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_step4_annotations", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------- helpers


def _write_extraction_tsv(path: Path, kind: str, rows: list[dict[str, Any]]) -> None:
    """Write an extraction TSV in the same format as run_paper_extractor.py."""
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
    # occurrences stay distinguishable.
    _write_extraction_tsv(
        extraction / "hp1_de_ch01_contracted.tsv",
        "contracted",
        [
            {
                "sentence_id": "hp1_de_ch01_p0001_s001",
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
                # Same (prep, noun) again — must NOT be deduped.
                "sentence_id": "hp1_de_ch01_p0001_s001",
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
                "sentence_id": "hp1_de_ch01_p0002_s001",
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
                "sentence_id": "hp1_de_ch01_p0003_s001",
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
                "sentence_id": "hp1_de_ch01_p0004_s001",
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
        if c["de_sentence_id"] == "hp1_de_ch01_p0001_s001" and c["de_form"] == "contracted"
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
        if c["de_sentence_id"] == "hp1_de_ch01_p0003_s001" and c["de_form"] == "uncontracted"
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
        if c["de_sentence_id"] == "hp1_de_ch01_p0001_s001" and c["de_form"] == "contracted"
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
        assert c["de_sentence_id"].startswith("hp1_de_")
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
                "sentence_id": "hp1_de_ch01_p0005_s001",
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
    missing = [c for c in candidates if c["de_sentence_id"] == "hp1_de_ch01_p0005_s001"]
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
                "sentence_id": "hp1_de_ch01_p0001_s001",
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
                "sentence_id": "hp1_de_ch01_p0005_s001",  # missing alignment
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
    assert all(c["de_sentence_id"] != "hp1_de_ch01_p0005_s001" for c in sel)


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
    # All EDITABLE_COLUMNS present but blank in initial state
    for col in EDITABLE_COLUMNS:
        assert col in rows[0]
        assert rows[0][col] == ""


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


def test_validator_initial_state_passes(tmp_path: Path, monkeypatch) -> None:
    validator = _load_validator()
    # Patch the expected pilot size down to 1+1 so our 2-row fixture is balanced.
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_CONTRACTED", 1)
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_UNCONTRACTED", 1)
    # The imported constants inside the validator's checks reference the
    # module's globals at call time, so monkeypatching works.
    pilot_tsv = _write_initial_pilot_tsv(tmp_path, validator)
    violations, summary = validator.validate_tsv(pilot_tsv)
    assert summary["initial_state"] is True
    assert violations == [], [str(v) for v in violations]


def test_validator_catches_edited_source_column(tmp_path: Path, monkeypatch) -> None:
    validator = _load_validator()
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_CONTRACTED", 1)
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_UNCONTRACTED", 1)
    pilot_tsv = _write_initial_pilot_tsv(tmp_path, validator)
    # Edit one source column on row 2 — but keep source_row_sha256 unchanged,
    # which is exactly the failure mode the hash detects.
    rows = pilot_tsv.read_text(encoding="utf-8").splitlines()
    header = rows[0]
    body = rows[1:]
    cols = header.split("\t")
    noun_idx = cols.index("de_head_lemma")
    parts = body[0].split("\t")
    parts[noun_idx] = "EDITED"
    body[0] = "\t".join(parts)
    pilot_tsv.write_text(header + "\n" + "\n".join(body) + "\n", encoding="utf-8")

    violations, _ = validator.validate_tsv(pilot_tsv)
    rules = {v.rule for v in violations}
    assert "SOURCE_ROW_HASH_MISMATCH" in rules


def test_validator_validates_direct_span(tmp_path: Path, monkeypatch) -> None:
    validator = _load_validator()
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_CONTRACTED", 1)
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_UNCONTRACTED", 1)
    pilot_tsv = _write_initial_pilot_tsv(tmp_path, validator)
    rows = pilot_tsv.read_text(encoding="utf-8").splitlines()
    header = rows[0]
    cols = header.split("\t")
    body = [r.split("\t") for r in rows[1:]]

    # EN aligned text is "synth EN one" (15 chars). Span [0, 6] = "synth ".
    # Wait — that includes trailing space. Use [0, 5] = "synth".
    en_text_idx = cols.index("en_aligned_text")
    en_span_idx = cols.index("en_span_text")
    en_range_idx = cols.index("en_char_ranges")
    en_rel_idx = cols.index("en_alignment_relation")
    en_form_idx = cols.index("en_form")
    en_conf_idx = cols.index("en_confidence")
    ann_status_idx = cols.index("annotation_status")

    en_text = body[0][en_text_idx]
    assert en_text.startswith("synth")
    body[0][en_rel_idx] = "direct"
    body[0][en_span_idx] = "synth"
    body[0][en_range_idx] = "[[0,5]]"
    body[0][en_form_idx] = "definite"
    body[0][en_conf_idx] = "high"
    body[0][ann_status_idx] = "in_progress"

    pilot_tsv.write_text(
        header + "\n" + "\n".join("\t".join(r) for r in body) + "\n", encoding="utf-8"
    )
    violations, _ = validator.validate_tsv(pilot_tsv)
    # Filter out only EN-annotation-related violations
    rel = [v for v in violations if "en_" in v.message or "alignment_relation" in v.message]
    assert rel == [], [str(v) for v in rel]


def test_validator_validates_discontinuous_span(tmp_path: Path, monkeypatch) -> None:
    validator = _load_validator()
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_CONTRACTED", 1)
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_UNCONTRACTED", 1)
    pilot_tsv = _write_initial_pilot_tsv(tmp_path, validator)
    rows = pilot_tsv.read_text(encoding="utf-8").splitlines()
    header = rows[0]
    cols = header.split("\t")
    body = [r.split("\t") for r in rows[1:]]

    en_span_idx = cols.index("en_span_text")
    en_range_idx = cols.index("en_char_ranges")
    en_rel_idx = cols.index("en_alignment_relation")

    # "synth EN one" indices: s0 y1 n2 t3 h4 (sp)5 E6 N7 (sp)8 o9 n10 e11
    # Discontinuous: "synth" ([0,5]) + "one" ([9,12]) joined by space.
    body[0][en_rel_idx] = "paraphrase"
    body[0][en_span_idx] = "synth one"
    body[0][en_range_idx] = "[[0,5],[9,12]]"

    pilot_tsv.write_text(
        header + "\n" + "\n".join("\t".join(r) for r in body) + "\n", encoding="utf-8"
    )
    violations, _ = validator.validate_tsv(pilot_tsv)
    span_vs = [v for v in violations if "en_" in v.message and "span" in v.message.lower()]
    assert span_vs == [], [str(v) for v in span_vs]


def test_validator_validates_omission(tmp_path: Path, monkeypatch) -> None:
    validator = _load_validator()
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_CONTRACTED", 1)
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_UNCONTRACTED", 1)
    pilot_tsv = _write_initial_pilot_tsv(tmp_path, validator)
    rows = pilot_tsv.read_text(encoding="utf-8").splitlines()
    header = rows[0]
    cols = header.split("\t")
    body = [r.split("\t") for r in rows[1:]]

    en_rel_idx = cols.index("en_alignment_relation")
    en_form_idx = cols.index("en_form")
    en_span_idx = cols.index("en_span_text")
    en_range_idx = cols.index("en_char_ranges")

    body[0][en_rel_idx] = "omitted"
    body[0][en_form_idx] = "omitted"
    body[0][en_span_idx] = ""
    body[0][en_range_idx] = ""

    pilot_tsv.write_text(
        header + "\n" + "\n".join("\t".join(r) for r in body) + "\n", encoding="utf-8"
    )
    violations, _ = validator.validate_tsv(pilot_tsv)
    omitted_vs = [
        v
        for v in violations
        if v.rule in ("OMITTED_HAS_SPAN", "OMITTED_WRONG_FORM") and "en_" in v.message
    ]
    assert omitted_vs == [], [str(v) for v in omitted_vs]


def test_validator_catches_invalid_form_value(tmp_path: Path, monkeypatch) -> None:
    validator = _load_validator()
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_CONTRACTED", 1)
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_UNCONTRACTED", 1)
    pilot_tsv = _write_initial_pilot_tsv(tmp_path, validator)
    rows = pilot_tsv.read_text(encoding="utf-8").splitlines()
    header = rows[0]
    cols = header.split("\t")
    body = [r.split("\t") for r in rows[1:]]

    en_form_idx = cols.index("en_form")
    body[0][en_form_idx] = "weak"  # invalid — must be one of the controlled set

    pilot_tsv.write_text(
        header + "\n" + "\n".join("\t".join(r) for r in body) + "\n", encoding="utf-8"
    )
    violations, _ = validator.validate_tsv(pilot_tsv)
    rules = {v.rule for v in violations}
    assert "BAD_VOCAB" in rules


def test_validator_catches_out_of_range_offsets(tmp_path: Path, monkeypatch) -> None:
    validator = _load_validator()
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_CONTRACTED", 1)
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_UNCONTRACTED", 1)
    pilot_tsv = _write_initial_pilot_tsv(tmp_path, validator)
    rows = pilot_tsv.read_text(encoding="utf-8").splitlines()
    header = rows[0]
    cols = header.split("\t")
    body = [r.split("\t") for r in rows[1:]]

    en_text_idx = cols.index("en_aligned_text")
    en_range_idx = cols.index("en_char_ranges")
    en_rel_idx = cols.index("en_alignment_relation")

    en_text = body[0][en_text_idx]
    # Set a range far past the end of the aligned text.
    body[0][en_rel_idx] = "direct"
    body[0][en_range_idx] = f"[[0,{len(en_text) + 100}]]"

    pilot_tsv.write_text(
        header + "\n" + "\n".join("\t".join(r) for r in body) + "\n", encoding="utf-8"
    )
    violations, _ = validator.validate_tsv(pilot_tsv)
    rules = {v.rule for v in violations}
    assert "RANGE_OUT_OF_BOUNDS" in rules


def test_validator_catches_span_mismatch(tmp_path: Path, monkeypatch) -> None:
    validator = _load_validator()
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_CONTRACTED", 1)
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_UNCONTRACTED", 1)
    pilot_tsv = _write_initial_pilot_tsv(tmp_path, validator)
    rows = pilot_tsv.read_text(encoding="utf-8").splitlines()
    header = rows[0]
    cols = header.split("\t")
    body = [r.split("\t") for r in rows[1:]]

    en_range_idx = cols.index("en_char_ranges")
    en_span_idx = cols.index("en_span_text")
    en_rel_idx = cols.index("en_alignment_relation")

    body[0][en_rel_idx] = "direct"
    body[0][en_range_idx] = "[[0,5]]"  # would reconstruct "synth"
    body[0][en_span_idx] = "wrong"  # mismatched text

    pilot_tsv.write_text(
        header + "\n" + "\n".join("\t".join(r) for r in body) + "\n", encoding="utf-8"
    )
    violations, _ = validator.validate_tsv(pilot_tsv)
    rules = {v.rule for v in violations}
    assert "SPAN_MISMATCH" in rules


def test_validator_catches_uncertain_without_note(tmp_path: Path, monkeypatch) -> None:
    validator = _load_validator()
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_CONTRACTED", 1)
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_UNCONTRACTED", 1)
    pilot_tsv = _write_initial_pilot_tsv(tmp_path, validator)
    rows = pilot_tsv.read_text(encoding="utf-8").splitlines()
    header = rows[0]
    cols = header.split("\t")
    body = [r.split("\t") for r in rows[1:]]

    en_rel_idx = cols.index("en_alignment_relation")
    en_notes_idx = cols.index("en_notes")

    body[0][en_rel_idx] = "uncertain"
    body[0][en_notes_idx] = ""  # missing required note

    pilot_tsv.write_text(
        header + "\n" + "\n".join("\t".join(r) for r in body) + "\n", encoding="utf-8"
    )
    violations, _ = validator.validate_tsv(pilot_tsv)
    rules = {v.rule for v in violations}
    assert "UNCERTAIN_NO_NOTE" in rules


def test_validator_stdout_no_source_text(tmp_path: Path, monkeypatch, capsys) -> None:
    validator = _load_validator()
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_CONTRACTED", 1)
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_UNCONTRACTED", 1)
    pilot_tsv = _write_initial_pilot_tsv(tmp_path, validator)
    rc = validator.main([str(pilot_tsv)])
    captured = capsys.readouterr()
    # Stdout must NOT carry any source novel text — only aggregate counts and
    # row indices.
    assert "synth DE" not in captured.out
    assert "synth EN" not in captured.out
    assert "synth ZH" not in captured.out
    assert "Haus" not in captured.out
    assert "Wald" not in captured.out
    assert rc == 0


# --------------------------------------------------------------------- paper-sample
# Tests for select_paper_sample() and the validator --full-sample flag.
# Synthetic candidate dicts only — no novel text, no filter-list lemmas.


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
    assert set(summary.keys()) == {
        "candidate_total", "selected_total", "dropped_total",
        "by_form", "dropped_by_form", "by_chapter",
        "shared_prepositions", "minimal_pair_groups_in_sample",
        "minimal_pair_groups_with_both_forms",
    }
    assert summary["candidate_total"] == 4
    assert summary["selected_total"] == 3
    assert summary["by_chapter"] == {"1": 2, "2": 1}
    assert summary["minimal_pair_groups_in_sample"] == 2  # in|Haus, an|Baum
    assert summary["minimal_pair_groups_with_both_forms"] == 1  # only in|Haus


def test_validator_full_sample_flag_skips_pilot_balance(tmp_path: Path) -> None:
    """A TSV whose contracted/uncontracted counts are NOT 10+10 must
    pass with --full-sample and fail without it."""
    validator = _load_validator()
    # Build a real candidate set from the synthetic Ch.1 fixture corpus,
    # then write it as a TSV. The synth corpus yields 3 contracted + 2
    # uncontracted — neither count matches the 10+10 pilot expectation.
    repo = _build_synth_repo(tmp_path)
    candidates = build_candidates(
        extraction_dir=repo["extraction_dir"],
        segmented_dir=repo["segmented_dir"],
        aligned_dir=repo["aligned_dir"],
        chapters=[1],
    )
    selected, _ = select_paper_sample(candidates)
    full_tsv = tmp_path / "full.tsv"
    write_pilot_tsv(selected, full_tsv)

    # Default mode: PILOT_IMBALANCE expected.
    violations_default, _ = validator.validate_tsv(full_tsv)
    rules_default = {v.rule for v in violations_default}
    assert "PILOT_IMBALANCE" in rules_default

    # Full-sample mode: must pass cleanly.
    violations_full, summary_full = validator.validate_tsv(full_tsv, full_sample=True)
    assert violations_full == []
    assert summary_full["full_sample"] is True


def test_validator_default_mode_still_enforces_pilot_balance(tmp_path: Path) -> None:
    """Regression: default-mode validator still flags imbalance on a
    non-pilot-sized TSV. Guards against accidentally removing the check."""
    validator = _load_validator()
    # Build candidates, then write a TSV without the balance override.
    repo = _build_synth_repo(tmp_path)
    candidates = build_candidates(
        extraction_dir=repo["extraction_dir"],
        segmented_dir=repo["segmented_dir"],
        aligned_dir=repo["aligned_dir"],
        chapters=[1],
    )
    selected, _ = select_paper_sample(candidates)
    full_tsv = tmp_path / "full.tsv"
    write_pilot_tsv(selected, full_tsv)

    # Sanity: the synth corpus really does produce a non-10+10 file.
    rows = full_tsv.read_text(encoding="utf-8").splitlines()
    body = [r for r in rows[1:] if r]
    assert len(body) != 20

    violations, _ = validator.validate_tsv(full_tsv)
    rules = {v.rule for v in violations}
    assert "PILOT_IMBALANCE" in rules


# --------------------------------------------------------------------- post-review
# Tests for fixes landed in the xhigh code-review pass:
#   - select_paper_sample / summarize_candidates: defensive de_form access
#   - validator: empty-body check, OMITTED_WRONG_FORM with blank form,
#     summary populated on early-return paths


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


def test_validator_empty_body_flagged(tmp_path: Path) -> None:
    """A header-only TSV must be flagged even in --full-sample mode (where
    PILOT_IMBALANCE is skipped and would otherwise let it pass silently)."""
    validator = _load_validator()
    # Write a header-only TSV by building a real one and stripping the body.
    repo = _build_synth_repo(tmp_path)
    candidates = build_candidates(
        extraction_dir=repo["extraction_dir"],
        segmented_dir=repo["segmented_dir"],
        aligned_dir=repo["aligned_dir"],
        chapters=[1],
    )
    selected, _ = select_paper_sample(candidates)
    real_tsv = tmp_path / "real.tsv"
    write_pilot_tsv(selected, real_tsv, scope_override="ch1_3_paper_sample")
    lines = real_tsv.read_text(encoding="utf-8").splitlines()
    header_only = tmp_path / "header_only.tsv"
    header_only.write_text(lines[0] + "\n", encoding="utf-8")

    violations, summary = validator.validate_tsv(header_only, full_sample=True)
    rules = {v.rule for v in violations}
    assert "EMPTY_BODY" in rules
    # Summary is now populated even on the would-be-silent path.
    assert summary["rows"] == 0
    assert summary["violation_count"] >= 1
    assert summary["full_sample"] is True


def test_validator_omitted_blank_form_flagged(tmp_path: Path, monkeypatch) -> None:
    """relation=omitted + blank form must be flagged OMITTED_WRONG_FORM
    (the canonical state is form=omitted; a blank cell is wrong, not
    merely unannotated)."""
    validator = _load_validator()
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_CONTRACTED", 1)
    monkeypatch.setattr(validator, "PILOT_DEFAULT_N_UNCONTRACTED", 1)
    pilot_tsv = _write_initial_pilot_tsv(tmp_path, validator)
    rows = pilot_tsv.read_text(encoding="utf-8").splitlines()
    header = rows[0]
    cols = header.split("\t")
    body = [r.split("\t") for r in rows[1:]]

    en_rel_idx = cols.index("en_alignment_relation")
    en_form_idx = cols.index("en_form")

    body[0][en_rel_idx] = "omitted"
    body[0][en_form_idx] = ""  # blank — used to slip through

    pilot_tsv.write_text(
        header + "\n" + "\n".join("\t".join(r) for r in body) + "\n", encoding="utf-8"
    )
    violations, _ = validator.validate_tsv(pilot_tsv)
    rules = {v.rule for v in violations}
    assert "OMITTED_WRONG_FORM" in rules


def test_validator_summary_populated_on_header_mismatch(tmp_path: Path) -> None:
    """Early-return paths (HEADER_MISMATCH, EMPTY_FILE) must populate the
    summary so main()'s `violations: {summary.get('violation_count', 0)}`
    doesn't print a misleading 'violations: 0' on a real failure."""
    validator = _load_validator()
    bad = tmp_path / "bad_header.tsv"
    # Wrong header — fewer columns than ALL_TSV_COLUMNS.
    bad.write_text("a\tb\tc\n1\t2\t3\n", encoding="utf-8")
    violations, summary = validator.validate_tsv(bad, full_sample=True)
    rules = {v.rule for v in violations}
    assert "HEADER_MISMATCH" in rules
    # Summary fields are populated; main() can read them.
    assert summary["violation_count"] == len(violations)
    assert summary["rows"] == 0
    assert summary["full_sample"] is True
    assert summary["by_rule"]["HEADER_MISMATCH"] == 1


def test_validator_empty_file_summary_populated(tmp_path: Path) -> None:
    """EMPTY_FILE return also populates the summary."""
    validator = _load_validator()
    empty = tmp_path / "empty.tsv"
    empty.write_text("", encoding="utf-8")
    violations, summary = validator.validate_tsv(empty)
    rules = {v.rule for v in violations}
    assert "EMPTY_FILE" in rules
    assert summary["violation_count"] == len(violations)
    assert summary["rows"] == 0
    assert summary["by_rule"]["EMPTY_FILE"] == 1


def test_write_pilot_tsv_scope_override(tmp_path: Path) -> None:
    """scope_override parameter rewrites dataset_scope and the source-row
    hash picks up the override (so the validator's hash check still passes)."""
    validator = _load_validator()
    repo = _build_synth_repo(tmp_path)
    candidates = build_candidates(
        extraction_dir=repo["extraction_dir"],
        segmented_dir=repo["segmented_dir"],
        aligned_dir=repo["aligned_dir"],
        chapters=[1],
    )
    selected, _ = select_paper_sample(candidates)
    out = tmp_path / "scoped.tsv"
    write_pilot_tsv(selected, out, scope_override="ch1_3_paper_sample")

    # Every row carries the overridden scope.
    rows = out.read_text(encoding="utf-8").splitlines()
    cols = rows[0].split("\t")
    scope_idx = cols.index("dataset_scope")
    for r in rows[1:]:
        assert r.split("\t")[scope_idx] == "ch1_3_paper_sample"

    # Validator (full-sample mode) passes — hash check is consistent with
    # the overridden scope.
    violations, _ = validator.validate_tsv(out, full_sample=True)
    assert not [v for v in violations if v.rule == "SOURCE_ROW_HASH_MISMATCH"]
