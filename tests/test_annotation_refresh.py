"""Synthetic-fixture tests for ``scripts/refresh_annotation_template.py``.

Every fixture here uses invented, non-novel text — no Harry Potter
content, no filter-list lemmas. The script under test is loaded as a
module (it lives under ``scripts/``, not under ``src/``).
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from hp_corpus.step4 import (
    ALL_TSV_COLUMNS,
    BUILDER_DEFAULT_EDITABLE,
    EDITABLE_COLUMNS,
    SOURCE_COLUMNS,
    compute_source_row_sha256,
)

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "refresh_annotation_template.py"
)

# Sentinel strings used in fixtures. Distinctive enough that the privacy
# guard test can assert they don't leak to stdout.
SYNTH_DE_TEXT = "synth DE alpha beta gamma"
SYNTH_EN_TEXT = "synth EN alpha beta gamma"
SYNTH_ZH_TEXT = "synth ZH alpha beta gamma"
SYNTH_LEMMA = "SynthNoun"
SYNTH_PREP = "im"


def _load_refresh_module():
    spec = importlib.util.spec_from_file_location(
        "refresh_annotation_template", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------- helpers


def _make_candidate(
    *,
    datapoint_id: str,
    chapter: int = 1,
    de_form: str = "contracted",
    de_sentence_id: str = "hp1_de_ch01_p0001_s001",
    token_start: int = 1,
    token_end: int = 2,
    de_pp_surface: str = "im SynthNoun",
    de_sentence_text: str = SYNTH_DE_TEXT,
    de_prep_surface: str = SYNTH_PREP,
    de_prep_normalized: str = "in",
    de_head_lemma: str = SYNTH_LEMMA,
    en_aligned_text: str = SYNTH_EN_TEXT,
    zh_aligned_text: str = SYNTH_ZH_TEXT,
    en_alignment_confidence: float = 0.85,
    zh_alignment_confidence: float = 0.80,
    en_alignment_status: str = "aligned",
    zh_alignment_status: str = "aligned",
    en_alignment_cardinality: str = "1:1",
    zh_alignment_cardinality: str = "1:1",
    en_sentence_ids: list[str] | None = None,
    zh_sentence_ids: list[str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete candidate dict with sensible synthetic defaults.

    ``source_row_sha256`` is computed via :func:`compute_source_row_sha256`
    so the resulting TSV row is internally consistent.
    """
    cand: dict[str, Any] = {
        "datapoint_id": datapoint_id,
        "dataset_scope": "ch1_3_annotation_target",
        "paper_final_sample": False,
        "chapter": chapter,
        "de_sentence_id": de_sentence_id,
        "de_token_start": token_start,
        "de_token_end": token_end,
        "de_pp_surface": de_pp_surface,
        "de_sentence_text": de_sentence_text,
        "de_prep_surface": de_prep_surface,
        "de_prep_normalized": de_prep_normalized,
        "de_head_lemma": de_head_lemma,
        "de_form": de_form,
        "author_resource_match": True,
        "minimal_pair_group": f"{de_prep_normalized}|{de_head_lemma}",
        "en_sentence_ids": en_sentence_ids
        if en_sentence_ids is not None
        else ["hp1_en_ch01_p0001_s001"],
        "en_aligned_text": en_aligned_text,
        "en_alignment_cardinality": en_alignment_cardinality,
        "en_alignment_status": en_alignment_status,
        "en_alignment_confidence": en_alignment_confidence,
        "zh_sentence_ids": zh_sentence_ids
        if zh_sentence_ids is not None
        else ["hp1_zh_ch01_p0001_s001"],
        "zh_aligned_text": zh_aligned_text,
        "zh_alignment_cardinality": zh_alignment_cardinality,
        "zh_alignment_status": zh_alignment_status,
        "zh_alignment_confidence": zh_alignment_confidence,
        "pilot_selected": False,
        "pilot_selection_reason": "",
    }
    if overrides:
        cand.update(overrides)
    cand["source_row_sha256"] = compute_source_row_sha256(cand)
    return cand


def _serialize_source_value(col: str, val: Any) -> str:
    """Mirror the writer's serialization for source columns when writing
    TSV fixtures by hand (separate from the candidate builder)."""
    if isinstance(val, list):
        return json.dumps(val, ensure_ascii=False)
    if isinstance(val, bool):
        return "true" if val else "false"
    if val is None:
        return ""
    return str(val)


def _candidate_to_tsv_row(cand: dict[str, Any]) -> dict[str, str]:
    """Convert a candidate dict to a TSV row dict with editable columns
    set to their builder defaults (i.e. the initial template state).

    Source columns are serialized exactly as the production writer does it.
    """
    row: dict[str, str] = {}
    for col in ALL_TSV_COLUMNS:
        if col in SOURCE_COLUMNS:
            if col == "source_row_sha256":
                row[col] = cand["source_row_sha256"]
                continue
            row[col] = _serialize_source_value(col, cand.get(col, ""))
        elif col in EDITABLE_COLUMNS:
            row[col] = BUILDER_DEFAULT_EDITABLE.get(col, "")
        else:
            row[col] = ""
    return row


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> Path:
    """Write a TSV with ``ALL_TSV_COLUMNS`` column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(ALL_TSV_COLUMNS),
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        w.writeheader()
        for r in rows:
            w.writerow({col: r.get(col, "") for col in ALL_TSV_COLUMNS})
    return path


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f, delimiter="\t")]


def _annotate_row(
    row: dict[str, str], annotations: dict[str, str]
) -> dict[str, str]:
    """Return a copy of ``row`` with the given editable-column annotations applied.

    Source columns cannot be edited via this helper — it's reserved for
    annotation columns only.
    """
    out = dict(row)
    for k, v in annotations.items():
        assert k in EDITABLE_COLUMNS, f"refusing to annotate non-editable column {k!r}"
        out[k] = v
    return out


# --------------------------------------------------------------------- tests


def test_happy_path_hash_match_copies_editable(tmp_path: Path) -> None:
    """OLD has annotations on row X with hash H; NEW has same row with same
    hash H → output's editable fields equal OLD's, source fields equal NEW's."""
    refresh = _load_refresh_module()

    cand = _make_candidate(datapoint_id="dp_test_001")
    new_row = _candidate_to_tsv_row(cand)
    old_row = _annotate_row(
        new_row,
        {
            "de_candidate_decision": "include",
            "en_alignment_qc": "confirmed",
            "en_form": "definite",
            "en_alignment_relation": "direct",
            "zh_alignment_qc": "confirmed",
            "annotation_status": "complete",
            "annotator": "annotator_a",
        },
    )

    old_tsv = _write_tsv(tmp_path / "old.tsv", [old_row])
    new_tsv = _write_tsv(tmp_path / "new.tsv", [new_row])
    out_tsv = tmp_path / "out.tsv"

    rc = refresh.main(
        [
            "--old-tsv", str(old_tsv),
            "--new-tsv", str(new_tsv),
            "--output", str(out_tsv),
        ]
    )
    assert rc == 0

    out_rows = _read_tsv(out_tsv)
    assert len(out_rows) == 1
    out_row = out_rows[0]
    # Editable fields copied from OLD.
    assert out_row["de_candidate_decision"] == "include"
    assert out_row["en_alignment_qc"] == "confirmed"
    assert out_row["en_form"] == "definite"
    assert out_row["en_alignment_relation"] == "direct"
    assert out_row["zh_alignment_qc"] == "confirmed"
    assert out_row["annotation_status"] == "complete"
    assert out_row["annotator"] == "annotator_a"
    # Source fields equal NEW's (i.e. the same string — old==new here).
    for col in SOURCE_COLUMNS:
        assert out_row[col] == new_row[col], col


def test_hash_mismatch_blanks_and_logs_conflict(tmp_path: Path) -> None:
    """OLD row X has hash H1, NEW has hash H2 → output's editable fields
    are blank (builder defaults), conflict ledger has one record with both
    hashes."""
    refresh = _load_refresh_module()

    cand_old = _make_candidate(datapoint_id="dp_test_002", de_head_lemma="OldLemma")
    cand_new = _make_candidate(datapoint_id="dp_test_002", de_head_lemma="NewLemma")
    # Sanity: hashes differ.
    assert cand_old["source_row_sha256"] != cand_new["source_row_sha256"]

    old_row = _annotate_row(
        _candidate_to_tsv_row(cand_old),
        {
            "de_candidate_decision": "include",
            "en_form": "definite",
            "annotation_status": "complete",
        },
    )
    new_row = _candidate_to_tsv_row(cand_new)

    old_tsv = _write_tsv(tmp_path / "old.tsv", [old_row])
    new_tsv = _write_tsv(tmp_path / "new.tsv", [new_row])
    out_tsv = tmp_path / "out.tsv"
    ledger = tmp_path / "out.tsv.conflicts.jsonl"

    rc = refresh.main(
        [
            "--old-tsv", str(old_tsv),
            "--new-tsv", str(new_tsv),
            "--output", str(out_tsv),
            "--conflict-ledger", str(ledger),
        ]
    )
    assert rc == 0

    out_rows = _read_tsv(out_tsv)
    assert len(out_rows) == 1
    out_row = out_rows[0]
    # Editable fields are blank (builder defaults).
    for col in EDITABLE_COLUMNS:
        assert out_row[col] == BUILDER_DEFAULT_EDITABLE.get(col, ""), col

    # Conflict ledger has exactly one record with both hashes.
    assert ledger.exists()
    lines = [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1
    rec = lines[0]
    assert rec["datapoint_id"] == "dp_test_002"
    assert rec["old_hash"] == cand_old["source_row_sha256"]
    assert rec["new_hash"] == cand_new["source_row_sha256"]
    assert rec["chapter"] == "1"
    assert rec["de_form"] == "contracted"
    # The conflict record must NOT contain source text.
    rec_blob = json.dumps(rec)
    assert SYNTH_DE_TEXT not in rec_blob
    assert SYNTH_EN_TEXT not in rec_blob


def test_new_id_in_new_blanked(tmp_path: Path) -> None:
    """NEW has row Y not in OLD → output's editable fields are blank."""
    refresh = _load_refresh_module()

    new_cand = _make_candidate(datapoint_id="dp_new_only")
    new_row = _candidate_to_tsv_row(new_cand)
    # Annotate the NEW row to confirm it gets blanked on output.
    annotated_new = _annotate_row(
        new_row, {"de_candidate_decision": "include", "annotator": "X"}
    )

    # OLD is empty (header-only).
    old_tsv = _write_tsv(tmp_path / "old.tsv", [])
    new_tsv = _write_tsv(tmp_path / "new.tsv", [annotated_new])
    out_tsv = tmp_path / "out.tsv"

    rc = refresh.main(
        [
            "--old-tsv", str(old_tsv),
            "--new-tsv", str(new_tsv),
            "--output", str(out_tsv),
        ]
    )
    assert rc == 0

    out_rows = _read_tsv(out_tsv)
    assert len(out_rows) == 1
    out_row = out_rows[0]
    # Editable fields are blank — NEW-side annotations ignored.
    assert out_row["de_candidate_decision"] == ""
    assert out_row["annotator"] == ""
    for col in EDITABLE_COLUMNS:
        assert out_row[col] == BUILDER_DEFAULT_EDITABLE.get(col, ""), col


def test_removed_id_recorded_in_summary(tmp_path: Path) -> None:
    """OLD has row Z not in NEW → summary's removed_ids contains Z."""
    refresh = _load_refresh_module()

    old_cand = _make_candidate(datapoint_id="dp_removed")
    old_row = _annotate_row(
        _candidate_to_tsv_row(old_cand),
        {"de_candidate_decision": "include", "annotation_status": "complete"},
    )
    old_tsv = _write_tsv(tmp_path / "old.tsv", [old_row])

    # NEW is empty.
    new_tsv = _write_tsv(tmp_path / "new.tsv", [])
    out_tsv = tmp_path / "out.tsv"
    summary_path = tmp_path / "out.tsv.summary.json"

    rc = refresh.main(
        [
            "--old-tsv", str(old_tsv),
            "--new-tsv", str(new_tsv),
            "--output", str(out_tsv),
            "--summary", str(summary_path),
        ]
    )
    assert rc == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["removed"] == 1
    assert summary["removed_ids"] == ["dp_removed"]
    assert summary["total_in_new"] == 0
    assert summary["total_in_old"] == 1


def test_refuse_overwrite_without_force_output(tmp_path: Path) -> None:
    """Refuse overwrite when --output exists and --force-output is absent."""
    refresh = _load_refresh_module()

    cand = _make_candidate(datapoint_id="dp_force")
    row = _candidate_to_tsv_row(cand)
    old_tsv = _write_tsv(tmp_path / "old.tsv", [row])
    new_tsv = _write_tsv(tmp_path / "new.tsv", [row])
    # Pre-create output.
    out_tsv = _write_tsv(tmp_path / "out.tsv", [row])

    rc = refresh.main(
        [
            "--old-tsv", str(old_tsv),
            "--new-tsv", str(new_tsv),
            "--output", str(out_tsv),
        ]
    )
    assert rc == 2

    # With --force-output, succeeds.
    rc2 = refresh.main(
        [
            "--old-tsv", str(old_tsv),
            "--new-tsv", str(new_tsv),
            "--output", str(out_tsv),
            "--force-output",
        ]
    )
    assert rc2 == 0


def test_refuse_same_input_paths(tmp_path: Path) -> None:
    """Refuse if --old-tsv and --new-tsv resolve to the same path."""
    refresh = _load_refresh_module()

    cand = _make_candidate(datapoint_id="dp_same")
    row = _candidate_to_tsv_row(cand)
    same = _write_tsv(tmp_path / "same.tsv", [row])
    out_tsv = tmp_path / "out.tsv"

    rc = refresh.main(
        [
            "--old-tsv", str(same),
            "--new-tsv", str(same),
            "--output", str(out_tsv),
        ]
    )
    assert rc == 2


def test_refuse_output_collides_with_input(tmp_path: Path) -> None:
    """Refuse if --output equals either input path."""
    refresh = _load_refresh_module()

    cand = _make_candidate(datapoint_id="dp_collide")
    row = _candidate_to_tsv_row(cand)
    old_tsv = _write_tsv(tmp_path / "old.tsv", [row])
    new_tsv = _write_tsv(tmp_path / "new.tsv", [row])

    # --output == --old-tsv
    rc1 = refresh.main(
        [
            "--old-tsv", str(old_tsv),
            "--new-tsv", str(new_tsv),
            "--output", str(old_tsv),
            "--force-output",
        ]
    )
    assert rc1 == 2

    # --output == --new-tsv
    rc2 = refresh.main(
        [
            "--old-tsv", str(old_tsv),
            "--new-tsv", str(new_tsv),
            "--output", str(new_tsv),
            "--force-output",
        ]
    )
    assert rc2 == 2


def test_multiple_rows_mixed_match_mismatch_new_removed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mix of match/mismatch/new/removed in one call; verify counts."""
    refresh = _load_refresh_module()

    # Three OLD rows, three NEW rows, with all four buckets exercised:
    #   match    : id="dp_match"     — same hash in OLD and NEW
    #   mismatch : id="dp_mismatch"  — different hash
    #   new      : id="dp_new"       — only in NEW
    #   removed  : id="dp_removed"   — only in OLD

    cand_match_old = _make_candidate(datapoint_id="dp_match", token_start=1)
    cand_match_new = _make_candidate(datapoint_id="dp_match", token_start=1)
    assert cand_match_old["source_row_sha256"] == cand_match_new["source_row_sha256"]

    cand_mismatch_old = _make_candidate(
        datapoint_id="dp_mismatch", de_head_lemma="OldLemma"
    )
    cand_mismatch_new = _make_candidate(
        datapoint_id="dp_mismatch", de_head_lemma="NewLemma"
    )
    assert cand_mismatch_old["source_row_sha256"] != cand_mismatch_new["source_row_sha256"]

    cand_removed = _make_candidate(datapoint_id="dp_removed")
    cand_new_only = _make_candidate(datapoint_id="dp_new_only")

    old_rows = [
        _annotate_row(
            _candidate_to_tsv_row(cand_match_old),
            {"de_candidate_decision": "include", "annotator": "OLD_A"},
        ),
        _annotate_row(
            _candidate_to_tsv_row(cand_mismatch_old),
            {"de_candidate_decision": "exclude", "de_exclusion_reason": "duplicate"},
        ),
        _annotate_row(
            _candidate_to_tsv_row(cand_removed),
            {"de_candidate_decision": "include", "annotation_status": "complete"},
        ),
    ]
    new_rows = [
        _candidate_to_tsv_row(cand_match_new),
        _candidate_to_tsv_row(cand_mismatch_new),
        _candidate_to_tsv_row(cand_new_only),
    ]

    old_tsv = _write_tsv(tmp_path / "old.tsv", old_rows)
    new_tsv = _write_tsv(tmp_path / "new.tsv", new_rows)
    out_tsv = tmp_path / "out.tsv"
    ledger = tmp_path / "out.tsv.conflicts.jsonl"
    summary_path = tmp_path / "out.tsv.summary.json"

    rc = refresh.main(
        [
            "--old-tsv", str(old_tsv),
            "--new-tsv", str(new_tsv),
            "--output", str(out_tsv),
            "--conflict-ledger", str(ledger),
            "--summary", str(summary_path),
        ]
    )
    assert rc == 0

    out_rows = _read_tsv(out_tsv)
    assert len(out_rows) == 3
    by_id = {r["datapoint_id"]: r for r in out_rows}

    # match: editable copied from OLD.
    assert by_id["dp_match"]["de_candidate_decision"] == "include"
    assert by_id["dp_match"]["annotator"] == "OLD_A"
    # mismatch: editable blank.
    assert by_id["dp_mismatch"]["de_candidate_decision"] == ""
    assert by_id["dp_mismatch"]["de_exclusion_reason"] == ""
    # new: editable blank.
    assert by_id["dp_new_only"]["de_candidate_decision"] == ""

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["matched"] == 1
    assert summary["hash_mismatched"] == 1
    assert summary["new"] == 1
    assert summary["removed"] == 1
    assert summary["total_in_new"] == 3
    assert summary["total_in_old"] == 3
    assert summary["removed_ids"] == ["dp_removed"]
    assert summary["conflict_ledger"] == ledger.name

    captured = capsys.readouterr()
    assert "matched: 1" in captured.out
    assert "hash_mismatched: 1" in captured.out
    assert "new: 1" in captured.out
    assert "removed: 1" in captured.out
    assert "total_in_new: 3" in captured.out
    assert "total_in_old: 3" in captured.out


def test_builder_defaults_applied_to_blanked_rows(tmp_path: Path) -> None:
    """Blanked rows have en_alignment_qc=assumed_ok, zh_alignment_qc=assumed_ok,
    NOT empty string."""
    refresh = _load_refresh_module()

    # Two NEW rows, neither in OLD → both must blank out with builder defaults.
    cand_a = _make_candidate(datapoint_id="dp_a")
    cand_b = _make_candidate(datapoint_id="dp_b")

    # Pre-annotate the NEW rows so we can verify they get blanked on output.
    new_rows = [
        _annotate_row(
            _candidate_to_tsv_row(cand_a),
            {"en_alignment_qc": "confirmed", "zh_alignment_qc": "incorrect"},
        ),
        _annotate_row(
            _candidate_to_tsv_row(cand_b),
            {"en_alignment_qc": "confirmed", "zh_alignment_qc": "incorrect"},
        ),
    ]
    old_tsv = _write_tsv(tmp_path / "old.tsv", [])  # empty
    new_tsv = _write_tsv(tmp_path / "new.tsv", new_rows)
    out_tsv = tmp_path / "out.tsv"

    rc = refresh.main(
        [
            "--old-tsv", str(old_tsv),
            "--new-tsv", str(new_tsv),
            "--output", str(out_tsv),
        ]
    )
    assert rc == 0

    out_rows = _read_tsv(out_tsv)
    assert len(out_rows) == 2
    for r in out_rows:
        assert r["en_alignment_qc"] == "assumed_ok"
        assert r["zh_alignment_qc"] == "assumed_ok"
        # Other editable columns stay blank.
        assert r["de_candidate_decision"] == ""
        assert r["annotator"] == ""
        assert r["en_form"] == ""


def test_stdout_privacy_guard(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Capture stdout via capsys, assert no synthetic-text fixture strings appear."""
    refresh = _load_refresh_module()

    cand = _make_candidate(datapoint_id="dp_priv")
    new_row = _candidate_to_tsv_row(cand)
    annotated_old = _annotate_row(
        new_row,
        {
            "de_candidate_decision": "include",
            "annotator": "annotator_alpha",
            "en_form": "definite",
            "en_span_text": "synthetic span text",
            "general_notes": "synth general notes",
        },
    )

    old_tsv = _write_tsv(tmp_path / "old.tsv", [annotated_old])
    new_tsv = _write_tsv(tmp_path / "new.tsv", [new_row])
    out_tsv = tmp_path / "out.tsv"

    rc = refresh.main(
        [
            "--old-tsv", str(old_tsv),
            "--new-tsv", str(new_tsv),
            "--output", str(out_tsv),
        ]
    )
    assert rc == 0

    captured = capsys.readouterr()
    stdout = captured.out

    # The fixture-text sentinels must not leak to stdout.
    for forbidden in (
        SYNTH_DE_TEXT,
        SYNTH_EN_TEXT,
        SYNTH_ZH_TEXT,
        SYNTH_LEMMA,
        "synthetic span text",
        "synth general notes",
        "annotator_alpha",
        "definite",
        # Hashes must never appear on stdout.
        cand["source_row_sha256"],
    ):
        assert forbidden not in stdout, f"stdout leaked: {forbidden!r}"

    # Datapoint IDs are also forbidden on stdout.
    assert "dp_priv" not in stdout
