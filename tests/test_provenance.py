"""Synthetic-fixture tests for block-level parse provenance.

Covers the ID construction rule, the deterministic text migration, the
fail-closed validators, and the extraction-level provenance gate. All
fixtures are synthetic — no corpus text.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hp_corpus.german_extraction import (  # noqa: E402
    ProvenanceDuplicateError,
    ProvenanceInconsistentError,
    ProvenanceMissingError,
    extract_chapter,
)
from hp_corpus.provenance import (  # noqa: E402
    DuplicateParseBlockIdError,
    InconsistentProvenanceError,
    MalformedParseBlockIdError,
    MissingProvenanceError,
    is_parse_block_id,
    make_parse_block_id,
    migrate_conllu_text,
    provenance_counts,
    scan_blocks,
    split_parse_block_id,
    validate_conllu_text,
)

SEG = "hp1_de_ch01_p0001_s001"


def block(sid: str, tokens: str, *, text: str = "x") -> str:
    return f"# sent_id = {sid}\n# text = {text}\n{tokens}\n\n"


TOKENS = "1\tHi\thi\tINTJ\tITJ\t_\t0\troot\t_\t_\n"


def migrated_block(pid: str, source: str, tokens: str = TOKENS, *, text: str = "x") -> str:
    return f"# sent_id = {pid}\n# source_segment_id = {source}\n# text = {text}\n{tokens}\n\n"


# ---------------------------------------------------------------------------
# ID construction rule.
# ---------------------------------------------------------------------------


def test_make_and_split_roundtrip() -> None:
    pid = make_parse_block_id(SEG, 1)
    assert pid == f"{SEG}#b001"
    assert split_parse_block_id(pid) == (SEG, 1)
    assert split_parse_block_id(make_parse_block_id(SEG, 12)) == (SEG, 12)


def test_make_rejects_nonpositive_ordinal() -> None:
    with pytest.raises(ValueError):
        make_parse_block_id(SEG, 0)


def test_split_rejects_raw_segment_id() -> None:
    with pytest.raises(MalformedParseBlockIdError):
        split_parse_block_id(SEG)
    assert not is_parse_block_id(SEG)
    assert is_parse_block_id(f"{SEG}#b001")


# ---------------------------------------------------------------------------
# Migration: multiple blocks from one source segment.
# ---------------------------------------------------------------------------


def test_migrate_multi_block_segment() -> None:
    text = block(SEG, TOKENS) + block(SEG, TOKENS) + block("hp1_de_ch01_p0002_s001", TOKENS)
    out, stats = migrate_conllu_text(text)
    assert "# sent_id = hp1_de_ch01_p0001_s001#b001" in out
    assert "# sent_id = hp1_de_ch01_p0001_s001#b002" in out
    assert "# sent_id = hp1_de_ch01_p0002_s001#b001" in out
    assert out.count(f"# source_segment_id = {SEG}") == 2
    assert stats.blocks_total == 3
    assert stats.blocks_migrated == 3
    assert stats.distinct_sent_ids_before == 2
    assert stats.multi_block_segments == 1
    validate_conllu_text(out)


def test_migrate_single_block_segment() -> None:
    out, stats = migrate_conllu_text(block(SEG, TOKENS))
    assert f"# sent_id = {SEG}#b001" in out
    assert stats.blocks_migrated == 1
    assert stats.multi_block_segments == 0


def test_migrate_is_idempotent_and_deterministic() -> None:
    text = block(SEG, TOKENS) + block(SEG, TOKENS)
    once, _ = migrate_conllu_text(text)
    twice, stats2 = migrate_conllu_text(once)
    assert once == twice
    assert stats2.blocks_migrated == 0
    assert stats2.blocks_already_migrated == 2
    # Deterministic regeneration: same input → byte-identical output.
    assert migrate_conllu_text(text)[0] == migrate_conllu_text(text)[0]


def test_migrate_refuses_mixed_state() -> None:
    migrated = block(f"{SEG}#b001", TOKENS)
    with pytest.raises(InconsistentProvenanceError):
        migrate_conllu_text(migrated + block("hp1_de_ch01_p0002_s001", TOKENS))


def test_migration_matches_expected_parse_emission() -> None:
    """The text migration must produce the ids parse.py would emit."""
    migrated, _ = migrate_conllu_text(block(SEG, TOKENS) + block(SEG, TOKENS))
    assert make_parse_block_id(SEG, 1) in migrated
    assert make_parse_block_id(SEG, 2) in migrated


# ---------------------------------------------------------------------------
# Fail-closed validation of migrated files.
# ---------------------------------------------------------------------------


def test_validate_fails_on_raw_segment_sent_id() -> None:
    with pytest.raises(MissingProvenanceError):
        validate_conllu_text(block(SEG, TOKENS))


def test_validate_fails_on_missing_source_segment_comment() -> None:
    text = f"# sent_id = {SEG}#b001\n# text = x\n{TOKENS}\n\n"
    with pytest.raises(MissingProvenanceError):
        validate_conllu_text(text)


def test_validate_fails_on_duplicate_parse_block_id() -> None:
    text = migrated_block(f"{SEG}#b001", SEG) + migrated_block(f"{SEG}#b001", SEG)
    with pytest.raises(DuplicateParseBlockIdError):
        validate_conllu_text(text)


def test_validate_fails_on_prefix_mismatch() -> None:
    text = (
        f"# sent_id = {SEG}#b001\n# source_segment_id = hp1_de_ch01_p0002_s001\n"
        f"# text = x\n{TOKENS}\n\n"
    )
    with pytest.raises(InconsistentProvenanceError):
        validate_conllu_text(text)


def test_validate_fails_on_ordinal_gap() -> None:
    # First block of the segment is numbered #b002 → ordinal 2 ≠ expected 1.
    with pytest.raises(InconsistentProvenanceError):
        validate_conllu_text(migrated_block(f"{SEG}#b002", SEG))


def test_provenance_counts() -> None:
    migrated, _ = migrate_conllu_text(
        block(SEG, TOKENS) + block(SEG, TOKENS) + block("hp1_de_ch01_p0002_s001", TOKENS)
    )
    counts = provenance_counts(scan_blocks(migrated.splitlines()))
    assert counts == {
        "blocks_total": 3,
        "distinct_parse_block_ids": 3,
        "distinct_source_segments": 2,
        "multi_block_segments": 1,
        "max_blocks_per_segment": 2,
    }


# ---------------------------------------------------------------------------
# Extraction-level fail-closed gate (synthetic CoNLL-U, monkeypatched lists).
# ---------------------------------------------------------------------------


def _synth_pp_conllu(sent_id: str, *, source: str | None = None, with_source: bool = True) -> str:
    source_line = (
        f"# source_segment_id = {source if source is not None else sent_id}\n"
        if with_source
        else ""
    )
    return (
        f"# sent_id = {sent_id}\n"
        f"{source_line}"
        "# text = im Haus\n"
        "1\tim\tim\tADP\tAPPR\t_\t2\tcase\t_\t_\n"
        "2\tHaus\tHaus\tNOUN\tNN\t_\t0\troot\t_\t_\n"
        "\n"
    )


@pytest.fixture()
def synth_lists(monkeypatch: pytest.MonkeyPatch):
    import hp_corpus.german_extraction as ge

    monkeypatch.setattr(ge, "CONTRACTED", ["im"])
    monkeypatch.setattr(ge, "PREPOSITIONS", ["in"])
    monkeypatch.setattr(ge, "DETERMINERS", ["dem"])


def test_extract_propagates_block_provenance(tmp_path: Path, synth_lists) -> None:
    sid = "hp1_de_ch01_p0001_s001"
    migrated = migrate_conllu_text(_synth_pp_conllu(sid))[0]
    path = tmp_path / "hp1_de_ch01_nomwt.conllu"
    path.write_text(migrated, encoding="utf-8")
    hits = extract_chapter(path, contracted=True)
    assert len(hits) == 1
    assert hits[0]["parse_block_id"] == f"{sid}#b001"
    assert hits[0]["source_segment_id"] == sid


def test_extract_fails_closed_on_unmigrated_input(
    tmp_path: Path, synth_lists
) -> None:
    path = tmp_path / "hp1_de_ch01_nomwt.conllu"
    path.write_text(_synth_pp_conllu(SEG), encoding="utf-8")
    with pytest.raises(ProvenanceMissingError):
        extract_chapter(path, contracted=True)


def test_extract_fails_closed_on_missing_source_comment(
    tmp_path: Path, synth_lists
) -> None:
    path = tmp_path / "hp1_de_ch01_nomwt.conllu"
    path.write_text(_synth_pp_conllu(f"{SEG}#b001", with_source=False), encoding="utf-8")
    with pytest.raises(ProvenanceMissingError):
        extract_chapter(path, contracted=True)


def test_extract_fails_closed_on_duplicate_block_id(
    tmp_path: Path, synth_lists
) -> None:
    dup = _synth_pp_conllu(f"{SEG}#b001", source=SEG) + _synth_pp_conllu(
        f"{SEG}#b001", source=SEG
    )
    path = tmp_path / "hp1_de_ch01_nomwt.conllu"
    path.write_text(dup, encoding="utf-8")
    with pytest.raises(ProvenanceDuplicateError):
        extract_chapter(path, contracted=True)


def test_extract_fails_closed_on_prefix_mismatch(tmp_path: Path, synth_lists) -> None:
    text = _synth_pp_conllu(f"{SEG}#b001", source="hp1_de_ch01_p0002_s001")
    path = tmp_path / "hp1_de_ch01_nomwt.conllu"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ProvenanceInconsistentError):
        extract_chapter(path, contracted=True)


def test_extract_fails_closed_on_ordinal_zero(tmp_path: Path, synth_lists) -> None:
    text = _synth_pp_conllu(f"{SEG}#b000", source=SEG)
    path = tmp_path / "hp1_de_ch01_nomwt.conllu"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ProvenanceInconsistentError):
        extract_chapter(path, contracted=True)


def test_extract_fails_closed_on_ordinal_gap(tmp_path: Path, synth_lists) -> None:
    # First block of the segment is #b002 → gap at ordinal 1.
    text = _synth_pp_conllu(f"{SEG}#b002", source=SEG)
    path = tmp_path / "hp1_de_ch01_nomwt.conllu"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ProvenanceInconsistentError):
        extract_chapter(path, contracted=True)


def test_extract_fails_closed_on_out_of_order_ordinals(
    tmp_path: Path, synth_lists
) -> None:
    text = _synth_pp_conllu(f"{SEG}#b002", source=SEG) + _synth_pp_conllu(
        f"{SEG}#b001", source=SEG
    )
    path = tmp_path / "hp1_de_ch01_nomwt.conllu"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ProvenanceInconsistentError):
        extract_chapter(path, contracted=True)


# ---------------------------------------------------------------------------
# Physical blocks with token content but no # sent_id (fail closed).
# ---------------------------------------------------------------------------

ORPHAN_TOKENS = "1\tNeu\tneu\tADJ\tADJA\t_\t0\troot\t_\t_\n"


def _orphan_after_good_block() -> str:
    """A valid migrated block followed by a physical block that has a
    comment header and token content but no ``# sent_id`` line."""
    return migrated_block(f"{SEG}#b001", SEG) + (
        f"# text = header without sent id\n{ORPHAN_TOKENS}\n\n"
    )


def test_scan_records_content_block_without_sent_id() -> None:
    blocks = scan_blocks(_orphan_after_good_block().splitlines())
    assert len(blocks) == 2
    orphan = blocks[1]
    assert orphan.orphan_content is True
    assert orphan.sent_id == ""
    # line_index points at the first token line of the orphan block.
    assert _orphan_after_good_block().splitlines()[orphan.line_index].startswith("1\t")


def test_scan_one_record_per_orphan_physical_block() -> None:
    """Consecutive token lines with no blank line between are ONE
    physical orphan block — a single orphan record, not one per line."""
    text = migrated_block(f"{SEG}#b001", SEG) + (
        f"# text = header without sent id\n{ORPHAN_TOKENS}"
        f"2\tNoch\tnoch\tADV\tADV\t_\t1\tadvmod\t_\t_\n\n"
    )
    blocks = scan_blocks(text.splitlines())
    orphans = [b for b in blocks if b.orphan_content]
    assert len(orphans) == 1
    assert len(blocks) == 2


def test_validate_fails_on_content_block_without_sent_id() -> None:
    with pytest.raises(MissingProvenanceError, match=r"no # sent_id"):
        validate_conllu_text(_orphan_after_good_block())


def test_migrate_refuses_content_block_without_sent_id() -> None:
    with pytest.raises(MissingProvenanceError, match=r"no # sent_id"):
        migrate_conllu_text(_orphan_after_good_block())


def test_parse_conllu_fails_on_content_block_without_sent_id(
    tmp_path: Path,
) -> None:
    from hp_corpus.crosslingual_map import parse_conllu

    path = tmp_path / "hp1_en_ch01.conllu"
    path.write_text(_orphan_after_good_block(), encoding="utf-8")
    with pytest.raises(MissingProvenanceError, match=r"no # sent_id"):
        parse_conllu(path)


def test_extract_fails_closed_on_content_block_without_sent_id(
    tmp_path: Path, synth_lists
) -> None:
    path = tmp_path / "hp1_de_ch01_nomwt.conllu"
    path.write_text(
        _synth_pp_conllu(f"{SEG}#b001", source=SEG)
        + f"# text = header without sent id\n{ORPHAN_TOKENS}\n\n",
        encoding="utf-8",
    )
    with pytest.raises(ProvenanceMissingError):
        extract_chapter(path, contracted=True)
