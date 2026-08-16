"""Regression tests for the cross-lingual parsed-sentence join.

After the block-level provenance migration, parsed files are keyed by
``parse_block_id`` while Step 4 candidates carry alignment-level
``{lang}_sentence_ids`` (source-segment ids). The join in
``scripts/map_crosslingual_pps.py`` must resolve an aligned source
segment to **all** of its parse blocks in document order — a migrated
EN/ZH segment the parser split into several blocks must not silently
disappear from the PP mapper. All fixtures are synthetic.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from hp_corpus.crosslingual_map import index_blocks_by_segment

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "map_crosslingual_pps.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("map_crosslingual_pps", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# One EN source segment split into two parse blocks (a PP in each), plus
# a second single-block segment.
EN_CONLLU = (
    "# sent_id = hp1_en_ch01_p0001_s001#b001\n"
    "# source_segment_id = hp1_en_ch01_p0001_s001\n"
    "# text = in house\n"
    "1\tin\tin\tADP\tIN\t_\t2\tcase\t_\t_\n"
    "2\thouse\thouse\tNOUN\tNN\t_\t0\troot\t_\t_\n"
    "\n"
    "# sent_id = hp1_en_ch01_p0001_s001#b002\n"
    "# source_segment_id = hp1_en_ch01_p0001_s001\n"
    "# text = on hill\n"
    "1\ton\ton\tADP\tIN\t_\t2\tcase\t_\t_\n"
    "2\thill\thill\tNOUN\tNN\t_\t0\troot\t_\t_\n"
    "\n"
    "# sent_id = hp1_en_ch01_p0002_s001#b001\n"
    "# source_segment_id = hp1_en_ch01_p0002_s001\n"
    "# text = at bay\n"
    "1\tat\tat\tADP\tIN\t_\t2\tcase\t_\t_\n"
    "2\tbay\tbay\tNOUN\tNN\t_\t0\troot\t_\t_\n"
    "\n"
)

ZH_CONLLU = (
    "# sent_id = hp1_zh_ch01_p0001_s001#b001\n"
    "# source_segment_id = hp1_zh_ch01_p0001_s001\n"
    "# text = synth zh\n"
    "1\t在\t在\tADP\tADP\t_\t2\tcase\t_\t_\n"
    "2\t屋\t屋\tNOUN\tNN\t_\t0\troot\t_\t_\n"
    "\n"
)


def _write_parsed(tmp_path: Path) -> Path:
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    (parsed / "hp1_en_ch01.conllu").write_text(EN_CONLLU, encoding="utf-8")
    (parsed / "hp1_zh_ch01.conllu").write_text(ZH_CONLLU, encoding="utf-8")
    return parsed


def _cand(en_ids: list[str], zh_ids: list[str]) -> dict:
    return {
        "datapoint_id": "dp_ch01_hp1_de_ch01_p0001_s001#b001_t1-2",
        "chapter": 1,
        "de_parse_block_id": "hp1_de_ch01_p0001_s001#b001",
        "de_source_segment_id": "hp1_de_ch01_p0001_s001",
        "de_token_start": 1,
        "de_token_end": 2,
        "de_pp_surface": "im Haus",
        "de_prep_normalized": "in",
        "de_head_lemma": "Haus",
        "de_form": "contracted",
        "author_resource_match": True,
        "en_alignment_cardinality": "1:1",
        "en_alignment_confidence": 0.95,
        "zh_alignment_cardinality": "1:1",
        "zh_alignment_confidence": 0.85,
        "en_sentence_ids": en_ids,
        "zh_sentence_ids": zh_ids,
    }


def test_multi_block_source_segment_reaches_the_mapper(tmp_path: Path) -> None:
    """Alignment id = source segment; the mapper must see the PPs from
    BOTH of its parse blocks, in document order."""
    script = _load_script()
    parsed = _write_parsed(tmp_path)

    en_parsed = script._load_parsed_chapters(parsed, "en", [1])
    zh_parsed = script._load_parsed_chapters(parsed, "zh", [1])
    en_by_segment = index_blocks_by_segment(en_parsed)
    zh_by_segment = index_blocks_by_segment(zh_parsed)

    # The alignment-level lookup returns both blocks of the split segment.
    assert [s.parse_block_id for s in en_by_segment["hp1_en_ch01_p0001_s001"]] == [
        "hp1_en_ch01_p0001_s001#b001",
        "hp1_en_ch01_p0001_s001#b002",
    ]

    record = script.build_mapping_record(
        _cand(["hp1_en_ch01_p0001_s001"], ["hp1_zh_ch01_p0001_s001"]),
        en_by_segment,
        zh_by_segment,
    )

    # Both EN blocks were passed to the mapper.
    assert record["n_en_sentences_seen"] == 2
    assert record["n_zh_sentences_seen"] == 1
    assert record["n_en_alignment_ids_unresolved"] == 0
    assert record["n_zh_alignment_ids_unresolved"] == 0
    # Candidates from both blocks were considered: the best/alternatives
    # cover the two distinct prep surfaces ("in" from #b001, "on" from
    # #b002) — one block alone cannot produce this set.
    surfaces = {record["en_best"]["prep_surface"]}
    surfaces |= {a["pp"]["prep_surface"] for a in record["en_alternatives"]}
    assert surfaces == {"in", "on"}
    assert record["en_n_candidates"] == 2


def test_unresolvable_alignment_id_is_counted_not_dropped(tmp_path: Path) -> None:
    """An alignment id resolving to zero blocks is counted in the record
    (visible in the summary), never silently absorbed."""
    script = _load_script()
    parsed = _write_parsed(tmp_path)

    en_by_segment = index_blocks_by_segment(script._load_parsed_chapters(parsed, "en", [1]))
    zh_by_segment = index_blocks_by_segment(script._load_parsed_chapters(parsed, "zh", [1]))

    record = script.build_mapping_record(
        _cand(["hp1_en_ch01_p0009_s999"], ["hp1_zh_ch01_p0009_s999"]),
        en_by_segment,
        zh_by_segment,
    )
    assert record["n_en_sentences_seen"] == 0
    assert record["n_zh_sentences_seen"] == 0
    assert record["n_en_alignment_ids_unresolved"] == 1
    assert record["n_zh_alignment_ids_unresolved"] == 1
    assert record["en_status"] == "unmappable"
    assert record["zh_status"] == "unmappable"


def test_load_parsed_chapters_fails_closed_on_block_id_collision(tmp_path: Path) -> None:
    """The same parse_block_id in two chapter files means corrupt input;
    loading must raise instead of silently overwriting one block."""
    script = _load_script()
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    dup = (
        "# sent_id = hp1_en_ch01_p0001_s001#b001\n"
        "# source_segment_id = hp1_en_ch01_p0001_s001\n"
        "# text = in house\n"
        "1\tin\tin\tADP\tIN\t_\t2\tcase\t_\t_\n"
        "2\thouse\thouse\tNOUN\tNN\t_\t0\troot\t_\t_\n"
        "\n"
    )
    # Segment ids embed the chapter, so a genuine collision requires a
    # malformed id — simulate by putting the same id in two chapter files.
    (parsed / "hp1_en_ch01.conllu").write_text(dup, encoding="utf-8")
    (parsed / "hp1_en_ch02.conllu").write_text(dup, encoding="utf-8")
    try:
        script._load_parsed_chapters(parsed, "en", [1, 2])
    except ValueError as exc:
        assert "duplicate parse_block_ids" in str(exc)
    else:
        raise AssertionError("expected ValueError on block-id collision")
