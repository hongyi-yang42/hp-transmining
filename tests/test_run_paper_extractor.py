"""Tests for the deprecated ``run_paper_extractor.py`` wrapper.

The script is now a thin delegate to
``run_full_novel_german_extraction.py`` (one extraction implementation,
``hp_corpus.german_extraction.extract_chapter``). These tests pin:

  * the delegation is wired and announces its deprecation on stderr;
  * stdout still carries aggregate counts only — never noun lemmas or
    any other token-level data from the source text;
  * the TSVs emitted through the documented path carry BOTH provenance
    columns (``parse_block_id`` + ``source_segment_id``) plus the
    occurrence-coordinate fields;
  * two PPs sharing a parse block stay distinguishable by coordinates.

Fixtures are migrated CoNLL-U (block-level provenance present). The
vendor ``conll-extractor`` is monkeypatched with synthetic lists so the
suite runs in CI without the clone.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_paper_extractor.py"

# Two synthetic migrated blocks. Nouns: "Haus" (contracted PP),
# "Wald" (uncontracted PP).
SYNTHETIC_CONLLU = """\
# sent_id = synthetic_contracted#b001
# source_segment_id = synthetic_contracted
# text = im Haus
1\tim\tim\tADP\tAPPR\t_\t2\tcase\t_\t_
2\tHaus\tHaus\tNOUN\tNN\t_\t0\troot\t_\t_

# sent_id = synthetic_uncontracted#b001
# source_segment_id = synthetic_uncontracted
# text = in dem Wald
1\tin\tin\tADP\tAPPR\t_\t3\tcase\t_\t_
2\tdem\tder\tDET\tART\t_\t3\tdet\t_\t_
3\tWald\tWald\tNOUN\tNN\t_\t0\troot\t_\t_
"""

# One migrated block with two occurrences of the same (prep, noun) pair,
# distinguishable only by token coordinates.
SYNTHETIC_TWO_SAME_CONLLU = """\
# sent_id = synthetic_two_same#b001
# source_segment_id = synthetic_two_same
# text = im Haus und im Haus
1\tim\tim\tADP\tAPPR\t_\t2\tcase\t_\t_
2\tHaus\tHaus\tNOUN\tNN\t_\t0\troot\t_\t_
3\tund\tund\tCCONJ\tKON\t_\t5\tcc\t_\t_
4\tim\tim\tADP\tAPPR\t_\t5\tcase\t_\t_
5\tHaus\tHaus\tNOUN\tNN\t_\t2\tconj\t_\t_
"""


@pytest.fixture
def run_paper_extractor_module(monkeypatch: pytest.MonkeyPatch):
    from hp_corpus import german_extraction as ge

    monkeypatch.setattr(ge, "CONTRACTED", ["im"])
    monkeypatch.setattr(ge, "PREPOSITIONS", ["in"])
    monkeypatch.setattr(ge, "DETERMINERS", ["dem"])
    monkeypatch.setattr(ge, "FILTER_CONTRACTED_123", {"im": ["Haus"]})
    monkeypatch.setattr(ge, "FILTER_PP", {"in": ["Wald"]})

    spec = importlib.util.spec_from_file_location("run_paper_extractor", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _run(module, tmp_path, conllu: str):
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    (parsed_dir / "hp1_de_ch01_nomwt.conllu").write_text(conllu, encoding="utf-8")
    out_dir = tmp_path / "out"
    # Explicit --chapters: the delegate's legacy default (1 2 3) is
    # fail-closed under the new CLI when only Ch.1 exists on disk.
    rc = module.main(
        ["--chapters", "1", "--parsed-dir", str(parsed_dir), "--output-dir", str(out_dir)]
    )
    assert rc == 0
    return out_dir


def test_delegates_and_announces_deprecation(
    tmp_path, capsys, run_paper_extractor_module
):
    out_dir = _run(run_paper_extractor_module, tmp_path, SYNTHETIC_CONLLU)
    captured = capsys.readouterr()
    # The delegated CLI produced the new-schema TSVs + manifest.
    assert (out_dir / "hp1_de_ch01_contracted.tsv").exists()
    assert (out_dir / "hp1_de_ch01_uncontracted.tsv").exists()
    assert (out_dir / "manifest.json").exists()
    # Depprecation notice on stderr, not stdout.
    assert "DEPRECATED" in captured.err
    assert "run_full_novel_german_extraction" in captured.err


def test_stdout_no_noun_lemmas(tmp_path, capsys, run_paper_extractor_module):
    _run(run_paper_extractor_module, tmp_path, SYNTHETIC_CONLLU)
    captured = capsys.readouterr().out

    # Aggregate counts ARE expected on stdout
    assert "contracted" in captured
    assert "uncontracted" in captured
    # Noun lemmas must NOT leak to stdout
    assert "Haus" not in captured
    assert "Wald" not in captured
    # Per-prep noun-list patterns must not reappear
    assert "by preposition" not in captured
    assert "[:6]" not in captured


def test_tsv_receives_lemma_and_both_provenance_columns(
    tmp_path, run_paper_extractor_module
):
    """The documented extraction path emits parse_block_id AND
    source_segment_id on every row (plus the noun lemma in the TSV)."""
    out_dir = _run(run_paper_extractor_module, tmp_path, SYNTHETIC_CONLLU)

    contracted_tsv = out_dir / "hp1_de_ch01_contracted.tsv"
    assert contracted_tsv.exists()
    rows = list(
        csv.DictReader(contracted_tsv.open(encoding="utf-8"), delimiter="\t")
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["parse_block_id"] == "synthetic_contracted#b001"
    assert row["source_segment_id"] == "synthetic_contracted"
    assert row["noun"] == "Haus"  # the noun lemma IS in the TSV
    # The header carries the coordinate fields.
    tsv_content = contracted_tsv.read_text(encoding="utf-8")
    for col in (
        "parse_block_id",
        "source_segment_id",
        "prep_token_id",
        "det_token_id",
        "noun_token_id",
        "pp_token_start",
        "pp_token_end",
        "pp_surface",
    ):
        assert col in tsv_content


def test_tsv_carries_occurrence_coordinates(tmp_path, run_paper_extractor_module):
    """Coordinate fields populate correctly for contracted and uncontracted PPs."""
    out_dir = _run(run_paper_extractor_module, tmp_path, SYNTHETIC_CONLLU)

    # Contracted: prep at id 1, noun at id 2, no det.
    contracted_rows = list(
        csv.DictReader(
            (out_dir / "hp1_de_ch01_contracted.tsv").open(encoding="utf-8"),
            delimiter="\t",
        )
    )
    assert len(contracted_rows) == 1
    row = contracted_rows[0]
    assert row["prep_token_id"] == "1"
    assert row["noun_token_id"] == "2"
    assert row["det_token_id"] == "-"  # placeholder for contracted
    assert row["pp_token_start"] == "1"
    assert row["pp_token_end"] == "2"
    assert row["pp_surface"] == "im Haus"

    # Uncontracted: prep at id 1, det at id 2, noun at id 3.
    unc_rows = list(
        csv.DictReader(
            (out_dir / "hp1_de_ch01_uncontracted.tsv").open(encoding="utf-8"),
            delimiter="\t",
        )
    )
    assert len(unc_rows) == 1
    row = unc_rows[0]
    assert row["prep_token_id"] == "1"
    assert row["det_token_id"] == "2"
    assert row["noun_token_id"] == "3"
    assert row["pp_token_start"] == "1"
    assert row["pp_token_end"] == "3"
    assert row["pp_surface"] == "in dem Wald"


def test_two_occurrences_same_prep_noun_are_distinguishable(
    tmp_path, run_paper_extractor_module
):
    """Two occurrences of (im, Haus) within one block must remain separate
    rows with different token coordinates — the extractor must NOT deduplicate."""
    out_dir = _run(run_paper_extractor_module, tmp_path, SYNTHETIC_TWO_SAME_CONLLU)

    rows = list(
        csv.DictReader(
            (out_dir / "hp1_de_ch01_contracted.tsv").open(encoding="utf-8"),
            delimiter="\t",
        )
    )
    assert len(rows) == 2, f"expected 2 occurrences, got {len(rows)}"
    # Same (prep, noun, parse_block_id) but different coordinates.
    assert rows[0]["parse_block_id"] == rows[1]["parse_block_id"]
    assert rows[0]["prep"] == rows[1]["prep"] == "im"
    assert rows[0]["noun"] == rows[1]["noun"] == "Haus"
    coords = {(r["prep_token_id"], r["noun_token_id"]) for r in rows}
    assert coords == {("1", "2"), ("4", "5")}, coords
    # Both surfaces reconstruct correctly from their own ranges.
    surfaces = {r["pp_surface"] for r in rows}
    assert surfaces == {"im Haus"}


def test_pp_surface_reconstructs_only_within_token_range(
    tmp_path, run_paper_extractor_module
):
    """pp_surface must be reconstructable from the inclusive token range, even
    when the range contains tokens between prep and noun (e.g. an adjective
    line that the extractor doesn't track)."""
    conllu = (
        "# sent_id = synthetic_with_adj#b001\n"
        "# source_segment_id = synthetic_with_adj\n"
        "# text = im großen Haus\n"
        "1\tim\tim\tADP\tAPPR\t_\t3\tcase\t_\t_\n"
        "2\tgroßen\tgroß\tADJ\tADJA\t_\t3\tamod\t_\t_\n"
        "3\tHaus\tHaus\tNOUN\tNN\t_\t0\troot\t_\t_\n"
    )
    out_dir = _run(run_paper_extractor_module, tmp_path, conllu)

    rows = list(
        csv.DictReader(
            (out_dir / "hp1_de_ch01_contracted.tsv").open(encoding="utf-8"),
            delimiter="\t",
        )
    )
    assert len(rows) == 1
    # pp_token_start=1, pp_token_end=3 → surface spans all 3 tokens
    assert rows[0]["pp_token_start"] == "1"
    assert rows[0]["pp_token_end"] == "3"
    assert rows[0]["pp_surface"] == "im großen Haus"
