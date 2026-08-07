"""Regression guard: run_paper_extractor.py stdout must contain only
aggregate counts, never noun lemmas (or any other token-level data)
from the source text.

Also verifies the occurrence-coordinate fields added in Step 4: each
hit carries CoNLL-U token IDs and a reconstructable ``pp_surface`` so
two PPs that share (preposition, noun) within one sentence can be told
apart.

The script normally depends on the gitignored vendor/conll-extractor.
We inject a synthetic conll_extractor package via sys.modules so this
test runs in CI without the vendor.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_paper_extractor.py"


def _install_fake_vendor() -> None:
    fake_data = ModuleType("conll_extractor.prepositions.data")
    fake_data.CONTRACTED = ["im"]
    fake_data.PREPOSITIONS = ["in"]
    fake_data.DETERMINERS = ["dem"]
    fake_data.FILTER_CONTRACTED_123 = {"im": ["Haus"]}
    fake_data.FILTER_PP = {"in": ["Wald"]}

    fake_prepositions = ModuleType("conll_extractor.prepositions")
    fake_prepositions.data = fake_data

    fake_pkg = ModuleType("conll_extractor")
    fake_pkg.prepositions = fake_prepositions
    fake_pkg.__path__ = []  # mark as package

    sys.modules["conll_extractor"] = fake_pkg
    sys.modules["conll_extractor.prepositions"] = fake_prepositions
    sys.modules["conll_extractor.prepositions.data"] = fake_data


# Two synthetic sentences. Nouns: "Haus" (contracted PP), "Wald" (uncontracted PP).
SYNTHETIC_CONLLU = """\
# sent_id = synthetic_contracted
# text = im Haus
1\tim\tim\tADP\tAPPR\t_\t2\tcase\t_\t_
2\tHaus\tHaus\tNOUN\tNN\t_\t0\troot\t_\t_

# sent_id = synthetic_uncontracted
# text = in dem Wald
1\tin\tin\tADP\tAPPR\t_\t3\tcase\t_\t_
2\tdem\tder\tDET\tART\t_\t3\tdet\t_\t_
3\tWald\tWald\tNOUN\tNN\t_\t0\troot\t_\t_
"""

# Sentence with two occurrences of the same (prep, noun) pair, distinguishable
# only by token coordinates.
SYNTHETIC_TWO_SAME_CONLLU = """\
# sent_id = synthetic_two_same
# text = im Haus und im Haus
1\tim\tim\tADP\tAPPR\t_\t2\tcase\t_\t_
2\tHaus\tHaus\tNOUN\tNN\t_\t0\troot\t_\t_
3\tund\tund\tCCONJ\tKON\t_\t5\tcc\t_\t_
4\tim\tim\tADP\tAPPR\t_\t5\tcase\t_\t_
5\tHaus\tHaus\tNOUN\tNN\t_\t2\tconj\t_\t_
"""


@pytest.fixture
def run_paper_extractor_module():
    saved = {k: sys.modules[k] for k in list(sys.modules) if k.startswith("conll_extractor")}
    _install_fake_vendor()
    spec = importlib.util.spec_from_file_location("run_paper_extractor", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        for k in [k for k in sys.modules if k.startswith("conll_extractor")]:
            del sys.modules[k]
        sys.modules.update(saved)


def test_stdout_no_noun_lemmas(tmp_path, capsys, run_paper_extractor_module):
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    (parsed_dir / "hp1_de_ch01_nomwt.conllu").write_text(SYNTHETIC_CONLLU, encoding="utf-8")
    out_dir = tmp_path / "out"

    rc = run_paper_extractor_module.main(
        ["--parsed-dir", str(parsed_dir), "--output-dir", str(out_dir)]
    )
    captured = capsys.readouterr().out

    assert rc == 0
    # Aggregate counts ARE expected on stdout
    assert "contracted" in captured
    assert "uncontracted" in captured
    assert "COMBINED Ch.1-3" in captured
    # Noun lemmas must NOT leak to stdout
    assert "Haus" not in captured
    assert "Wald" not in captured
    # Per-prep noun-list patterns must not reappear
    assert "by preposition" not in captured
    assert "[:6]" not in captured


def test_tsv_receives_noun_lemmas(tmp_path, run_paper_extractor_module):
    """Counterpart to the stdout guard: noun lemmas DO go to the TSV."""
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    (parsed_dir / "hp1_de_ch01_nomwt.conllu").write_text(SYNTHETIC_CONLLU, encoding="utf-8")
    out_dir = tmp_path / "out"

    rc = run_paper_extractor_module.main(
        ["--parsed-dir", str(parsed_dir), "--output-dir", str(out_dir)]
    )
    assert rc == 0

    contracted_tsv = out_dir / "hp1_de_ch01_contracted.tsv"
    assert contracted_tsv.exists()
    tsv_content = contracted_tsv.read_text(encoding="utf-8")
    # The header now carries the new coordinate fields
    assert "prep_token_id" in tsv_content
    assert "det_token_id" in tsv_content
    assert "noun_token_id" in tsv_content
    assert "pp_token_start" in tsv_content
    assert "pp_token_end" in tsv_content
    assert "pp_surface" in tsv_content
    assert "Haus" in tsv_content  # the noun lemma IS in the TSV


def test_tsv_carries_occurrence_coordinates(tmp_path, run_paper_extractor_module):
    """Coordinate fields populate correctly for contracted and uncontracted PPs."""
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    (parsed_dir / "hp1_de_ch01_nomwt.conllu").write_text(SYNTHETIC_CONLLU, encoding="utf-8")
    out_dir = tmp_path / "out"

    rc = run_paper_extractor_module.main(
        ["--parsed-dir", str(parsed_dir), "--output-dir", str(out_dir)]
    )
    assert rc == 0

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


def test_two_occurrences_same_prep_noun_are_distinguishable(tmp_path, run_paper_extractor_module):
    """Two occurrences of (im, Haus) within one sentence must remain separate
    rows with different token coordinates — the extractor must NOT deduplicate."""
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    (parsed_dir / "hp1_de_ch01_nomwt.conllu").write_text(
        SYNTHETIC_TWO_SAME_CONLLU, encoding="utf-8"
    )
    out_dir = tmp_path / "out"

    rc = run_paper_extractor_module.main(
        ["--parsed-dir", str(parsed_dir), "--output-dir", str(out_dir)]
    )
    assert rc == 0

    rows = list(
        csv.DictReader(
            (out_dir / "hp1_de_ch01_contracted.tsv").open(encoding="utf-8"),
            delimiter="\t",
        )
    )
    assert len(rows) == 2, f"expected 2 occurrences, got {len(rows)}"
    # Same (prep, noun, sentence_id) but different coordinates.
    assert rows[0]["sentence_id"] == rows[1]["sentence_id"]
    assert rows[0]["prep"] == rows[1]["prep"] == "im"
    assert rows[0]["noun"] == rows[1]["noun"] == "Haus"
    coords = {(r["prep_token_id"], r["noun_token_id"]) for r in rows}
    assert coords == {("1", "2"), ("4", "5")}, coords
    # Both surfaces reconstruct correctly from their own ranges.
    surfaces = {r["pp_surface"] for r in rows}
    assert surfaces == {"im Haus"}


def test_pp_surface_reconstructs_only_within_token_range(tmp_path, run_paper_extractor_module):
    """pp_surface must be reconstructable from the inclusive token range, even
    when the range contains tokens between prep and noun (e.g. an adjective
    line that the extractor doesn't track)."""
    conllu = (
        "# sent_id = synthetic_with_adj\n"
        "# text = im großen Haus\n"
        "1\tim\tim\tADP\tAPPR\t_\t3\tcase\t_\t_\n"
        "2\tgroßen\tgroß\tADJ\tADJA\t_\t3\tamod\t_\t_\n"
        "3\tHaus\tHaus\tNOUN\tNN\t_\t0\troot\t_\t_\n"
    )
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    (parsed_dir / "hp1_de_ch01_nomwt.conllu").write_text(conllu, encoding="utf-8")
    out_dir = tmp_path / "out"

    rc = run_paper_extractor_module.main(
        ["--parsed-dir", str(parsed_dir), "--output-dir", str(out_dir)]
    )
    assert rc == 0

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
