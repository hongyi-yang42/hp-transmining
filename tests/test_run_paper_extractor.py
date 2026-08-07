"""Regression guard: run_paper_extractor.py stdout must contain only
aggregate counts, never noun lemmas (or any other token-level data)
from the source text.

The script normally depends on the gitignored vendor/conll-extractor.
We inject a synthetic conll_extractor package via sys.modules so this
test runs in CI without the vendor.
"""

from __future__ import annotations

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


@pytest.fixture
def run_paper_extractor_module():
    saved = {
        k: sys.modules[k] for k in list(sys.modules) if k.startswith("conll_extractor")
    }
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
    (parsed_dir / "hp1_de_ch01_nomwt.conllu").write_text(
        SYNTHETIC_CONLLU, encoding="utf-8"
    )
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
    (parsed_dir / "hp1_de_ch01_nomwt.conllu").write_text(
        SYNTHETIC_CONLLU, encoding="utf-8"
    )
    out_dir = tmp_path / "out"

    rc = run_paper_extractor_module.main(
        ["--parsed-dir", str(parsed_dir), "--output-dir", str(out_dir)]
    )
    assert rc == 0

    contracted_tsv = out_dir / "hp1_de_ch01_contracted.tsv"
    assert contracted_tsv.exists()
    tsv_content = contracted_tsv.read_text(encoding="utf-8")
    # Header + 1 data row
    assert "sentence_id\tprep\tdet\tnoun\tin_filter" in tsv_content
    assert "Haus" in tsv_content  # the noun lemma IS in the TSV
