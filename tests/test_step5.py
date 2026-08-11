"""Synthetic-fixture tests for Step 5 category derivation and analysis.

Tests exercise:

  * src/hp_corpus/step5.py — pure functions
  * scripts/derive_paper_categories.py — CLI
  * scripts/analyze_step5.py — CLI

All fixtures use invented text and datapoint IDs. No novel content.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from hp_corpus.step5 import (
    DE_PAPER_CATEGORIES,
    DERIVED_COLUMNS,
    EN_COARSE_MAP,
    EN_PAPER_CATEGORIES,
    ZH_COARSE_MAP,
    ZH_PAPER_CATEGORIES,
    analyze,
    cross_tab,
    de_paper_category,
    derive_row,
    derive_rows,
    distribution,
    en_paper_category,
    is_analysis_ready,
    source_labels_rolled_up,
    uncontracted_mandarin_bare_ids,
    zh_paper_category,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------- helpers


def _row(
    datapoint_id: str,
    *,
    chapter: int = 1,
    de_form: str = "contracted",
    en_form: str = "definite",
    zh_form: str = "bare",
    decision: str = "include",
    annotation_status: str = "complete",
    adjudication_status: str = "",
    en_aligned_text: str = "synth EN one",
    zh_aligned_text: str = "synth ZH one",
) -> dict[str, str]:
    return {
        "datapoint_id": datapoint_id,
        "chapter": str(chapter),
        "de_form": de_form,
        "en_aligned_text": en_aligned_text,
        "zh_aligned_text": zh_aligned_text,
        "de_candidate_decision": decision,
        "annotation_status": annotation_status,
        "adjudication_status": adjudication_status,
        "en_form": en_form,
        "zh_form": zh_form,
    }


# --------------------------------------------------------------------- predicates


def test_is_analysis_ready_happy_path() -> None:
    assert is_analysis_ready(_row("dp1")) is True


def test_is_analysis_ready_excludes_non_include_decision() -> None:
    assert is_analysis_ready(_row("dp1", decision="exclude")) is False
    assert is_analysis_ready(_row("dp1", decision="uncertain")) is False
    assert is_analysis_ready(_row("dp1", decision="")) is False


def test_is_analysis_ready_excludes_incomplete_annotation() -> None:
    assert is_analysis_ready(_row("dp1", annotation_status="in_progress")) is False
    assert is_analysis_ready(_row("dp1", annotation_status="")) is False


def test_is_analysis_ready_rejects_pending_adjudication() -> None:
    assert is_analysis_ready(_row("dp1", adjudication_status="pending")) is False
    assert is_analysis_ready(_row("dp1", adjudication_status="disputed")) is False


def test_is_analysis_ready_accepts_blank_or_adjudicated() -> None:
    assert is_analysis_ready(_row("dp1", adjudication_status="")) is True
    assert is_analysis_ready(_row("dp1", adjudication_status="adjudicated")) is True


def test_is_analysis_ready_requires_both_target_texts() -> None:
    assert is_analysis_ready(_row("dp1", en_aligned_text="")) is False
    assert is_analysis_ready(_row("dp1", zh_aligned_text="")) is False


# --------------------------------------------------------------------- derivation


def test_de_paper_category_mirrors_de_form() -> None:
    assert de_paper_category(_row("dp1", de_form="contracted")) == "contracted"
    assert de_paper_category(_row("dp1", de_form="uncontracted")) == "uncontracted"


def test_en_paper_category_coarse_mapping() -> None:
    assert en_paper_category(_row("dp1", en_form="definite")) == "definite"
    assert en_paper_category(_row("dp1", en_form="bare_singular")) == "bare_singular"
    assert en_paper_category(_row("dp1", en_form="demonstrative")) == "demonstrative"


def test_en_paper_category_rolls_up_to_other() -> None:
    for fine in (
        "indefinite",
        "possessive",
        "pronoun",
        "proper_name",
        "other",
        "omitted",
        "uncertain",
    ):
        assert en_paper_category(_row("dp1", en_form=fine)) == "other"


def test_zh_paper_category_coarse_mapping() -> None:
    assert zh_paper_category(_row("dp1", zh_form="bare")) == "bare"
    assert zh_paper_category(_row("dp1", zh_form="demonstrative")) == "demonstrative"


def test_zh_paper_category_rolls_up_to_other() -> None:
    for fine in (
        "numeral_classifier",
        "possessive",
        "pronoun",
        "proper_name",
        "other",
        "omitted",
        "uncertain",
    ):
        assert zh_paper_category(_row("dp1", zh_form=fine)) == "other"


def test_derive_row_preserves_fine_labels() -> None:
    """The fine annotator label must survive in the derived row so
    nothing disappears silently from the denominator."""
    row = _row("dp1", en_form="possessive", zh_form="numeral_classifier")
    out = derive_row(row)
    assert out["en_form_fine"] == "possessive"
    assert out["zh_form_fine"] == "numeral_classifier"
    assert out["en_paper_category"] == "other"
    assert out["zh_paper_category"] == "other"


def test_derive_row_columns_match_derived_columns() -> None:
    row = _row("dp1")
    out = derive_row(row)
    assert set(out.keys()) == set(DERIVED_COLUMNS)


def test_derive_rows_filters_to_analysis_ready() -> None:
    rows = [
        _row("dp1"),  # eligible
        _row("dp2", decision="exclude"),  # filtered
        _row("dp3", annotation_status="in_progress"),  # filtered
        _row("dp4"),  # eligible
    ]
    out = derive_rows(rows)
    assert {r["datapoint_id"] for r in out} == {"dp1", "dp4"}


# --------------------------------------------------------------------- distribution


def test_distribution_counts_and_percentages() -> None:
    derived = [
        {"k": "a"},
        {"k": "a"},
        {"k": "b"},
    ]
    d = distribution(derived, "k")
    assert d["a"] == {"count": 2, "percent": 66.67}
    assert d["b"] == {"count": 1, "percent": 33.33}


def test_distribution_empty_denominator() -> None:
    d = distribution([], "k")
    assert d == {}


def test_cross_tab_row_percentages_sum_to_100() -> None:
    derived = [
        {"r": "x", "c": "p"},
        {"r": "x", "c": "p"},
        {"r": "x", "c": "q"},
        {"r": "y", "c": "p"},
    ]
    t = cross_tab(derived, "r", "c")
    x_total = sum(v["percent"] for k, v in t["x"].items())
    assert abs(x_total - 100.0) < 0.01
    # 2 of 3 → 66.67%; 1 of 3 → 33.33%
    assert t["x"]["p"]["count"] == 2
    assert t["x"]["p"]["percent"] == 66.67
    assert t["x"]["q"]["count"] == 1
    assert t["y"]["p"]["count"] == 1
    assert t["y"]["p"]["percent"] == 100.0


def test_cross_tab_denominators_block() -> None:
    derived = [
        {"r": "x", "c": "p"},
        {"r": "x", "c": "q"},
        {"r": "y", "c": "p"},
    ]
    t = cross_tab(derived, "r", "c")
    assert t["_denominators"]["x"]["row_total"] == 2
    assert t["_denominators"]["y"]["row_total"] == 1


# --------------------------------------------------------------------- roll-up


def test_source_labels_rolled_up_enumerates_other_fine_labels() -> None:
    derived = [
        {
            "en_form_fine": "indefinite",
            "en_paper_category": "other",
            "zh_form_fine": "bare",
            "zh_paper_category": "bare",
        },
        {
            "en_form_fine": "indefinite",
            "en_paper_category": "other",
            "zh_form_fine": "bare",
            "zh_paper_category": "bare",
        },
        {
            "en_form_fine": "pronoun",
            "en_paper_category": "other",
            "zh_form_fine": "numeral_classifier",
            "zh_paper_category": "other",
        },
        {
            "en_form_fine": "definite",
            "en_paper_category": "definite",
            "zh_form_fine": "demonstrative",
            "zh_paper_category": "demonstrative",
        },
    ]
    out = source_labels_rolled_up(derived)
    assert out["en_other"]["indefinite"] == 2
    assert out["en_other"]["pronoun"] == 1
    assert "definite" not in out["en_other"]
    assert out["zh_other"]["numeral_classifier"] == 1
    assert "bare" not in out["zh_other"]


def test_uncontracted_mandarin_bare_ids() -> None:
    derived = [
        {
            "datapoint_id": "dp1",
            "de_paper_category": "uncontracted",
            "zh_paper_category": "bare",
        },
        {
            "datapoint_id": "dp2",
            "de_paper_category": "uncontracted",
            "zh_paper_category": "demonstrative",
        },
        {
            "datapoint_id": "dp3",
            "de_paper_category": "contracted",
            "zh_paper_category": "bare",
        },
    ]
    ids = uncontracted_mandarin_bare_ids(derived)
    assert ids == ["dp1"]


# --------------------------------------------------------------------- analyze


def test_analyze_returns_full_summary() -> None:
    derived = [
        derive_row(
            _row("dp1", de_form="contracted", en_form="definite", zh_form="bare")
        ),
        derive_row(
            _row(
                "dp2",
                de_form="uncontracted",
                en_form="bare_singular",
                zh_form="bare",
            )
        ),
        derive_row(
            _row(
                "dp3",
                de_form="contracted",
                en_form="possessive",
                zh_form="numeral_classifier",
            )
        ),
    ]
    s = analyze(derived)
    assert s["analysis_ready_total"] == 3
    assert "de_distribution" in s
    assert "en_distribution" in s
    assert "zh_distribution" in s
    assert "de_x_zh" in s
    assert "de_x_en" in s
    assert "source_labels_rolled_up" in s
    # possessive rolled up to en_other
    assert s["source_labels_rolled_up"]["en_other"]["possessive"] == 1
    # numeral_classifier rolled up to zh_other
    assert s["source_labels_rolled_up"]["zh_other"]["numeral_classifier"] == 1
    # dp1 + dp2 are uncontracted-mandarin-bare? dp1: contracted+bare → no.
    # dp2: uncontracted+bare → yes. dp3: zh_form=other → no.
    assert s["uncontracted_mandarin_bare_count"] == 1


# --------------------------------------------------------------------- CLIs


def _write_gold_tsv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def test_derive_cli_writes_detailed_tsv(tmp_path: Path) -> None:
    cli = _load_script("derive_paper_categories.py")
    gold = _write_gold_tsv(
        tmp_path / "gold.tsv",
        [
            _row("dp1", en_form="definite", zh_form="bare"),
            _row("dp2", decision="exclude"),  # filtered
            _row("dp3", en_form="bare_singular", zh_form="demonstrative"),
        ],
    )
    out = tmp_path / "derived.tsv"
    rc = cli.main(["--gold-tsv", str(gold), "--output", str(out)])
    assert rc == 0
    with open(out, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert {r["datapoint_id"] for r in rows} == {"dp1", "dp3"}
    # Columns match DERIVED_COLUMNS exactly.
    assert set(rows[0].keys()) == set(DERIVED_COLUMNS)


def test_derive_cli_refuses_overwrite(tmp_path: Path) -> None:
    cli = _load_script("derive_paper_categories.py")
    gold = _write_gold_tsv(tmp_path / "gold.tsv", [_row("dp1")])
    out = tmp_path / "derived.tsv"
    out.write_text("placeholder", encoding="utf-8")
    rc = cli.main(["--gold-tsv", str(gold), "--output", str(out)])
    assert rc == 2


def test_analyze_cli_writes_all_outputs(tmp_path: Path) -> None:
    derive_cli = _load_script("derive_paper_categories.py")
    analyze_cli = _load_script("analyze_step5.py")
    gold = _write_gold_tsv(
        tmp_path / "gold.tsv",
        [
            _row("dp1", de_form="contracted", en_form="definite", zh_form="bare"),
            _row("dp2", de_form="uncontracted", en_form="bare_singular", zh_form="bare"),
            _row("dp3", de_form="contracted", en_form="possessive", zh_form="demonstrative"),
        ],
    )
    derived = tmp_path / "derived.tsv"
    derive_cli.main(["--gold-tsv", str(gold), "--output", str(derived)])
    out_dir = tmp_path / "step5_out"
    rc = analyze_cli.main(["--derived-tsv", str(derived), "--out-dir", str(out_dir)])
    assert rc == 0
    for fname in (
        "summary.json",
        "de_distribution.json",
        "en_distribution.json",
        "zh_distribution.json",
        "de_x_zh_table.json",
        "de_x_en_table.json",
        "uncontracted_mandarin_bare_review.tsv",
    ):
        assert (out_dir / fname).exists(), fname

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["analysis_ready_total"] == 3
    assert summary["uncontracted_mandarin_bare_count"] == 1

    # Review TSV has the right ID.
    with open(out_dir / "uncontracted_mandarin_bare_review.tsv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert {r["datapoint_id"] for r in rows} == {"dp2"}


def test_analyze_cli_refuses_overwrite(tmp_path: Path) -> None:
    analyze_cli = _load_script("analyze_step5.py")
    derived = tmp_path / "derived.tsv"
    derived.write_text("datapoint_id\tchapter\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "summary.json").write_text("{}", encoding="utf-8")
    rc = analyze_cli.main(["--derived-tsv", str(derived), "--out-dir", str(out_dir)])
    assert rc == 2


def test_derive_cli_stdout_privacy(tmp_path: Path, capsys) -> None:
    cli = _load_script("derive_paper_categories.py")
    gold = _write_gold_tsv(tmp_path / "gold.tsv", [_row("dp1")])
    cli.main(["--gold-tsv", str(gold), "--output", str(tmp_path / "derived.tsv")])
    out = capsys.readouterr().out
    assert "dp1" not in out
    assert "synth EN one" not in out


def test_constants_are_complete() -> None:
    assert DE_PAPER_CATEGORIES == {"contracted", "uncontracted"}
    assert EN_PAPER_CATEGORIES == {"definite", "bare_singular", "demonstrative", "other"}
    assert ZH_PAPER_CATEGORIES == {"bare", "demonstrative", "other"}
    # The coarse map must NOT include "other" — that's the fallback.
    assert "other" not in EN_COARSE_MAP
    assert "other" not in ZH_COARSE_MAP
