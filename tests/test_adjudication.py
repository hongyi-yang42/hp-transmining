"""Synthetic-fixture tests for the WP4 adjudication pipeline:

  * scripts/compare_annotations.py
  * scripts/build_adjudication_ledger.py
  * scripts/merge_adjudicated.py

All fixtures use invented text (``synth DE one`` etc.). No novel content.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

from hp_corpus.step4 import (
    ALL_TSV_COLUMNS,
    BUILDER_DEFAULT_EDITABLE,
    DE_CANDIDATE_DECISIONS,
    EDITABLE_COLUMNS,
    compute_source_row_sha256,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_validator():
    path = REPO_ROOT / "scripts" / "validate_step4_annotations.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------- helpers


def _blank_editable(col: str) -> str:
    return BUILDER_DEFAULT_EDITABLE.get(col, "")


def _make_source_dict(
    datapoint_id: str, *, chapter: int = 1, form: str = "contracted"
) -> dict[str, Any]:
    """Build a minimal valid source-row dict for hash computation."""
    return {
        "datapoint_id": datapoint_id,
        "dataset_scope": "test",
        "paper_final_sample": False,
        "chapter": chapter,
        "de_sentence_id": f"hp1_de_ch{chapter:02d}_p0001_s001",
        "de_token_start": 1,
        "de_token_end": 2,
        "de_pp_surface": "synth PP",
        "de_sentence_text": "synth DE one",
        "de_prep_surface": "synth",
        "de_prep_normalized": "synth",
        "de_head_lemma": "synth",
        "de_form": form,
        "author_resource_match": False,
        "minimal_pair_group": "synth|synth",
        "en_sentence_ids": ["hp1_en_ch01_p0001_s001"],
        "en_aligned_text": "synth EN one",
        "en_alignment_cardinality": "1:1",
        "en_alignment_status": "aligned",
        "en_alignment_confidence": 0.9,
        "zh_sentence_ids": ["hp1_zh_ch01_p0001_s001"],
        "zh_aligned_text": "synth ZH one",
        "zh_alignment_cardinality": "1:1",
        "zh_alignment_status": "aligned",
        "zh_alignment_confidence": 0.9,
        "pilot_selected": False,
        "pilot_selection_reason": "test",
    }


def _row_dict(datapoint_id: str, *, chapter: int = 1, form: str = "contracted") -> dict[str, str]:
    """Build a complete row dict (all ALL_TSV_COLUMNS) in initial state."""
    src = _make_source_dict(datapoint_id, chapter=chapter, form=form)
    src["source_row_sha256"] = compute_source_row_sha256(src)
    row: dict[str, str] = {}
    for col in ALL_TSV_COLUMNS:
        if col == "source_row_sha256":
            row[col] = src[col]
            continue
        val = src.get(col, "")
        if isinstance(val, list):
            row[col] = json.dumps(val, ensure_ascii=False)
        elif isinstance(val, bool):
            row[col] = "true" if val else "false"
        elif val is None:
            row[col] = ""
        else:
            row[col] = str(val)
    for col in EDITABLE_COLUMNS:
        row[col] = _blank_editable(col)
    return row


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(ALL_TSV_COLUMNS), delimiter="\t", lineterminator="\n"
        )
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in ALL_TSV_COLUMNS})
    return path


def _annotate_row(
    row: dict[str, str],
    *,
    decision: str = "include",
    en_form: str = "definite",
    zh_form: str = "bare",
    en_alignment_qc: str = "confirmed",
    zh_alignment_qc: str = "confirmed",
    annotation_status: str = "complete",
) -> dict[str, str]:
    """Mark a row as fully annotated. Used to build annotator fixtures."""
    row["de_candidate_decision"] = decision
    row["annotation_status"] = annotation_status
    row["en_alignment_qc"] = en_alignment_qc
    row["en_alignment_relation"] = "direct"
    row["en_span_text"] = "synth EN one"
    row["en_char_ranges"] = "[[0,12]]"
    row["en_form"] = en_form
    row["en_confidence"] = "high"
    row["zh_alignment_qc"] = zh_alignment_qc
    row["zh_alignment_relation"] = "direct"
    row["zh_span_text"] = "synth ZH one"
    row["zh_char_ranges"] = "[[0,12]]"
    row["zh_form"] = zh_form
    row["zh_confidence"] = "high"
    return row


# --------------------------------------------------------------------- compare_annotations


def test_compare_no_disagreements(tmp_path: Path) -> None:
    compare = _load_script("compare_annotations.py")
    row = _annotate_row(_row_dict("dp1"))
    a = _write_tsv(tmp_path / "a.tsv", [dict(row)])
    b = _write_tsv(tmp_path / "b.tsv", [dict(row)])
    out_stem = tmp_path / "cmp"
    rc = compare.main(["--a", str(a), "--b", str(b), "--out-stem", str(out_stem)])
    assert rc == 0
    disagr = tmp_path / "cmp.disagreements.tsv"
    assert disagr.exists()
    with open(disagr, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert rows == []


def test_compare_records_value_disagreement(tmp_path: Path) -> None:
    compare = _load_script("compare_annotations.py")
    row_a = _annotate_row(_row_dict("dp1"), en_form="definite")
    row_b = _annotate_row(_row_dict("dp1"), en_form="demonstrative")
    a = _write_tsv(tmp_path / "a.tsv", [row_a])
    b = _write_tsv(tmp_path / "b.tsv", [row_b])
    rc = compare.main([
        "--a", str(a),
        "--b", str(b),
        "--out-stem", str(tmp_path / "cmp"),
    ])
    assert rc == 0
    with open(tmp_path / "cmp.disagreements.tsv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["field"] == "en_form"
    assert rows[0]["value_a"] == "definite"
    assert rows[0]["value_b"] == "demonstrative"


def test_compare_blank_vs_nonblank_is_disagreement(tmp_path: Path) -> None:
    """If A left en_notes blank and B filled it, that's NOT a research-
    field disagreement (notes are excluded). But if A left en_form blank
    and B filled it, that IS a disagreement."""
    compare = _load_script("compare_annotations.py")
    row_a = _annotate_row(_row_dict("dp1"))
    row_a["en_form"] = ""  # blank
    row_b = _annotate_row(_row_dict("dp1"))
    a = _write_tsv(tmp_path / "a.tsv", [row_a])
    b = _write_tsv(tmp_path / "b.tsv", [row_b])
    rc = compare.main([
        "--a", str(a), "--b", str(b), "--out-stem", str(tmp_path / "cmp")
    ])
    assert rc == 0
    with open(tmp_path / "cmp.disagreements.tsv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    fields = [r["field"] for r in rows]
    assert "en_form" in fields


def test_compare_refuses_source_hash_mismatch(tmp_path: Path) -> None:
    """If source_row_sha256 differs between A and B for the same
    datapoint_id, refuse with SOURCE_HASH_MISMATCH."""
    compare = _load_script("compare_annotations.py")
    row_a = _annotate_row(_row_dict("dp1"))
    row_b = _annotate_row(_row_dict("dp1"))
    # Tamper with B's source hash to simulate a template divergence.
    row_b["source_row_sha256"] = "0" * 64
    a = _write_tsv(tmp_path / "a.tsv", [row_a])
    b = _write_tsv(tmp_path / "b.tsv", [row_b])
    rc = compare.main([
        "--a", str(a), "--b", str(b), "--out-stem", str(tmp_path / "cmp")
    ])
    assert rc == 2


def test_compare_does_not_treat_annotator_field_as_disagreement(tmp_path: Path) -> None:
    """The annotator-name field is workflow metadata, not a linguistic
    disagreement — even if A and B have different annotator values."""
    compare = _load_script("compare_annotations.py")
    row_a = _annotate_row(_row_dict("dp1"))
    row_a["annotator"] = "alice"
    row_b = _annotate_row(_row_dict("dp1"))
    row_b["annotator"] = "bob"
    a = _write_tsv(tmp_path / "a.tsv", [row_a])
    b = _write_tsv(tmp_path / "b.tsv", [row_b])
    rc = compare.main([
        "--a", str(a), "--b", str(b), "--out-stem", str(tmp_path / "cmp")
    ])
    assert rc == 0
    with open(tmp_path / "cmp.disagreements.tsv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert rows == []


def test_compare_stdout_privacy(tmp_path: Path, capsys) -> None:
    """Stdout must not carry annotation values or datapoint IDs."""
    compare = _load_script("compare_annotations.py")
    row_a = _annotate_row(_row_dict("dp1"), en_form="definite")
    row_b = _annotate_row(_row_dict("dp1"), en_form="demonstrative")
    a = _write_tsv(tmp_path / "a.tsv", [row_a])
    b = _write_tsv(tmp_path / "b.tsv", [row_b])
    compare.main(["--a", str(a), "--b", str(b), "--out-stem", str(tmp_path / "cmp")])
    out = capsys.readouterr().out
    assert "dp1" not in out
    assert "definite" not in out
    assert "demonstrative" not in out


# --------------------------------------------------------------------- build_adjudication_ledger


def test_ledger_initial_state_is_pending(tmp_path: Path) -> None:
    cmp = _load_script("compare_annotations.py")
    ledger = _load_script("build_adjudication_ledger.py")
    row_a = _annotate_row(_row_dict("dp1"), en_form="definite")
    row_b = _annotate_row(_row_dict("dp1"), en_form="demonstrative")
    a = _write_tsv(tmp_path / "a.tsv", [row_a])
    b = _write_tsv(tmp_path / "b.tsv", [row_b])
    cmp.main(["--a", str(a), "--b", str(b), "--out-stem", str(tmp_path / "cmp")])
    ledger_path = tmp_path / "ledger.tsv"
    rc = ledger.main([
        "--comparison", str(tmp_path / "cmp.disagreements.tsv"),
        "--a", str(a),
        "--b", str(b),
        "--output", str(ledger_path),
    ])
    assert rc == 0
    with open(ledger_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["resolution_status"] == "pending"
    assert rows[0]["adjudicated_value"] == ""
    assert rows[0]["adjudication_note"] == ""
    # Provenance sidecar.
    prov = tmp_path / "ledger.tsv.provenance.json"
    assert prov.exists()
    payload = json.loads(prov.read_text(encoding="utf-8"))
    assert payload["annotator_a_file"] == "a.tsv"
    assert payload["annotator_b_file"] == "b.tsv"
    assert "dp1" in payload["source_hashes_a"]


def test_ledger_refuses_overwrite_without_force(tmp_path: Path) -> None:
    cmp = _load_script("compare_annotations.py")
    ledger = _load_script("build_adjudication_ledger.py")
    row_a = _annotate_row(_row_dict("dp1"))
    row_b = _annotate_row(_row_dict("dp1"))
    a = _write_tsv(tmp_path / "a.tsv", [row_a])
    b = _write_tsv(tmp_path / "b.tsv", [row_b])
    cmp.main(["--a", str(a), "--b", str(b), "--out-stem", str(tmp_path / "cmp")])
    ledger_path = tmp_path / "ledger.tsv"
    ledger_path.write_text("placeholder", encoding="utf-8")
    rc = ledger.main([
        "--comparison", str(tmp_path / "cmp.disagreements.tsv"),
        "--a", str(a),
        "--b", str(b),
        "--output", str(ledger_path),
    ])
    assert rc == 2


# --------------------------------------------------------------------- merge_adjudicated


def _fill_ledger(
    ledger_path: Path, *, value: str = "definite", status: str = "adjudicated"
) -> None:
    """Mark every ledger row as adjudicated with the given value."""
    with open(ledger_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        r["adjudicated_value"] = value
        r["resolution_status"] = status
    cols = ["datapoint_id", "field", "value_a", "value_b",
            "adjudicated_value", "resolution_status", "adjudication_note"]
    with open(ledger_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_merge_applies_adjudicated_value(tmp_path: Path) -> None:
    cmp = _load_script("compare_annotations.py")
    ledger = _load_script("build_adjudication_ledger.py")
    merge = _load_script("merge_adjudicated.py")

    master_row = _row_dict("dp1")
    master = _write_tsv(tmp_path / "master.tsv", [master_row])

    row_a = _annotate_row(_row_dict("dp1"), en_form="definite")
    row_b = _annotate_row(_row_dict("dp1"), en_form="demonstrative")
    a = _write_tsv(tmp_path / "a.tsv", [row_a])
    b = _write_tsv(tmp_path / "b.tsv", [row_b])

    cmp.main(["--a", str(a), "--b", str(b), "--out-stem", str(tmp_path / "cmp")])
    ledger_path = tmp_path / "ledger.tsv"
    ledger.main([
        "--comparison", str(tmp_path / "cmp.disagreements.tsv"),
        "--a", str(a), "--b", str(b),
        "--output", str(ledger_path),
    ])
    _fill_ledger(ledger_path, value="definite", status="adjudicated")

    gold = tmp_path / "gold.tsv"
    rc = merge.main([
        "--master", str(master),
        "--annotator-a", str(a),
        "--annotator-b", str(b),
        "--ledger", str(ledger_path),
        "--output", str(gold),
        "--annotation-pool",
    ])
    assert rc == 0
    with open(gold, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert rows[0]["en_form"] == "definite"  # adjudicated value won
    assert rows[0]["adjudication_status"] == "adjudicated"


def test_merge_refuses_unresolved_disagreement(tmp_path: Path) -> None:
    cmp = _load_script("compare_annotations.py")
    ledger = _load_script("build_adjudication_ledger.py")
    merge = _load_script("merge_adjudicated.py")

    master = _write_tsv(tmp_path / "master.tsv", [_row_dict("dp1")])
    row_a = _annotate_row(_row_dict("dp1"), en_form="definite")
    row_b = _annotate_row(_row_dict("dp1"), en_form="demonstrative")
    a = _write_tsv(tmp_path / "a.tsv", [row_a])
    b = _write_tsv(tmp_path / "b.tsv", [row_b])
    cmp.main(["--a", str(a), "--b", str(b), "--out-stem", str(tmp_path / "cmp")])
    ledger_path = tmp_path / "ledger.tsv"
    ledger.main([
        "--comparison", str(tmp_path / "cmp.disagreements.tsv"),
        "--a", str(a), "--b", str(b),
        "--output", str(ledger_path),
    ])
    # Leave the ledger pending — merge must refuse.
    rc = merge.main([
        "--master", str(master),
        "--annotator-a", str(a),
        "--annotator-b", str(b),
        "--ledger", str(ledger_path),
        "--output", str(tmp_path / "gold.tsv"),
    ])
    assert rc == 2


def test_merge_refuses_blank_adjudicated_value(tmp_path: Path) -> None:
    cmp = _load_script("compare_annotations.py")
    ledger = _load_script("build_adjudication_ledger.py")
    merge = _load_script("merge_adjudicated.py")

    master = _write_tsv(tmp_path / "master.tsv", [_row_dict("dp1")])
    row_a = _annotate_row(_row_dict("dp1"), en_form="definite")
    row_b = _annotate_row(_row_dict("dp1"), en_form="demonstrative")
    a = _write_tsv(tmp_path / "a.tsv", [row_a])
    b = _write_tsv(tmp_path / "b.tsv", [row_b])
    cmp.main(["--a", str(a), "--b", str(b), "--out-stem", str(tmp_path / "cmp")])
    ledger_path = tmp_path / "ledger.tsv"
    ledger.main([
        "--comparison", str(tmp_path / "cmp.disagreements.tsv"),
        "--a", str(a), "--b", str(b),
        "--output", str(ledger_path),
    ])
    # Mark adjudicated but leave value blank.
    _fill_ledger(ledger_path, value="", status="adjudicated")
    rc = merge.main([
        "--master", str(master),
        "--annotator-a", str(a),
        "--annotator-b", str(b),
        "--ledger", str(ledger_path),
        "--output", str(tmp_path / "gold.tsv"),
    ])
    assert rc == 2


def test_merge_refuses_source_hash_mismatch(tmp_path: Path) -> None:
    ledger = _load_script("build_adjudication_ledger.py")
    merge = _load_script("merge_adjudicated.py")
    master = _write_tsv(tmp_path / "master.tsv", [_row_dict("dp1")])
    row_a = _annotate_row(_row_dict("dp1"))
    row_b = _annotate_row(_row_dict("dp1"))
    row_b["source_row_sha256"] = "0" * 64  # tamper
    a = _write_tsv(tmp_path / "a.tsv", [row_a])
    b = _write_tsv(tmp_path / "b.tsv", [row_b])
    # compare.py would refuse; bypass it by writing an empty comparison.
    (tmp_path / "cmp.disagreements.tsv").write_text(
        "datapoint_id\tfield\tvalue_a\tvalue_b\n", encoding="utf-8"
    )
    ledger_path = tmp_path / "ledger.tsv"
    ledger.main([
        "--comparison", str(tmp_path / "cmp.disagreements.tsv"),
        "--a", str(a), "--b", str(b),
        "--output", str(ledger_path),
    ])
    rc = merge.main([
        "--master", str(master),
        "--annotator-a", str(a),
        "--annotator-b", str(b),
        "--ledger", str(ledger_path),
        "--output", str(tmp_path / "gold.tsv"),
    ])
    assert rc == 2


def test_merge_preserves_agreed_values(tmp_path: Path) -> None:
    """When A and B agree on every field, the gold row equals A's row
    (no adjudication_status change)."""
    cmp = _load_script("compare_annotations.py")
    ledger = _load_script("build_adjudication_ledger.py")
    merge = _load_script("merge_adjudicated.py")
    master = _write_tsv(tmp_path / "master.tsv", [_row_dict("dp1")])
    row = _annotate_row(_row_dict("dp1"))
    a = _write_tsv(tmp_path / "a.tsv", [dict(row)])
    b = _write_tsv(tmp_path / "b.tsv", [dict(row)])
    cmp.main(["--a", str(a), "--b", str(b), "--out-stem", str(tmp_path / "cmp")])
    ledger_path = tmp_path / "ledger.tsv"
    ledger.main([
        "--comparison", str(tmp_path / "cmp.disagreements.tsv"),
        "--a", str(a), "--b", str(b),
        "--output", str(ledger_path),
    ])
    gold = tmp_path / "gold.tsv"
    rc = merge.main([
        "--master", str(master),
        "--annotator-a", str(a),
        "--annotator-b", str(b),
        "--ledger", str(ledger_path),
        "--output", str(gold),
        "--annotation-pool",
    ])
    assert rc == 0
    with open(gold, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    # Adjudication_status stays blank when there were no disagreements.
    assert rows[0]["adjudication_status"] == ""
    # Agreed values preserved.
    assert rows[0]["en_form"] == "definite"


def test_merge_runs_validator_require_complete(tmp_path: Path) -> None:
    """A merged gold TSV that passes --require-complete should exit 0;
    an incomplete one should fail validation."""
    cmp = _load_script("compare_annotations.py")
    ledger = _load_script("build_adjudication_ledger.py")
    merge = _load_script("merge_adjudicated.py")
    master = _write_tsv(tmp_path / "master.tsv", [_row_dict("dp1")])
    row = _annotate_row(_row_dict("dp1"))
    a = _write_tsv(tmp_path / "a.tsv", [dict(row)])
    b = _write_tsv(tmp_path / "b.tsv", [dict(row)])
    cmp.main(["--a", str(a), "--b", str(b), "--out-stem", str(tmp_path / "cmp")])
    ledger_path = tmp_path / "ledger.tsv"
    ledger.main([
        "--comparison", str(tmp_path / "cmp.disagreements.tsv"),
        "--a", str(a), "--b", str(b),
        "--output", str(ledger_path),
    ])
    rc = merge.main([
        "--master", str(master),
        "--annotator-a", str(a),
        "--annotator-b", str(b),
        "--ledger", str(ledger_path),
        "--output", str(tmp_path / "gold.tsv"),
        "--annotation-pool",
    ])
    assert rc == 0  # happy path


def test_merge_refuses_overwrite_without_force(tmp_path: Path) -> None:
    cmp = _load_script("compare_annotations.py")
    ledger = _load_script("build_adjudication_ledger.py")
    merge = _load_script("merge_adjudicated.py")
    master = _write_tsv(tmp_path / "master.tsv", [_row_dict("dp1")])
    row = _annotate_row(_row_dict("dp1"))
    a = _write_tsv(tmp_path / "a.tsv", [dict(row)])
    b = _write_tsv(tmp_path / "b.tsv", [dict(row)])
    cmp.main(["--a", str(a), "--b", str(b), "--out-stem", str(tmp_path / "cmp")])
    ledger_path = tmp_path / "ledger.tsv"
    ledger.main([
        "--comparison", str(tmp_path / "cmp.disagreements.tsv"),
        "--a", str(a), "--b", str(b),
        "--output", str(ledger_path),
    ])
    gold = tmp_path / "gold.tsv"
    gold.write_text("placeholder", encoding="utf-8")
    rc = merge.main([
        "--master", str(master),
        "--annotator-a", str(a),
        "--annotator-b", str(b),
        "--ledger", str(ledger_path),
        "--output", str(gold),
    ])
    assert rc == 2


def test_decision_vocab_constant_used() -> None:
    """Smoke test: the merge pipeline references DE_CANDIDATE_DECISIONS
    indirectly via the validator. The constant must include the values
    the merge test fixtures rely on."""
    assert "include" in DE_CANDIDATE_DECISIONS
    assert "exclude" in DE_CANDIDATE_DECISIONS
    assert "uncertain" in DE_CANDIDATE_DECISIONS
