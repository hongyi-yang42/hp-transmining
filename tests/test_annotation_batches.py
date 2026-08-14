"""Synthetic-fixture tests for scripts/prepare_annotation_batches.py.

Every fixture uses invented, non-novel text — no Harry Potter content,
no filter-list lemmas. Tests cover the invariants enumerated in the WP3
spec: determinism, calibration sharing, coverage, no within-file
duplicates, routing correctness, seed sensitivity, overlap > 0, the
three refusal rules (empty out-dir, calibration too large, <2
annotators), stratification order, and the stdout privacy guard.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from hp_corpus.step4 import ALL_TSV_COLUMNS, BUILDER_DEFAULT_EDITABLE

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "prepare_annotation_batches.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "prepare_annotation_batches", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------- fixtures

# Sentinel strings used in synthetic fixtures. Chosen to be obviously
# non-novel and to never appear in the production corpus.
_SYNTH_DE_TEXT = "synthDE Alpha beta gamma"
_SYNTH_EN_TEXT = "synthEN alpha beta gamma"
_SYNTH_ZH_TEXT = "合成甲乙丙"
_SYNTH_LEMMA = "SynthLemma"


def _make_row(
    *,
    datapoint_id: str,
    chapter: int,
    de_form: str,
    decision: str = "",
    excluded_reason: str = "",
    de_candidate_notes: str = "",
) -> dict[str, str]:
    """Build a valid master-TSV row with synthetic values for every column."""
    row: dict[str, str] = {col: "" for col in ALL_TSV_COLUMNS}
    row["datapoint_id"] = datapoint_id
    row["dataset_scope"] = "synth_test_scope"
    row["paper_final_sample"] = "false"
    row["chapter"] = str(chapter)
    row["de_sentence_id"] = f"hp1_de_ch{chapter:02d}_p0001_s001"
    row["de_token_start"] = "1"
    row["de_token_end"] = "2"
    row["de_pp_surface"] = "synth pp"
    row["de_sentence_text"] = _SYNTH_DE_TEXT
    row["de_prep_surface"] = "im"
    row["de_prep_normalized"] = "in"
    row["de_head_lemma"] = _SYNTH_LEMMA
    row["de_form"] = de_form
    row["author_resource_match"] = "false"
    row["minimal_pair_group"] = f"in|{_SYNTH_LEMMA}"
    row["en_sentence_ids"] = "[]"
    row["en_aligned_text"] = _SYNTH_EN_TEXT
    row["en_alignment_cardinality"] = "1:1"
    row["en_alignment_status"] = "aligned"
    row["en_alignment_confidence"] = "0.9"
    row["zh_sentence_ids"] = "[]"
    row["zh_aligned_text"] = _SYNTH_ZH_TEXT
    row["zh_alignment_cardinality"] = "1:1"
    row["zh_alignment_status"] = "aligned"
    row["zh_alignment_confidence"] = "0.9"
    row["pilot_selected"] = "false"
    row["pilot_selection_reason"] = ""
    row["source_row_sha256"] = hashlib.sha256(
        b"synth"  # Stand-in hash; the master is "validated" upstream.
    ).hexdigest()
    # Editable defaults.
    row["de_candidate_decision"] = decision
    row["de_exclusion_reason"] = excluded_reason
    row["de_candidate_notes"] = de_candidate_notes
    # Apply BUILDER_DEFAULT_EDITABLE so the row matches what a freshly
    # built master would look like.
    for col, val in BUILDER_DEFAULT_EDITABLE.items():
        row[col] = val
    row["annotator"] = ""
    row["annotation_status"] = ""
    row["adjudication_status"] = ""
    row["general_notes"] = ""
    return row


def _write_master(path: Path, rows: list[dict[str, str]]) -> None:
    """Write a master TSV with the canonical header."""
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


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def _build_master(
    path: Path,
    *,
    n_include: int = 20,
    n_exclude: int = 3,
    n_uncertain: int = 2,
    n_blank: int = 1,
    chapters: tuple[int, ...] = (1, 2),
    forms: tuple[str, str] = ("contracted", "uncontracted"),
) -> list[dict[str, str]]:
    """Build a master TSV with the requested mix of routing buckets.

    Returns the row list (caller may inspect IDs directly).
    """
    rows: list[dict[str, str]] = []
    counter = 0

    def _add(bucket: str, *, chapter: int, form: str, decision: str) -> None:
        nonlocal counter
        counter += 1
        dp_id = f"dp_{bucket}_{counter:03d}_ch{chapter:02d}_{form}"
        if decision == "exclude":
            rows.append(
                _make_row(
                    datapoint_id=dp_id,
                    chapter=chapter,
                    de_form=form,
                    decision="exclude",
                    excluded_reason="duplicate",
                )
            )
        elif decision == "uncertain":
            rows.append(
                _make_row(
                    datapoint_id=dp_id,
                    chapter=chapter,
                    de_form=form,
                    decision="uncertain",
                    de_candidate_notes="synth uncertain note",
                )
            )
        else:
            rows.append(
                _make_row(
                    datapoint_id=dp_id,
                    chapter=chapter,
                    de_form=form,
                    decision=decision,
                )
            )

    # Distribute include rows across (chapter, form) cells.
    for i in range(n_include):
        ch = chapters[i % len(chapters)]
        form = forms[i % len(forms)]
        _add("include", chapter=ch, form=form, decision="include")

    for i in range(n_exclude):
        ch = chapters[i % len(chapters)]
        form = forms[i % len(forms)]
        _add("exclude", chapter=ch, form=form, decision="exclude")

    for i in range(n_uncertain):
        ch = chapters[i % len(chapters)]
        form = forms[i % len(forms)]
        _add("uncertain", chapter=ch, form=form, decision="uncertain")

    for i in range(n_blank):
        ch = chapters[i % len(chapters)]
        form = forms[i % len(forms)]
        _add("blank", chapter=ch, form=form, decision="")

    _write_master(path, rows)
    return rows


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dir_sha256_snapshot(directory: Path) -> dict[str, str]:
    """Hash every file in ``directory`` for byte-identical comparisons."""
    out: dict[str, str] = {}
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(directory))] = _file_sha256(p)
    return out


# --------------------------------------------------------------------- tests


def test_determinism(tmp_path: Path) -> None:
    """Same inputs + same seed → byte-identical output."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    _build_master(master, n_include=20)
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    rc1 = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out1),
            "--annotators", "alice", "bob",
            "--calibration-size", "4",
        ]
    )
    rc2 = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out2),
            "--annotators", "alice", "bob",
            "--calibration-size", "4",
        ]
    )
    assert rc1 == 0 and rc2 == 0
    snap1 = _dir_sha256_snapshot(out1)
    snap2 = _dir_sha256_snapshot(out2)
    assert snap1 == snap2, "two runs with same inputs must be byte-identical"


def test_calibration_in_every_annotator_file(tmp_path: Path) -> None:
    """With calibration_size=3 and 2 annotators, every per-annotator TSV
    contains the same 3 calibration IDs."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    _build_master(master, n_include=12, n_exclude=0, n_uncertain=0, n_blank=0)
    out = tmp_path / "out"
    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out),
            "--annotators", "alice", "bob",
            "--calibration-size", "3",
            "--overlap", "0.0",
        ]
    )
    assert rc == 0

    manifest = json.loads((out / "batch_manifest.json").read_text())
    cal_ids = manifest["calibration_ids"]
    assert len(cal_ids) == 3

    for annotator in ("alice", "bob"):
        annotator_rows = _read_tsv(out / f"{annotator}.tsv")
        ids = [r["datapoint_id"] for r in annotator_rows]
        for cid in cal_ids:
            assert cid in ids, f"calibration {cid} missing from {annotator}"
        # Manifest and file agree on calibration IDs.
        assert sorted(manifest["per_annotator"][annotator]["calibration_ids"]) == sorted(cal_ids)


def test_coverage(tmp_path: Path) -> None:
    """Every eligible ID appears in ≥1 annotator file."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    rows = _build_master(master, n_include=20, n_exclude=0, n_uncertain=0, n_blank=0)
    include_ids = {
        r["datapoint_id"] for r in rows if r["de_candidate_decision"] == "include"
    }
    out = tmp_path / "out"
    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out),
            "--annotators", "alice", "bob",
            "--calibration-size", "4",
        ]
    )
    assert rc == 0

    seen: set[str] = set()
    for annotator in ("alice", "bob"):
        for r in _read_tsv(out / f"{annotator}.tsv"):
            seen.add(r["datapoint_id"])
    assert include_ids.issubset(seen), (
        f"{len(include_ids - seen)} eligible IDs missing from all annotator files"
    )


def test_no_within_file_duplicates(tmp_path: Path) -> None:
    """Each annotator file has unique IDs (calibration + main + overlap)."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    _build_master(master, n_include=30, n_exclude=0, n_uncertain=0, n_blank=0)
    out = tmp_path / "out"
    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out),
            "--annotators", "alice", "bob", "carol",
            "--calibration-size", "5",
            "--overlap", "0.3",
        ]
    )
    assert rc == 0
    for annotator in ("alice", "bob", "carol"):
        ids = [r["datapoint_id"] for r in _read_tsv(out / f"{annotator}.tsv")]
        assert len(ids) == len(set(ids)), f"{annotator} has duplicate IDs"


def test_routing(tmp_path: Path) -> None:
    """include → batches, exclude → excluded.tsv, blank/uncertain → blocked_review.tsv."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    rows = _build_master(
        master, n_include=6, n_exclude=3, n_uncertain=2, n_blank=1
    )
    out = tmp_path / "out"
    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out),
            "--annotators", "alice", "bob",
            "--calibration-size", "2",
        ]
    )
    assert rc == 0

    excluded_ids = {r["datapoint_id"] for r in _read_tsv(out / "excluded.tsv")}
    blocked_ids = {r["datapoint_id"] for r in _read_tsv(out / "blocked_review.tsv")}
    annotator_ids: set[str] = set()
    for annotator in ("alice", "bob"):
        annotator_ids.update(
            r["datapoint_id"] for r in _read_tsv(out / f"{annotator}.tsv")
        )

    expected_excluded = {
        r["datapoint_id"] for r in rows if r["de_candidate_decision"] == "exclude"
    }
    expected_blocked = {
        r["datapoint_id"]
        for r in rows
        if r["de_candidate_decision"] in ("", "uncertain")
    }
    expected_include = {
        r["datapoint_id"] for r in rows if r["de_candidate_decision"] == "include"
    }

    assert excluded_ids == expected_excluded
    assert blocked_ids == expected_blocked
    assert expected_include.issubset(annotator_ids)


def test_excluded_blocked_never_in_batches(tmp_path: Path) -> None:
    """Excluded and blocked IDs never appear in any annotator batch."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    rows = _build_master(
        master, n_include=10, n_exclude=4, n_uncertain=2, n_blank=2
    )
    out = tmp_path / "out"
    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out),
            "--annotators", "alice", "bob",
            "--calibration-size", "2",
        ]
    )
    assert rc == 0

    bad_ids = {
        r["datapoint_id"]
        for r in rows
        if r["de_candidate_decision"] in ("", "exclude", "uncertain")
    }
    for annotator in ("alice", "bob"):
        ids = {r["datapoint_id"] for r in _read_tsv(out / f"{annotator}.tsv")}
        assert not (bad_ids & ids), (
            f"{annotator} contains excluded/blocked IDs: {bad_ids & ids}"
        )


def test_seed_sensitivity(tmp_path: Path) -> None:
    """Changing the seed produces a different (but still valid) assignment."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    _build_master(master, n_include=30, n_exclude=0, n_uncertain=0, n_blank=0)
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"

    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out_a),
            "--annotators", "alice", "bob",
            "--calibration-size", "3",
            "--seed", "SEED-A",
        ]
    )
    assert rc == 0
    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out_b),
            "--annotators", "alice", "bob",
            "--calibration-size", "3",
            "--seed", "SEED-B",
        ]
    )
    assert rc == 0

    snap_a = _dir_sha256_snapshot(out_a)
    snap_b = _dir_sha256_snapshot(out_b)
    # Outputs differ — at minimum the manifest's seed field changes.
    assert snap_a != snap_b

    # Both runs are still valid: full coverage, no within-file duplicates.
    for out in (out_a, out_b):
        all_ids: set[str] = set()
        for annotator in ("alice", "bob"):
            ids = [r["datapoint_id"] for r in _read_tsv(out / f"{annotator}.tsv")]
            assert len(ids) == len(set(ids))
            all_ids.update(ids)
        # 30 eligible rows must all appear somewhere.
        assert len(all_ids) == 30


def test_overlap_default_creates_overlap(tmp_path: Path) -> None:
    """Default overlap_rate=0.2 actually creates some overlap with enough rows."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    _build_master(master, n_include=60, n_exclude=0, n_uncertain=0, n_blank=0)
    out = tmp_path / "out"
    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out),
            "--annotators", "alice", "bob",
            "--calibration-size", "5",
            "--overlap", "0.2",
        ]
    )
    assert rc == 0

    manifest = json.loads((out / "batch_manifest.json").read_text())
    # At least one annotator must have non-zero overlap when there are
    # enough remaining rows.
    total_overlap = sum(
        len(v["overlap_ids"]) for v in manifest["per_annotator"].values()
    )
    assert total_overlap > 0, "expected non-zero overlap with default 0.2 and 55 remaining rows"


def test_refuse_nonempty_outdir_without_force(tmp_path: Path) -> None:
    """Refuse empty out-dir without --force-output."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    _build_master(master, n_include=10)
    out = tmp_path / "out"
    out.mkdir()
    (out / "preexisting.txt").write_text("blocker")

    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out),
            "--annotators", "alice", "bob",
        ]
    )
    assert rc == 2, "non-empty out-dir without --force-output must exit 2"


def test_force_output_overwrites(tmp_path: Path) -> None:
    """--force-output lets the run proceed even when out-dir is non-empty."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    _build_master(master, n_include=10)
    out = tmp_path / "out"
    out.mkdir()
    (out / "preexisting.txt").write_text("blocker")

    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out),
            "--annotators", "alice", "bob",
            "--force-output",
        ]
    )
    assert rc == 0


def test_refuse_calibration_too_large(tmp_path: Path) -> None:
    """Refuse calibration_size > eligible_count."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    _build_master(master, n_include=5)
    out = tmp_path / "out"
    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out),
            "--annotators", "alice", "bob",
            "--calibration-size", "6",  # > 5 eligible
        ]
    )
    assert rc == 2


def test_refuse_too_few_annotators(tmp_path: Path) -> None:
    """Refuse <2 annotators."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    _build_master(master, n_include=10)
    out = tmp_path / "out"
    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out),
            "--annotators", "solo",
        ]
    )
    assert rc == 2


def test_stratification_sort_order(tmp_path: Path) -> None:
    """The eligible sort order is (stable_key, chapter, de_form, datapoint_id).
    Verify by comparing the manifest's first annotator's main_batch_ids
    against the expected sort computed independently."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    rows = _build_master(
        master,
        # Use a large enough n_include to populate both chapters × both forms
        # plus enough rows in the first annotator's main batch to make the
        # sort visible.
        n_include=24,
        n_exclude=0,
        n_uncertain=0,
        n_blank=0,
        chapters=(1, 2),
        forms=("contracted", "uncontracted"),
    )
    seed = "strat-test-seed"
    out = tmp_path / "out"
    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out),
            "--annotators", "alice", "bob",
            "--calibration-size", "0",  # disable calibration so remaining = all eligible
            "--overlap", "0.0",
            "--seed", seed,
        ]
    )
    assert rc == 0

    manifest = json.loads((out / "batch_manifest.json").read_text())

    # Independently compute the eligible sort + partition. Primary
    # assignment is stable_key % n_annotators — NOT position-in-sorted-list.
    include_rows = [r for r in rows if r["de_candidate_decision"] == "include"]

    def _key(r: dict[str, str]) -> tuple[Any, ...]:
        dp_id = r["datapoint_id"]
        h = hashlib.sha256(f"{dp_id}|{seed}".encode()).hexdigest()
        stable_key = int(h[:16], 16)
        ch = int(r["chapter"])
        return (stable_key, ch, r["de_form"], dp_id)

    def _stable_key(r: dict[str, str]) -> int:
        h = hashlib.sha256(
            f"{r['datapoint_id']}|{seed}".encode()
        ).hexdigest()
        return int(h[:16], 16)

    sorted_rows = sorted(include_rows, key=_key)
    # 0 calibration rows → remaining = sorted_rows. alice = index 0.
    alice_expected = [
        r["datapoint_id"]
        for r in sorted_rows
        if _stable_key(r) % 2 == 0  # alice = index 0; n_annotators = 2
    ]
    assert manifest["per_annotator"]["alice"]["main_batch_ids"] == alice_expected
    # Within alice's batch, the order must follow the global sort —
    # this is what "stratification" buys us (alice sees a chapter/form
    # balanced slice, in stable order).
    bob_expected = [
        r["datapoint_id"]
        for r in sorted_rows
        if _stable_key(r) % 2 == 1
    ]
    assert manifest["per_annotator"]["bob"]["main_batch_ids"] == bob_expected


def test_stdout_privacy(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Stdout must not leak any synthetic-fixture strings."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    _build_master(master, n_include=10)
    out = tmp_path / "out"
    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out),
            "--annotators", "alice", "bob",
            "--calibration-size", "2",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out

    # Sentinels that must NEVER appear in stdout — text, lemmas,
    # datapoint IDs, and column names. Decision *labels* (include /
    # exclude / uncertain) are NOT considered leaks when they appear as
    # substrings of legitimate count labels like "excluded: 3" — what
    # we care about is that no row's actual data leaks.
    forbidden_substrings = [
        _SYNTH_DE_TEXT,
        _SYNTH_EN_TEXT,
        _SYNTH_ZH_TEXT,
        _SYNTH_LEMMA,
        "datapoint_id",   # column names never printed
        "de_form",
        "de_sentence_text",
        "en_aligned_text",
        "zh_aligned_text",
        "source_row_sha256",
    ]
    # Plus a few actual datapoint IDs from the fixture.
    rows = _read_tsv(master)
    for r in rows[:5]:
        forbidden_substrings.append(r["datapoint_id"])

    for needle in forbidden_substrings:
        assert needle not in captured, (
            f"stdout leaked fixture string: {needle!r}\ncaptured:\n{captured}"
        )

    # Decision values must not appear as bare/standalone tokens. The
    # legitimate count labels are "excluded: <N>" and "blocked: <N>"
    # — neither produces a bare "include" / "uncertain" token.
    for token in ("include", "uncertain"):
        # Match the token surrounded by non-word boundaries to avoid
        # matching substrings of other words.
        pattern = rf"\b{re.escape(token)}\b"
        assert re.search(pattern, captured) is None, (
            f"stdout leaked decision token: {token!r}\ncaptured:\n{captured}"
        )


def test_drop_excluded(tmp_path: Path) -> None:
    """--drop-excluded omits excluded.tsv entirely."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    _build_master(master, n_include=6, n_exclude=3, n_uncertain=1, n_blank=1)
    out = tmp_path / "out"
    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out),
            "--annotators", "alice", "bob",
            "--calibration-size", "1",
            "--drop-excluded",
        ]
    )
    assert rc == 0
    assert not (out / "excluded.tsv").exists(), "excluded.tsv should not exist"
    # blocked_review.tsv is always written.
    assert (out / "blocked_review.tsv").exists()


def test_calibration_zero_disables(tmp_path: Path) -> None:
    """--calibration-size 0 disables calibration: every annotator's
    calibration_ids is empty and no calibration.tsv rows are written."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    _build_master(master, n_include=10)
    out = tmp_path / "out"
    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out),
            "--annotators", "alice", "bob",
            "--calibration-size", "0",
            "--overlap", "0.0",
        ]
    )
    assert rc == 0
    manifest = json.loads((out / "batch_manifest.json").read_text())
    assert manifest["calibration_ids"] == []
    for annotator in ("alice", "bob"):
        assert manifest["per_annotator"][annotator]["calibration_ids"] == []
    cal_rows = _read_tsv(out / "calibration.tsv")
    assert cal_rows == []


# ------------------------------------------------------------ calibration quota


def test_calibration_quota_exact_composition(tmp_path: Path) -> None:
    """--calibration-quota contracted=3,uncontracted=3 yields exactly
    3 + 3, recorded in the manifest, shared by every annotator."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    rows = _build_master(master, n_include=16, n_exclude=0, n_uncertain=0, n_blank=0)
    out = tmp_path / "out"
    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out),
            "--annotators", "alice", "bob",
            "--calibration-quota", "contracted=3,uncontracted=3",
            "--overlap", "0.0",
        ]
    )
    assert rc == 0

    manifest = json.loads((out / "batch_manifest.json").read_text())
    assert manifest["calibration_mode"] == "form_quota"
    assert manifest["calibration_quota"] == {"contracted": 3, "uncontracted": 3}
    assert manifest["calibration_composition"] == {
        "contracted": 3,
        "uncontracted": 3,
    }
    assert len(manifest["calibration_ids"]) == 6

    cal_rows = _read_tsv(out / "calibration.tsv")
    assert len(cal_rows) == 6
    by_id = {r["datapoint_id"]: r for r in rows}
    comp = Counter(by_id[r["datapoint_id"]]["de_form"] for r in cal_rows)
    assert comp == Counter({"contracted": 3, "uncontracted": 3})

    # Shared calibration in every annotator file.
    for annotator in ("alice", "bob"):
        ids = [r["datapoint_id"] for r in _read_tsv(out / f"{annotator}.tsv")]
        for cid in manifest["calibration_ids"]:
            assert cid in ids

    # Coverage: all 16 eligible rows still appear somewhere.
    seen: set[str] = set()
    for annotator in ("alice", "bob"):
        seen.update(
            r["datapoint_id"] for r in _read_tsv(out / f"{annotator}.tsv")
        )
    eligible_ids = {
        r["datapoint_id"] for r in rows if r["de_candidate_decision"] == "include"
    }
    assert eligible_ids.issubset(seen)


def test_calibration_quota_determinism(tmp_path: Path) -> None:
    """Quota mode is byte-identical across runs with the same inputs."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    _build_master(master, n_include=24)
    out1, out2 = tmp_path / "run1", tmp_path / "run2"
    argv = [
        "--master-tsv", str(master),
        "--annotators", "alice", "bob",
        "--calibration-quota", "contracted=5,uncontracted=5",
    ]
    assert mod.main(argv + ["--out-dir", str(out1)]) == 0
    assert mod.main(argv + ["--out-dir", str(out2)]) == 0
    assert _dir_sha256_snapshot(out1) == _dir_sha256_snapshot(out2)


def test_calibration_quota_stable_order_subsequence(tmp_path: Path) -> None:
    """Quota-selected calibration IDs are a subsequence of the global
    stable-sorted eligible order (selection preserves order, only caps
    per stratum)."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    rows = _build_master(master, n_include=24, n_exclude=0, n_uncertain=0, n_blank=0)
    seed = "quota-order-seed"
    out = tmp_path / "out"
    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out),
            "--annotators", "alice", "bob",
            "--calibration-quota", "contracted=4,uncontracted=4",
            "--seed", seed,
        ]
    )
    assert rc == 0
    manifest = json.loads((out / "batch_manifest.json").read_text())

    def _key(r: dict[str, str]) -> tuple[Any, ...]:
        h = hashlib.sha256(
            f"{r['datapoint_id']}|{seed}".encode()
        ).hexdigest()
        return (int(h[:16], 16), int(r["chapter"]), r["de_form"], r["datapoint_id"])

    include_rows = [r for r in rows if r["de_candidate_decision"] == "include"]
    global_order = [r["datapoint_id"] for r in sorted(include_rows, key=_key)]
    pos = {dp: i for i, dp in enumerate(global_order)}
    cal = manifest["calibration_ids"]
    assert [pos[dp] for dp in cal] == sorted(pos[dp] for dp in cal)


def test_calibration_quota_refuses_conflict_with_size(tmp_path: Path) -> None:
    """Both --calibration-size and --calibration-quota → exit 2."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    _build_master(master, n_include=20)
    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(tmp_path / "out"),
            "--annotators", "alice", "bob",
            "--calibration-size", "4",
            "--calibration-quota", "contracted=2,uncontracted=2",
        ]
    )
    assert rc == 2


def test_calibration_quota_refuses_incomplete(tmp_path: Path) -> None:
    """A quota that does not list every eligible form → exit 2."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    _build_master(master, n_include=20)  # both forms present
    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(tmp_path / "out"),
            "--annotators", "alice", "bob",
            "--calibration-quota", "contracted=10",  # uncontracted missing
        ]
    )
    assert rc == 2


def test_calibration_quota_refuses_unsatisfiable(tmp_path: Path) -> None:
    """Quota exceeding a stratum's eligible count → exit 2."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    # forms alternate → 5 contracted / 5 uncontracted among 10 include rows.
    _build_master(master, n_include=10)
    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(tmp_path / "out"),
            "--annotators", "alice", "bob",
            "--calibration-quota", "contracted=6,uncontracted=4",
        ]
    )
    assert rc == 2


def test_calibration_quota_bad_spec(tmp_path: Path) -> None:
    """Malformed quota specs → exit 2."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    _build_master(master, n_include=10)
    for bad in ["contracted=x,uncontracted=2", "contracted", "a=1,a=2", "a=-1,b=2"]:
        rc = mod.main(
            [
                "--master-tsv", str(master),
                "--out-dir", str(tmp_path / "out"),
                "--annotators", "alice", "bob",
                "--calibration-quota", bad,
            ]
        )
        assert rc == 2, f"spec {bad!r} must be refused"


def test_calibration_quota_zero_excludes_stratum(tmp_path: Path) -> None:
    """A 0 quota explicitly excludes a form from calibration; those rows
    stay in the main batches (coverage unaffected)."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    rows = _build_master(master, n_include=12, n_exclude=0, n_uncertain=0, n_blank=0)
    out = tmp_path / "out"
    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out),
            "--annotators", "alice", "bob",
            "--calibration-quota", "contracted=4,uncontracted=0",
            "--overlap", "0.0",
        ]
    )
    assert rc == 0
    manifest = json.loads((out / "batch_manifest.json").read_text())
    assert manifest["calibration_composition"] == {"contracted": 4}

    seen: set[str] = set()
    for annotator in ("alice", "bob"):
        seen.update(
            r["datapoint_id"] for r in _read_tsv(out / f"{annotator}.tsv")
        )
    eligible_ids = {
        r["datapoint_id"] for r in rows if r["de_candidate_decision"] == "include"
    }
    assert eligible_ids.issubset(seen)


def test_calibration_quota_stdout_privacy(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Quota-mode stdout carries aggregate counts only — no fixture text,
    datapoint IDs, or column names."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    _build_master(master, n_include=10)
    out = tmp_path / "out"
    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out),
            "--annotators", "alice", "bob",
            "--calibration-quota", "contracted=2,uncontracted=2",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out

    forbidden = [
        _SYNTH_DE_TEXT, _SYNTH_EN_TEXT, _SYNTH_ZH_TEXT, _SYNTH_LEMMA,
        "datapoint_id", "de_form", "source_row_sha256",
    ]
    for r in _read_tsv(master)[:5]:
        forbidden.append(r["datapoint_id"])
    for needle in forbidden:
        assert needle not in captured, f"stdout leaked {needle!r}"

    # Composition is printed as aggregate counts.
    assert "calibration_composition:" in captured
    assert "calibration_mode: form_quota" in captured


def test_calibration_composition_recorded_in_size_mode(tmp_path: Path) -> None:
    """Size mode also records the actual composition — the coordinator
    can verify the draw without touching IDs."""
    mod = _load_script()
    master = tmp_path / "master.tsv"
    _build_master(master, n_include=10)
    out = tmp_path / "out"
    rc = mod.main(
        [
            "--master-tsv", str(master),
            "--out-dir", str(out),
            "--annotators", "alice", "bob",
            "--calibration-size", "4",
        ]
    )
    assert rc == 0
    manifest = json.loads((out / "batch_manifest.json").read_text())
    assert manifest["calibration_mode"] == "size"
    assert sum(manifest["calibration_composition"].values()) == 4
    assert "calibration_quota" not in manifest
