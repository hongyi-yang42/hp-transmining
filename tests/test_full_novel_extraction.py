"""Synthetic-fixture tests for the full-novel German extraction pipeline.

These tests cover Work Package 5:

* Path generation for chapters 9, 10, 17 (the original ``ch0{ch}`` bug
  surfaced at ch=10 → ``hp1_de_ch010_nomwt.conllu``).
* Fail-closed rules: missing input (default), missing with ``--allow-missing``,
  empty input (always fails), extraction error.
* Legitimate zero-hit chapter returns ``status="zero_hits_ok"`` and an
  empty TSV (header only) — NOT an error.
* Happy path with hits + ``in_filter`` flag from FILTER_CONTRACTED_123 /
  FILTER_PP without those flags gating the row count.
* Determinism: same input → same SHA-256 + row count.
* Chapter range validation (1..17): ch=0 and ch=18 fail with exit 2.
* Manifest structure: every required field is present.
* Stdout privacy guard: no synthetic noun/prep strings leak to stdout.

The vendor ``conll-extractor`` clone is NOT required by this suite. Each
test monkeypatches ``hp_corpus.german_extraction``'s filter-list handles
with small synthetic lists before invoking the CLI.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_full_novel_german_extraction.py"

# ---------------------------------------------------------------------------
# Synthetic CoNLL-U fixtures.
# ---------------------------------------------------------------------------
# Token columns: id, form, lemma, upos, xpos, head, deprel, deps, misc.
# The 10-field form is what Stanza emits; pyconll parses it cleanly.
# Every block carries migrated provenance: sent_id is a <segment_id>#bNNN
# parse_block_id and the original segment id lives in the
# # source_segment_id comment.

# Contracted PP: "im Haus" — "im" ∈ CONTRACTED, "Haus" ∈ FILTER_CONTRACTED_123["im"].
SYNTH_CONTRACTED_HIT = """\
# sent_id = synth_contracted#b001
# source_segment_id = synth_contracted
# text = im Haus
1\tim\tim\tADP\tAPPR\t_\t2\tcase\t_\t_
2\tHaus\tHaus\tNOUN\tNN\t_\t0\troot\t_\t_
"""

# Uncontracted PP: "in dem Wald" — "in" ∈ PREPOSITIONS, "dem" ∈ DETERMINERS,
# "Wald" ∈ FILTER_PP["in"].
SYNTH_UNCONTRACTED_HIT = """\
# sent_id = synth_uncontracted#b001
# source_segment_id = synth_uncontracted
# text = in dem Wald
1\tin\tin\tADP\tAPPR\t_\t3\tcase\t_\t_
2\tdem\tder\tDET\tART\t_\t3\tdet\t_\t_
3\tWald\tWald\tNOUN\tNN\t_\t0\troot\t_\t_
"""

# Sentence with no CONTRACTED/PREPOSITIONS forms → zero hits.
SYNTH_NO_HITS = """\
# sent_id = synth_no_hits#b001
# source_segment_id = synth_no_hits
# text = rien
1\trien\trien\tNOUN\tNN\t_\t0\troot\t_\t_
"""

# A real "two-form" file: one contracted + one uncontracted sentence.
SYNTH_BOTH = (
    "# sent_id = synth_c#b001\n"
    "# source_segment_id = synth_c\n"
    "# text = im Haus\n"
    "1\tim\tim\tADP\tAPPR\t_\t2\tcase\t_\t_\n"
    "2\tHaus\tHaus\tNOUN\tNN\t_\t0\troot\t_\t_\n"
    "\n"
    "# sent_id = synth_u#b001\n"
    "# source_segment_id = synth_u\n"
    "# text = in dem Wald\n"
    "1\tin\tin\tADP\tAPPR\t_\t3\tcase\t_\t_\n"
    "2\tdem\tder\tDET\tART\t_\t3\tdet\t_\t_\n"
    "3\tWald\tWald\tNOUN\tNN\t_\t0\troot\t_\t_\n"
)

# Sentences with prep forms that aren't in any FILTER — exercises the
# "in_filter exists but doesn't gate row count" rule.
SYNTH_OUT_OF_FILTER = """\
# sent_id = synth_outside#b001
# source_segment_id = synth_outside
# text = im Berg
1\tim\tim\tADP\tAPPR\t_\t2\tcase\t_\t_
2\tBerg\tBerg\tNOUN\tNN\t_\t0\troot\t_\t_
"""


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _load_cli_module():
    """Import the CLI script as a module (with src/ on sys.path).

    The script injects ``src/`` and ``vendor/conll-extractor/`` into
    ``sys.path`` itself, so we don't need to do that here. We do strip
    the vendor path if the clone isn't present locally so the test
    doesn't accidentally import a stale vendor.
    """
    spec = importlib.util.spec_from_file_location(
        "run_full_novel_german_extraction", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cli(monkeypatch):
    """Load the CLI module with synthetic vendor lists injected.

    We replace the module-global filter handles with small synthetic
    lists AFTER the module is imported. ``extract_chapter`` reads these
    globals at call time, so monkeypatching before invoking ``main``
    is sufficient. This isolates the suite from the vendor clone.
    """
    from hp_corpus import german_extraction as ge

    monkeypatch.setattr(ge, "CONTRACTED", ["im"])
    monkeypatch.setattr(ge, "PREPOSITIONS", ["in"])
    monkeypatch.setattr(ge, "DETERMINERS", ["dem"])
    monkeypatch.setattr(
        ge, "FILTER_CONTRACTED_123", {"im": ["Haus"]}
    )
    monkeypatch.setattr(ge, "FILTER_PP", {"in": ["Wald"]})

    return _load_cli_module()


def _write_parsed(parsed_dir: Path, chapter: int, content: str) -> Path:
    """Write a synthetic .conllu file with the correct ch{NN:02d} name."""
    parsed_dir.mkdir(parents=True, exist_ok=True)
    p = parsed_dir / f"hp1_de_ch{chapter:02d}_nomwt.conllu"
    p.write_text(content, encoding="utf-8")
    return p


def _read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ---------------------------------------------------------------------------
# 1. Chapter path generation (the ch0{ch} bug).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chapter,expected", [(9, 9), (10, 10), (17, 17)])
def test_chapter_path_generation_two_digit(
    tmp_path, monkeypatch, chapter, expected
):
    """The CLI must request ``hp1_de_ch{NN:02d}_nomwt.conllu`` for ch ≥ 9.

    Regression for the ``ch0{ch}`` bug that produced ``ch010`` for ch=10.
    For ch=9 the bug form ``ch09`` is identical to the fixed form (since
    ``f"ch0{9}" == f"ch{9:02d}"``); the meaningful regression is ch ≥ 10.
    """
    from hp_corpus import german_extraction as ge

    monkeypatch.setattr(ge, "CONTRACTED", ["im"])
    monkeypatch.setattr(ge, "PREPOSITIONS", ["in"])
    monkeypatch.setattr(ge, "DETERMINERS", ["dem"])
    monkeypatch.setattr(ge, "FILTER_CONTRACTED_123", {"im": ["Haus"]})
    monkeypatch.setattr(ge, "FILTER_PP", {"in": ["Wald"]})

    cli = _load_cli_module()
    parsed_dir = tmp_path / "parsed"
    # Don't create the file — we want to capture the path the CLI requests.
    rc = cli.main(
        [
            "--chapters",
            str(chapter),
            "--parsed-dir",
            str(parsed_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--allow-missing",
        ]
    )
    assert rc == 0
    manifest_path = tmp_path / "out" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requested_paths = {entry["parsed_path"] for entry in manifest}
    expected_name = f"hp1_de_ch{expected:02d}_nomwt.conllu"
    assert any(p.endswith(expected_name) for p in requested_paths), (
        f"expected CLI to request {expected_name}, got {requested_paths}"
    )
    # Negative regression: for two-digit chapters, the buggy ch0{NN} form
    # would produce ch0{NN} (3 digits), which is wrong. Only meaningful
    # for chapter >= 10 — for ch=9 both forms are identical.
    if chapter >= 10:
        bad_name = f"hp1_de_ch0{chapter}_nomwt.conllu"
        assert not any(p.endswith(bad_name) for p in requested_paths), (
            f"CLI requested buggy path {bad_name}"
        )


def test_chapter_10_original_bug_case(tmp_path, cli):
    """Ch.10 specifically — the bug case from run_paper_extractor.py:202.

    With ``ch0{ch}`` formatting, ch=10 produced ``hp1_de_ch010_nomwt.conllu``;
    with the fix it must produce ``hp1_de_ch10_nomwt.conllu``.
    """
    parsed_dir = tmp_path / "parsed"
    _write_parsed(parsed_dir, 10, SYNTH_BOTH)
    out_dir = tmp_path / "out"

    rc = cli.main(
        [
            "--chapters",
            "10",
            "--parsed-dir",
            str(parsed_dir),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    # Output TSVs must be at the ch{10:02d} = ch10 path.
    assert (out_dir / "hp1_de_ch10_contracted.tsv").exists()
    assert (out_dir / "hp1_de_ch10_uncontracted.tsv").exists()
    # The wrong ch010 path must NOT be produced.
    assert not (out_dir / "hp1_de_ch010_contracted.tsv").exists()
    assert not (out_dir / "hp1_de_ch010_uncontracted.tsv").exists()


# ---------------------------------------------------------------------------
# 3, 4. Missing input handling.
# ---------------------------------------------------------------------------


def test_missing_input_default_fails_closed(tmp_path, cli, capsys):
    """Default behavior: missing input → non-zero exit, rule MISSING_PARSED_INPUT."""
    parsed_dir = tmp_path / "parsed"  # empty
    out_dir = tmp_path / "out"

    rc = cli.main(
        [
            "--chapters",
            "5",
            "--parsed-dir",
            str(parsed_dir),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc != 0
    err = capsys.readouterr().err
    assert "MISSING_PARSED_INPUT" in err


def test_missing_input_allow_missing_continues(tmp_path, cli):
    """With --allow-missing, missing input → manifest entry, exit 0."""
    parsed_dir = tmp_path / "parsed"  # empty
    out_dir = tmp_path / "out"

    rc = cli.main(
        [
            "--chapters",
            "5",
            "--parsed-dir",
            str(parsed_dir),
            "--output-dir",
            str(out_dir),
            "--allow-missing",
        ]
    )
    assert rc == 0
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 2  # contracted + uncontracted entries
    assert all(m["status"] == "missing_input" for m in manifest)
    assert all(m["chapter"] == 5 for m in manifest)


# ---------------------------------------------------------------------------
# 5. Empty input always fails closed.
# ---------------------------------------------------------------------------


def test_empty_input_fails_even_with_allow_missing(tmp_path, cli, capsys):
    """File exists but contains zero sentences → EMPTY_PARSED_INPUT.

    Even with --allow-missing, an empty file is upstream corruption, not
    a missing file. The script must surface it.
    """
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir(parents=True)
    # Truly empty file (no comment lines, no blank lines) — pyconll yields
    # zero sentences from this, which our extractor treats as corruption.
    (parsed_dir / "hp1_de_ch07_nomwt.conllu").write_text("", encoding="utf-8")
    out_dir = tmp_path / "out"

    rc = cli.main(
        [
            "--chapters",
            "7",
            "--parsed-dir",
            str(parsed_dir),
            "--output-dir",
            str(out_dir),
            "--allow-missing",
        ]
    )
    assert rc != 0
    err = capsys.readouterr().err
    assert "EMPTY_PARSED_INPUT" in err


# ---------------------------------------------------------------------------
# 6. Legitimate zero hits.
# ---------------------------------------------------------------------------


def test_legitimate_zero_hits_writes_empty_tsv(tmp_path, cli):
    """File exists, nonempty, no tokens match the filter lists → zero_hits_ok."""
    parsed_dir = tmp_path / "parsed"
    _write_parsed(parsed_dir, 3, SYNTH_NO_HITS)
    out_dir = tmp_path / "out"

    rc = cli.main(
        [
            "--chapters",
            "3",
            "--parsed-dir",
            str(parsed_dir),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert all(m["status"] == "zero_hits_ok" for m in manifest)
    assert all(m["row_count"] == 0 for m in manifest)

    # Empty TSV with header only.
    contracted_tsv = out_dir / "hp1_de_ch03_contracted.tsv"
    assert contracted_tsv.exists()
    rows = _read_tsv(contracted_tsv)
    assert rows == []


# ---------------------------------------------------------------------------
# 7. Happy path with hits.
# ---------------------------------------------------------------------------


def test_happy_path_with_hits(tmp_path, cli):
    """File exists, sentences contain contracted/uncontracted PPs → status=ok."""
    parsed_dir = tmp_path / "parsed"
    _write_parsed(parsed_dir, 1, SYNTH_BOTH)
    out_dir = tmp_path / "out"

    rc = cli.main(
        [
            "--chapters",
            "1",
            "--parsed-dir",
            str(parsed_dir),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    c_rows = _read_tsv(out_dir / "hp1_de_ch01_contracted.tsv")
    u_rows = _read_tsv(out_dir / "hp1_de_ch01_uncontracted.tsv")
    assert len(c_rows) == 1
    assert len(u_rows) == 1

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    statuses = {m["form"]: m["status"] for m in manifest}
    assert statuses["contracted"] == "ok"
    assert statuses["uncontracted"] == "ok"
    assert all(m["row_count"] == 1 for m in manifest)


def test_documented_path_emits_both_provenance_columns(tmp_path, cli):
    """The README-documented extraction command must emit rows carrying
    BOTH ``parse_block_id`` and ``source_segment_id`` — the two-level
    provenance downstream (Step 4 master, cross-lingual mapping) joins on."""
    parsed_dir = tmp_path / "parsed"
    _write_parsed(parsed_dir, 1, SYNTH_BOTH)
    out_dir = tmp_path / "out"

    rc = cli.main(
        [
            "--chapters",
            "1",
            "--parsed-dir",
            str(parsed_dir),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    for name, block, segment in (
        ("hp1_de_ch01_contracted.tsv", "synth_c#b001", "synth_c"),
        ("hp1_de_ch01_uncontracted.tsv", "synth_u#b001", "synth_u"),
    ):
        rows = _read_tsv(out_dir / name)
        assert len(rows) == 1
        assert rows[0]["parse_block_id"] == block
        assert rows[0]["source_segment_id"] == segment


# ---------------------------------------------------------------------------
# 8. in_filter flag doesn't gate row count.
# ---------------------------------------------------------------------------


def test_in_filter_flag_doesnt_gate_row_count(tmp_path, cli):
    """A hit OUTSIDE FILTER_CONTRACTED_123 still appears in the TSV with in_filter=N."""
    parsed_dir = tmp_path / "parsed"
    _write_parsed(parsed_dir, 2, SYNTH_OUT_OF_FILTER)
    out_dir = tmp_path / "out"

    rc = cli.main(
        [
            "--chapters",
            "2",
            "--parsed-dir",
            str(parsed_dir),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    rows = _read_tsv(out_dir / "hp1_de_ch02_contracted.tsv")
    assert len(rows) == 1  # not gated
    assert rows[0]["in_filter"] == "N"
    assert rows[0]["prep"] == "im"


def test_in_filter_y_for_paper_match(tmp_path, cli):
    """A hit IN FILTER_CONTRACTED_123 gets in_filter=Y."""
    parsed_dir = tmp_path / "parsed"
    _write_parsed(parsed_dir, 1, SYNTH_CONTRACTED_HIT)
    out_dir = tmp_path / "out"

    rc = cli.main(
        [
            "--chapters",
            "1",
            "--parsed-dir",
            str(parsed_dir),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    rows = _read_tsv(out_dir / "hp1_de_ch01_contracted.tsv")
    assert len(rows) == 1
    assert rows[0]["in_filter"] == "Y"


# ---------------------------------------------------------------------------
# 9. Determinism.
# ---------------------------------------------------------------------------


def test_determinism_same_sha_and_row_count(tmp_path, cli):
    """Same input → same parsed_path_sha256 and same row_count across runs."""
    parsed_dir = tmp_path / "parsed"
    _write_parsed(parsed_dir, 4, SYNTH_BOTH)

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"

    rc = cli.main(
        ["--chapters", "4", "--parsed-dir", str(parsed_dir), "--output-dir", str(out_a)]
    )
    assert rc == 0
    rc = cli.main(
        ["--chapters", "4", "--parsed-dir", str(parsed_dir), "--output-dir", str(out_b)]
    )
    assert rc == 0

    m_a = json.loads((out_a / "manifest.json").read_text(encoding="utf-8"))
    m_b = json.loads((out_b / "manifest.json").read_text(encoding="utf-8"))
    for a, b in zip(m_a, m_b, strict=False):
        assert a["parsed_path_sha256"] == b["parsed_path_sha256"]
        assert a["row_count"] == b["row_count"]


# ---------------------------------------------------------------------------
# 10. Chapter range validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_chapter", [0, 18, 99, -1])
def test_chapter_out_of_range_exits_2(tmp_path, cli, bad_chapter):
    rc = cli.main(
        [
            "--chapters",
            str(bad_chapter),
            "--parsed-dir",
            str(tmp_path / "parsed"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2


def test_chapter_out_of_range_emits_correct_rule(tmp_path, cli, capsys):
    rc = cli.main(
        [
            "--chapters",
            "18",
            "--parsed-dir",
            str(tmp_path / "parsed"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "CHAPTER_OUT_OF_RANGE" in err


# ---------------------------------------------------------------------------
# 11. Manifest structure.
# ---------------------------------------------------------------------------


REQUIRED_FIELDS = {
    "chapter",
    "form",
    "parsed_path",
    "parsed_path_sha256",
    "parsed_path_size",
    "parsed_path_sentence_count",
    "extractor_version",
    "status",
    "row_count",
}


def test_manifest_has_all_required_fields(tmp_path, cli):
    parsed_dir = tmp_path / "parsed"
    _write_parsed(parsed_dir, 1, SYNTH_BOTH)
    out_dir = tmp_path / "out"

    rc = cli.main(
        [
            "--chapters",
            "1",
            "--parsed-dir",
            str(parsed_dir),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) >= 2
    for entry in manifest:
        missing = REQUIRED_FIELDS - entry.keys()
        assert not missing, f"manifest entry missing fields: {missing}"
        assert entry["form"] in {"contracted", "uncontracted"}
        assert entry["extractor_version"]
        assert entry["status"] in {
            "ok",
            "zero_hits_ok",
            "missing_input",
            "empty_input",
            "extraction_error",
        }
        assert isinstance(entry["row_count"], int)
        assert isinstance(entry["parsed_path_sha256"], str)
        assert isinstance(entry["parsed_path_size"], int)
        assert isinstance(entry["parsed_path_sentence_count"], int)


def test_manifest_error_field_only_on_errors(tmp_path, cli):
    parsed_dir = tmp_path / "parsed"
    _write_parsed(parsed_dir, 1, SYNTH_BOTH)
    out_dir = tmp_path / "out"

    rc = cli.main(
        [
            "--chapters",
            "1",
            "--parsed-dir",
            str(parsed_dir),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest:
        if entry["status"] in {"ok", "zero_hits_ok"}:
            assert "error" not in entry


def test_manifest_missing_input_has_error_field(tmp_path, cli):
    parsed_dir = tmp_path / "parsed"  # empty
    out_dir = tmp_path / "out"

    rc = cli.main(
        [
            "--chapters",
            "5",
            "--parsed-dir",
            str(parsed_dir),
            "--output-dir",
            str(out_dir),
            "--allow-missing",
        ]
    )
    assert rc == 0

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest:
        assert entry["status"] == "missing_input"
        assert "error" in entry
        assert entry["error"] == "MISSING_PARSED_INPUT"


# ---------------------------------------------------------------------------
# 12. Stdout privacy guard.
# ---------------------------------------------------------------------------


PRIVACY_SENSITIVE_TOKENS = ["Haus", "Wald", "Berg", "im", "in", "dem", "synth"]


def test_stdout_privacy_no_noun_or_prep_strings(tmp_path, cli, capsys):
    """No synthetic noun/prep/sentence-id strings may appear on stdout."""
    parsed_dir = tmp_path / "parsed"
    _write_parsed(parsed_dir, 1, SYNTH_BOTH)
    out_dir = tmp_path / "out"

    rc = cli.main(
        [
            "--chapters",
            "1",
            "--parsed-dir",
            str(parsed_dir),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0

    captured = capsys.readouterr()
    for token in PRIVACY_SENSITIVE_TOKENS:
        assert token not in captured.out, (
            f"privacy leak: {token!r} appeared in stdout: {captured.out!r}"
        )
        assert token not in captured.err, (
            f"privacy leak: {token!r} appeared in stderr: {captured.err!r}"
        )

    # Aggregate counts ARE allowed on stdout.
    assert "Ch.1" in captured.out
    assert "manifest:" in captured.out
    assert "total_chapter_entries:" in captured.out


def test_stdout_privacy_missing_input_no_path_contents(tmp_path, cli, capsys):
    """A missing-input failure must not print the path's contents."""
    parsed_dir = tmp_path / "parsed"  # empty
    out_dir = tmp_path / "out"

    rc = cli.main(
        [
            "--chapters",
            "5",
            "--parsed-dir",
            str(parsed_dir),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc != 0
    captured = capsys.readouterr()
    # The rule name and chapter number are allowed.
    assert "MISSING_PARSED_INPUT" in captured.err
    assert "Ch.5" in captured.err
    # But not the directory listing or path's contents.
    assert "ls " not in captured.err
    assert "No such file" not in captured.err


# ---------------------------------------------------------------------------
# Extras: default output-dir manifest path, --manifest override.
# ---------------------------------------------------------------------------


def test_default_manifest_path_under_output_dir(tmp_path, cli):
    parsed_dir = tmp_path / "parsed"
    _write_parsed(parsed_dir, 1, SYNTH_CONTRACTED_HIT)
    out_dir = tmp_path / "out"

    rc = cli.main(
        [
            "--chapters",
            "1",
            "--parsed-dir",
            str(parsed_dir),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "manifest.json").exists()


def test_manifest_path_override(tmp_path, cli):
    parsed_dir = tmp_path / "parsed"
    _write_parsed(parsed_dir, 1, SYNTH_CONTRACTED_HIT)
    out_dir = tmp_path / "out"
    manifest_path = tmp_path / "elsewhere" / "m.json"

    rc = cli.main(
        [
            "--chapters",
            "1",
            "--parsed-dir",
            str(parsed_dir),
            "--output-dir",
            str(out_dir),
            "--manifest",
            str(manifest_path),
        ]
    )
    assert rc == 0
    assert manifest_path.exists()
    # Default path is NOT created when --manifest is set.
    assert not (out_dir / "manifest.json").exists()


# ---------------------------------------------------------------------------
# Direct API tests (no CLI indirection) for the core extract_chapter.
# ---------------------------------------------------------------------------


def test_extract_chapter_raises_on_missing_path(tmp_path, monkeypatch):
    from hp_corpus import german_extraction as ge

    monkeypatch.setattr(ge, "CONTRACTED", ["im"])
    monkeypatch.setattr(ge, "PREPOSITIONS", ["in"])
    monkeypatch.setattr(ge, "DETERMINERS", ["dem"])

    missing = tmp_path / "does_not_exist.conllu"
    with pytest.raises(ge.MissingParsedInputError):
        ge.extract_chapter(missing, contracted=True)


def test_extract_chapter_raises_on_empty_input(tmp_path, monkeypatch):
    from hp_corpus import german_extraction as ge

    monkeypatch.setattr(ge, "CONTRACTED", ["im"])
    monkeypatch.setattr(ge, "PREPOSITIONS", ["in"])
    monkeypatch.setattr(ge, "DETERMINERS", ["dem"])

    empty = tmp_path / "empty.conllu"
    # Truly empty — pyconll yields zero sentences. (A file with only a
    # comment line yields one empty sentence, which is a different case.)
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ge.EmptyParsedInputError):
        ge.extract_chapter(empty, contracted=True)


def test_extract_chapter_zero_hits_returns_empty_list(tmp_path, monkeypatch):
    from hp_corpus import german_extraction as ge

    monkeypatch.setattr(ge, "CONTRACTED", ["im"])
    monkeypatch.setattr(ge, "PREPOSITIONS", ["in"])
    monkeypatch.setattr(ge, "DETERMINERS", ["dem"])

    p = tmp_path / "nohits.conllu"
    p.write_text(SYNTH_NO_HITS, encoding="utf-8")
    hits = ge.extract_chapter(p, contracted=True)
    assert hits == []


def test_extract_chapter_hit_shape(tmp_path, monkeypatch):
    """extract_chapter's hit dict carries the paper's (prep, det, noun)
    fields plus occurrence coordinates and both provenance columns — the
    shape the full-novel TSV writer and Step 4 consume. (The legacy
    run_paper_extractor.extract was removed; this script's german_extraction
    module is the single implementation.)"""
    from hp_corpus import german_extraction as ge

    monkeypatch.setattr(ge, "CONTRACTED", ["im"])
    monkeypatch.setattr(ge, "PREPOSITIONS", ["in"])
    monkeypatch.setattr(ge, "DETERMINERS", ["dem"])

    p = tmp_path / "synth.conllu"
    p.write_text(SYNTH_BOTH, encoding="utf-8")

    c_hits = ge.extract_chapter(p, contracted=True)
    u_hits = ge.extract_chapter(p, contracted=False)

    assert len(c_hits) == 1
    assert len(u_hits) == 1
    c = c_hits[0]
    assert c["prep"] == "im"
    assert c["noun"] == "Haus"
    assert c["det"] is None
    assert c["parse_block_id"] == "synth_c#b001"
    assert c["source_segment_id"] == "synth_c"
    assert c["prep_token_id"] == "1"
    assert c["noun_token_id"] == "2"
    assert c["pp_token_start"] == "1"
    assert c["pp_token_end"] == "2"
    assert c["pp_surface"] == "im Haus"

    u = u_hits[0]
    assert u["prep"] == "in"
    assert u["det"] == "dem"
    assert u["noun"] == "Wald"
    assert u["parse_block_id"] == "synth_u#b001"
    assert u["source_segment_id"] == "synth_u"
    assert u["prep_token_id"] == "1"
    assert u["det_token_id"] == "2"
    assert u["noun_token_id"] == "3"
    assert u["pp_token_start"] == "1"
    assert u["pp_token_end"] == "3"
    assert u["pp_surface"] == "in dem Wald"


def test_write_chapter_tsv_writes_header_only_for_empty(tmp_path):
    from hp_corpus import german_extraction as ge

    out = tmp_path / "empty.tsv"
    ge.write_chapter_tsv(out, [], [])
    text = out.read_text(encoding="utf-8")
    # Header line + trailing newline only.
    lines = text.rstrip("\n").split("\n")
    assert lines == ["\t".join(ge.TSV_FIELDS)]


def test_chapter_manifest_validates_status():
    from hp_corpus import german_extraction as ge

    with pytest.raises(ValueError):
        ge.chapter_manifest(
            chapter=1,
            contracted=True,
            parsed_path=Path("/nonexistent"),
            hits=[],
            status="bogus",
        )
