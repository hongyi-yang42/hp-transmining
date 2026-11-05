"""Synthetic positive/negative tests for scripts/audit_contractibility.py.

The audit script is diagnostic-only (contrast variant); these tests pin its
rule surface — the (prep, det) contractibility map, the strict normalizer,
and the stdlib xlsx parser — so silent edits get caught. No real corpus
files are read; the xlsx fixtures are built in-memory with zipfile.
"""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_contractibility.py"
_spec = importlib.util.spec_from_file_location("audit_contractibility", _SCRIPT)
audit = importlib.util.module_from_spec(_spec)
sys.modules["audit_contractibility"] = audit
_spec.loader.exec_module(audit)

PAPER_PREPS = {
    "zu",
    "an",
    "auf",
    "für",
    "von",
    "gegen",
    "aus",
    "in",
    "durch",
    "über",
    "unter",
    "hinter",
    "bei",
}
DETS = {"das", "dem", "der"}

STANDARD_CELLS = {
    ("an", "das"),
    ("an", "dem"),
    ("auf", "das"),
    ("bei", "dem"),
    ("durch", "das"),
    ("für", "das"),
    ("in", "das"),
    ("in", "dem"),
    ("über", "das"),
    ("von", "dem"),
    ("zu", "dem"),
    ("zu", "der"),
}
COLLOQUIAL_CELLS = {
    ("auf", "dem"),
    ("für", "dem"),
    ("über", "dem"),
    ("aus", "dem"),
    ("gegen", "das"),
    ("hinter", "das"),
    ("hinter", "dem"),
    ("unter", "das"),
    ("unter", "dem"),
}


class TestContractibilityMap:
    def test_covers_all_paper_cells(self):
        assert set(audit.CONTRACTIBILITY) == {(p, d) for p in PAPER_PREPS for d in DETS}

    def test_values_are_valid_classes(self):
        for (prep, det), (cls, _form) in audit.CONTRACTIBILITY.items():
            assert cls in {"standard", "colloquial", "none"}, (prep, det, cls)

    def test_standard_cells_exact(self):
        got = {k for k, v in audit.CONTRACTIBILITY.items() if v[0] == "standard"}
        assert got == STANDARD_CELLS

    def test_colloquial_cells_exact(self):
        got = {k for k, v in audit.CONTRACTIBILITY.items() if v[0] == "colloquial"}
        assert got == COLLOQUIAL_CELLS

    def test_feminine_der_standard_only_for_zu(self):
        der_cells = {p for p in PAPER_PREPS if audit.CONTRACTIBILITY[(p, "der")][0] == "standard"}
        assert der_cells == {"zu"}

    def test_nine_prep_scope_is_standard_preps(self):
        with_std = {
            p
            for p in PAPER_PREPS
            if any(audit.CONTRACTIBILITY[(p, d)][0] == "standard" for d in DETS)
        }
        assert with_std == audit.EXPECTED_9PREPS
        # colloquial-only preps are exactly the four the paper's FILTER_PP lacks
        assert PAPER_PREPS - with_std == {"aus", "gegen", "hinter", "unter"}

    def test_classify_boundaries(self):
        assert audit.classify("zu", "der") == "standard"  # zur
        assert audit.classify("auf", "der") == "none"  # fem: no form
        assert audit.classify("auf", "dem") == "colloquial"  # aufm
        assert audit.classify("in", "dem") == "standard"  # im
        assert audit.classify("durch", "dem") == "none"  # no durchm anywhere

    def test_classify_unknown_combo(self):
        assert audit.classify("von", "das") == "none"
        assert audit.classify("um", "dem") == "unmapped"  # not a paper prep


class TestStrictNorm:
    def test_spelling_reform(self):
        assert audit.strict_norm("Schloß") == "schloss"
        assert audit.strict_norm("Schluß") == "schluss"

    def test_trailing_period(self):
        assert audit.strict_norm("Arbeit.") == "arbeit"

    def test_lowercase(self):
        assert audit.strict_norm("Tribüne") == "tribüne"

    def test_clean_lemma_noop(self):
        assert audit.strict_norm("Haus") == "haus"


_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _sheet_xml(rows: list[list[tuple[str, str, str]]]) -> bytes:
    """rows = list of (ref, t_attr, value) cell tuples, one list per row."""
    body = []
    for i, cells in enumerate(rows, start=1):
        parts = []
        for ref, t, v in cells:
            attr = f' t="{t}"' if t else ""
            inner = f"<is><t>{v}</t></is>" if t == "inlineStr" else f"<v>{v}</v>"
            parts.append(f'<c r="{ref}"{attr}>{inner}</c>')
        body.append(f'<row r="{i}">{"".join(parts)}</row>')
    return (
        f'<?xml version="1.0"?><worksheet xmlns="{_NS}"><sheetData>'
        + "".join(body)
        + "</sheetData></worksheet>"
    ).encode()


def _xlsx(path: Path, sheet: bytes, shared: bytes | None = None) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("xl/worksheets/sheet1.xml", sheet)
        if shared is not None:
            zf.writestr("xl/sharedStrings.xml", shared)


class TestReadXlsx:
    def test_inline_strings_and_numbers(self, tmp_path):
        sheet = _sheet_xml(
            [
                [("A1", "inlineStr", "id"), ("B1", "inlineStr", "de_valid")],
                [("A2", "inlineStr", "row-1"), ("B2", "", "42")],
                [("A3", "inlineStr", "row-2"), ("B3", "inlineStr", "include")],
            ]
        )
        p = tmp_path / "t.xlsx"
        _xlsx(p, sheet)
        rows = audit.read_xlsx_rows(p)
        assert rows[0] == {"id": "row-1", "de_valid": "42"}
        assert rows[1] == {"id": "row-2", "de_valid": "include"}

    def test_shared_strings(self, tmp_path):
        shared = (
            f'<?xml version="1.0"?><sst xmlns="{_NS}" count="2" uniqueCount="2">'
            "<si><t>id</t></si><si><t>x</t></si></sst>"
        ).encode()
        sheet = _sheet_xml([[("A1", "s", "0"), ("B1", "s", "1")], [("A2", "s", "1")]])
        p = tmp_path / "t.xlsx"
        _xlsx(p, sheet, shared)
        rows = audit.read_xlsx_rows(p)
        assert rows[0] == {"id": "x", "x": ""}

    def test_empty_cells_default_blank(self, tmp_path):
        sheet = _sheet_xml(
            [
                [("A1", "inlineStr", "id")],
                [("A2", "inlineStr", "r"), ("B2", "inlineStr", "")],
            ]
        )
        p = tmp_path / "t.xlsx"
        _xlsx(p, sheet)
        rows = audit.read_xlsx_rows(p)
        assert rows[0]["id"] == "r"


class TestAnchors:
    def test_add_and_failures(self):
        a = audit.Anchors()
        a.add("x", 1, 1)
        a.add("y", 2, 3)
        assert a.failures() == [("y", "2", "3")]
        report = a.report()
        assert "PASS" in report and "FAIL" in report


def test_anchor_id_sets_are_disjoint():
    # the 4 residual rows must not collide with the 5 in-list colloquial rows
    assert not (audit.RESIDUAL_STANDARD_IDS & audit.INLIST_COLLOQUIAL_IDS)
    assert not (audit.RESIDUAL_STANDARD_IDS & audit.SPELLING_ROW_IDS)
