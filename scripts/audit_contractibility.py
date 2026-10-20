#!/usr/bin/env python3
"""Contractibility audit for the Bremmers reproduction — diagnostic only.

Checks the "could have contracted" condition (AC2019 section 4.2: "contexts
in which the preposition could have contracted"): an uncontracted German PP
candidate (preposition + separate article) falls inside the paper's
candidate space only if that specific (preposition, article) combination has
a STANDARD WRITTEN contracted form — in+dem->im, zu+der->zur, but NOT fem
"auf der" (no such form exists) and NOT "auf dem"/"über dem" (aufm/überm are
colloquial only).

Status: contrast variant. Per the project policy (paper-verbatim-first,
2026-10-12) the morphological gate is NOT a selection rule; this script
never feeds the pipeline, the annotator CSV, or the eligible pool. It (a)
re-derives the ad-hoc numbers of tmp/research/PAPER_96_FORENSICS.md section
6.2 as a persistent, re-runnable artifact with anchor checks, (b) maps
contractibility onto the current machine candidates (master TSV + annotator
CSV), (c) cross-tabs it against the human review batch, and (d) quantifies
the deferred ch4+ minimal-pair gate (paper condition 3).

Note: the authors' own CONTRACTED token list (vendor data.py) includes
colloquial forms (aufm/fürn/fürm/übern/überm/untern/unterm/hinters/hinterm/
gegens/ausm/ums/vorm) while their FILTER lists exclude every colloquial-only
preposition — their selection is consistent with the standard-written-form
reading adopted here.

Outputs (gitignored): tmp/research/contractibility_audit/
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

REPO = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = REPO / "data" / "extracted" / "full_novel"
MASTER_TSV = REPO / "data" / "derived" / "step4" / "full_novel_annotation_master.tsv"
PAIRS_CSV = REPO / "data" / "derived" / "annotation" / "annotation_pairs.csv"
LOWCONF_CSV = REPO / "data" / "derived" / "annotation" / "annotation_pairs_low_confidence.csv"
REVIEW_XLSX = (
    REPO / "data" / "derived" / "annotation" / "HP_annotation_pairs_001_150_integrated.xlsx"
)
VENDOR_DATA = REPO / "vendor" / "conll-extractor" / "conll_extractor" / "prepositions" / "data.py"

# (prep, det) -> (class, contracted form). 13 paper prepositions x {das, dem, der}.
# Standard vs colloquial per Duden grammar; colloquial cells carry the form
# that appears in the paper's own CONTRACTED token list.
CONTRACTIBILITY: dict[tuple[str, str], tuple[str, str | None]] = {
    ("an", "das"): ("standard", "ans"),
    ("an", "dem"): ("standard", "am"),
    ("an", "der"): ("none", None),  # no standard written form (*anr)
    ("auf", "das"): ("standard", "aufs"),
    ("auf", "dem"): ("colloquial", "aufm"),
    ("auf", "der"): ("none", None),
    ("bei", "das"): ("none", None),  # no *beis; beim = bei dem only
    ("bei", "dem"): ("standard", "beim"),
    ("bei", "der"): ("none", None),
    ("durch", "das"): ("standard", "durchs"),
    ("durch", "dem"): ("none", None),  # durchm: neither standard nor in the paper list
    ("durch", "der"): ("none", None),
    ("für", "das"): ("standard", "fürs"),
    ("für", "dem"): ("colloquial", "fürm"),
    ("für", "der"): ("none", None),
    ("gegen", "das"): ("colloquial", "gegens"),
    ("gegen", "dem"): ("none", None),
    ("gegen", "der"): ("none", None),
    ("aus", "das"): ("none", None),  # no *auss
    ("aus", "dem"): ("colloquial", "ausm"),
    ("aus", "der"): ("none", None),
    ("in", "das"): ("standard", "ins"),
    ("in", "dem"): ("standard", "im"),
    ("in", "der"): ("none", None),
    ("über", "das"): ("standard", "übers"),
    ("über", "dem"): ("colloquial", "überm"),
    ("über", "der"): ("none", None),
    ("unter", "das"): ("colloquial", "unters"),
    ("unter", "dem"): ("colloquial", "unterm"),
    ("unter", "der"): ("none", None),
    ("hinter", "das"): ("colloquial", "hinters"),
    ("hinter", "dem"): ("colloquial", "hinterm"),
    ("hinter", "der"): ("none", None),
    ("von", "das"): ("none", None),  # no *vons; vom = von dem only
    ("von", "dem"): ("standard", "vom"),
    ("von", "der"): ("none", None),
    ("zu", "das"): ("none", None),  # no *zus
    ("zu", "dem"): ("standard", "zum"),
    ("zu", "der"): ("standard", "zur"),  # the only feminine-der contraction
}

EXPECTED_9PREPS = {"an", "auf", "bei", "durch", "für", "in", "über", "von", "zu"}

SPELLING_ROW_IDS = {
    "hp1_de_ch06_p0112_s004",
    "hp1_de_ch09_p0022_s003",
    "hp1_de_ch13_p0037_s015",
    "hp1_de_ch16_p0095_s008",
    "hp1_de_ch17_p0043_s003",
}
RESIDUAL_STANDARD_IDS = {
    "hp1_de_ch03_p0003_s019",  # in dem grauen Wasser — true miss (im-able)
    "hp1_de_ch10_p0048_s002",  # zu der offenen Tür — true miss (zur-able)
    "hp1_de_ch12_p0002_s006",  # zu mit dem Vorschlag — our artifact (stray zu)
    "hp1_de_ch06_p0076_s006",  # Haufen vs list 'Haufe' — lemma-convention diff
}
INLIST_COLLOQUIAL_IDS = {
    "hp1_de_ch03_p0048_s006",  # auf dem Sofa
    "hp1_de_ch05_p0002_s013",  # auf dem Sofa
    "hp1_de_ch08_p0017_s005",  # über dem Feuer
    "hp1_de_ch11_p0001_s009",  # auf dem Spielfeld
    "hp1_de_ch13_p0029_s007",  # über dem Spiel
}
ZU_DER_TYPES = ["Frau", "Katze", "Schlange", "Tribüne"]


def strict_norm(lemma: str) -> str:
    """FORENSICS section 6.4 'strict': sz->ss, strip trailing period, lowercase."""
    return lemma.replace("ß", "ss").rstrip(".").lower()


def load_vendor_lists() -> dict:
    spec = importlib.util.spec_from_file_location("bremmers_data", VENDOR_DATA)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {"FILTER_PP": mod.FILTER_PP, "PREPOSITIONS": mod.PREPOSITIONS}


def row_key(rec: dict) -> tuple:
    return (rec["parse_block_id"], rec["pp_token_start"], rec["pp_token_end"])


def load_extraction() -> list[dict]:
    rows = []
    for path in sorted(EXTRACTED_DIR.glob("hp1_de_ch*.tsv")):
        with open(path, encoding="utf-8", newline="") as fh:
            for rec in csv.DictReader(fh, delimiter="\t"):
                rec["pp_token_start"] = int(rec["pp_token_start"])
                rec["pp_token_end"] = int(rec["pp_token_end"])
                rec["side"] = "contracted" if "_contracted." in path.name else "uncontracted"
                rows.append(rec)
    return rows


def load_master() -> list[dict]:
    keep = [
        "datapoint_id",
        "chapter",
        "de_parse_block_id",
        "de_token_start",
        "de_token_end",
        "de_prep_normalized",
        "de_head_lemma",
        "de_form",
        "author_resource_match",
        "minimal_pair_group",
        "source_row_sha256",
    ]
    rows = []
    with open(MASTER_TSV, encoding="utf-8-sig", newline="") as fh:
        for rec in csv.DictReader(fh, delimiter="\t"):
            rows.append({k: rec.get(k, "") for k in keep})
    return rows


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def read_xlsx_rows(path: Path) -> list[dict]:
    """First worksheet of an xlsx via stdlib (inline strings or sharedStrings)."""
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root:
                shared.append("".join(t.text or "" for t in si.iter() if _local(t.tag) == "t"))
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    rows: list[dict] = []
    header: list[str] = []
    for row_el in root.iter():
        if _local(row_el.tag) != "row":
            continue
        cells: dict[int, str] = {}
        for c in row_el:
            if _local(c.tag) != "c":
                continue
            col_idx = 0
            for ch in "".join(ch for ch in c.get("r", "") if ch.isalpha()):
                col_idx = col_idx * 26 + (ord(ch) - 64)
            t = c.get("t", "n")
            if t == "inlineStr":
                val = "".join(x.text or "" for x in c.iter() if _local(x.tag) == "t")
            elif t == "s":
                v = next((x.text for x in c if _local(x.tag) == "v"), None)
                val = shared[int(v)] if v is not None else ""
            else:
                val = next((x.text or "" for x in c if _local(x.tag) == "v"), "")
            cells[col_idx] = val
        if not header:
            header = [cells.get(i, "") for i in range(1, max(cells) + 1)] if cells else []
            continue
        rows.append({header[i - 1]: cells.get(i, "") for i in range(1, len(header) + 1)})
    return [r for r in rows if any((v or "").strip() for v in r.values())]


def classify(prep: str, det: str) -> str:
    entry = CONTRACTIBILITY.get((prep, det))
    return entry[0] if entry else "unmapped"


@dataclass
class Anchors:
    checks: list[tuple[str, str, str, bool]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, name: str, expected, actual) -> None:
        self.checks.append((name, str(expected), str(actual), expected == actual))

    def failures(self) -> list[tuple[str, str, str]]:
        return [(n, e, a) for n, e, a, ok in self.checks if not ok]

    def report(self) -> str:
        lines = ["## Anchor checks", "", "| check | expected | actual | |"]
        lines.append("|---|---|---|---|")
        for name, exp, act, ok in self.checks:
            lines.append(f"| {name} | `{exp}` | `{act}` | {'PASS' if ok else '**FAIL**'} |")
        if self.notes:
            lines += ["", "## Notes (non-anchored observations)"] + [f"- {n}" for n in self.notes]
        return "\n".join(lines)


def fmt_counter(c: Counter | dict, n: int = 10) -> str:
    return ", ".join(f"{k}×{v}" for k, v in Counter(c).most_common(n))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--outdir", type=Path, default=REPO / "tmp" / "research" / "contractibility_audit"
    )
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    A = Anchors()

    vendor = load_vendor_lists()
    filter_pp = vendor["FILTER_PP"]
    nine_preps = set(filter_pp)
    A.add(
        "A:FILTER_PP preps == 9-with-standard-contraction",
        sorted(EXPECTED_9PREPS),
        sorted(nine_preps),
    )
    fpp_norm = {p: {strict_norm(x) for x in nouns} for p, nouns in filter_pp.items()}

    extraction = load_extraction()
    u_rows = [r for r in extraction if r["side"] == "uncontracted"]
    A.add("A:uncontracted occ (master ladder base)", 668, len(u_rows))
    # extraction carries ums x2 (um not in the 13 paper prepositions); the
    # canonical-prep pool gate drops them at master build (726 -> 724)
    A.add(
        "A:contracted occ at extraction level (incl. ums x2)",
        726,
        sum(1 for r in extraction if r["side"] == "contracted"),
    )
    ums_rows = [r for r in extraction if r["side"] == "contracted" and r["prep"] in {"ums", "vorm"}]
    A.add(
        "A:ums/vorm rows dropped by 13-prep canonical gate",
        ["hp1_de_ch04_p0030_s002", "hp1_de_ch15_p0021_s003"],
        sorted({r["source_segment_id"] for r in ums_rows}),
    )

    # ---------------- Phase A: re-derive FORENSICS section 6.2 ----------------
    in_scope = [r for r in u_rows if r["prep"] in nine_preps]
    out_scope = [r for r in u_rows if r["prep"] not in nine_preps]
    A.add("A:U occ in 9-prep scope", 512, len(in_scope))
    A.add("A:U occ 4-prep extra (13->9 prep drop)", 156, len(out_scope))
    A.add(
        "A:4-prep breakdown",
        {"aus": 109, "hinter": 17, "unter": 20, "gegen": 10},
        dict(Counter(r["prep"] for r in out_scope)),
    )

    raw_outside = [r for r in in_scope if r["in_filter"] == "N"]
    strict_outside = [r for r in in_scope if strict_norm(r["noun"]) not in fpp_norm[r["prep"]]]
    strict_keys = {row_key(r) for r in strict_outside}
    A.add("A:raw outside (pipeline in_filter=N)", 395, len(raw_outside))
    A.add("A:strict outside", 390, len(strict_outside))
    spelling = [r for r in raw_outside if row_key(r) not in strict_keys]
    A.add(
        "A:spelling-reform rows (raw-outside, strict-inside)",
        sorted(SPELLING_ROW_IDS),
        sorted({r["source_segment_id"] for r in spelling}),
    )

    impossible = [r for r in strict_outside if r["det"] == "der" and r["prep"] != "zu"]
    colloq_out = [r for r in strict_outside if classify(r["prep"], r["det"]) == "colloquial"]
    std_out = [r for r in strict_outside if classify(r["prep"], r["det"]) == "standard"]
    A.add("A:outside impossible fem-der (X!=zu)", 281, len(impossible))
    A.add("A:outside colloquial-dem", 105, len(colloq_out))
    A.add("A:outside standard residual", 4, len(std_out))
    A.add(
        "A:outside colloquial combos",
        [("auf", "dem"), ("über", "dem")],
        sorted({(r["prep"], r["det"]) for r in colloq_out}),
    )
    A.add(
        "A:residual standard row ids",
        sorted(RESIDUAL_STANDARD_IDS),
        sorted({r["source_segment_id"] for r in std_out}),
    )
    # FORENSICS 6.2 contributor counts are per (prep, noun) pair, not per noun
    imp_pairs = Counter((r["prep"], r["noun"]) for r in impossible)
    A.add(
        "A:impossible anchor (prep,noun) pairs",
        {("in", "Luft"): 15, ("in", "Hand"): 12, ("in", "Nähe"): 10, ("auf", "Seite"): 10},
        {
            k: imp_pairs[k]
            for k in (("in", "Luft"), ("in", "Hand"), ("in", "Nähe"), ("auf", "Seite"))
        },
    )
    coll_pairs = Counter((r["prep"], r["noun"]) for r in colloq_out)
    A.add(
        "A:colloquial anchor (prep,noun) pairs",
        {("auf", "Boden"): 26, ("auf", "Weg"): 16},
        {k: coll_pairs[k] for k in (("auf", "Boden"), ("auf", "Weg"))},
    )
    A.notes.append(
        "impossible-class det_xpos split: "
        f"{fmt_counter(Counter(r['det_xpos'] for r in impossible))}"
    )
    A.notes.append(
        "impossible-class det_deprel split: "
        f"{fmt_counter(Counter(r['det_deprel'] for r in impossible))}"
    )

    in_list = [r for r in in_scope if row_key(r) not in strict_keys]
    in_colloq = [r for r in in_list if classify(r["prep"], r["det"]) == "colloquial"]
    in_impossible = [r for r in in_list if r["det"] == "der" and r["prep"] != "zu"]
    A.add(
        "A:in-list colloquial rows (type-resolved das-frame)",
        sorted(INLIST_COLLOQUIAL_IDS),
        sorted({r["source_segment_id"] for r in in_colloq}),
    )
    A.add("A:in-list impossible rows", 0, len(in_impossible))
    zu_der = [r for r in in_list if (r["prep"], r["det"]) == ("zu", "der")]
    A.add(
        "A:(zu,der) in-list types = zur frame",
        sorted(ZU_DER_TYPES),
        sorted({r["noun"] for r in zu_der}),
    )
    A.notes.append(
        "(zu,der) in-list rows: "
        + fmt_counter(
            Counter(f"{r['noun']}(ch{r['source_segment_id'].split('_ch')[1][:2]})" for r in zu_der)
        )
    )
    std_total = [r for r in in_scope if classify(r["prep"], r["det"]) == "standard"]
    std_inlist = [r for r in in_list if classify(r["prep"], r["det"]) == "standard"]
    A.notes.append(
        f"STD-class occ in 9-prep scope: {len(std_total)}; in-list: {len(std_inlist)} "
        f"(FORENSICS 6.2 says '~121 occ / 119 in-list')"
    )

    # ---------------- Phase B: map onto current machine candidates ----------------
    master = load_master()
    A.add("B:master rows", 1392, len(master))
    ex_by_key = {row_key(r): r for r in extraction}
    join_failed = prep_mismatch = 0
    for m in master:
        m["chapter_int"] = int(m["chapter"])
        ex = ex_by_key.get(
            (m["de_parse_block_id"], int(m["de_token_start"]), int(m["de_token_end"]))
        )
        if ex is None:
            join_failed += 1
            m.update(
                det="",
                det_xpos="",
                det_deprel="",
                cls="join_failed",
                in_9prep="",
                strict_in="",
                in_filter_raw="",
            )
            continue
        m["det"], m["det_xpos"], m["det_deprel"] = ex["det"], ex["det_xpos"], ex["det_deprel"]
        m["in_filter_raw"] = ex["in_filter"]
        if m["de_form"] == "uncontracted":
            if ex["prep"] != m["de_prep_normalized"]:
                prep_mismatch += 1
            m["cls"] = classify(ex["prep"], ex["det"])
            m["in_9prep"] = "Y" if ex["prep"] in nine_preps else "N"
            m["strict_in"] = (
                "Y" if strict_norm(ex["noun"]) in fpp_norm.get(ex["prep"], set()) else "N"
            )
        else:
            m["cls"] = "contracted"
            m["in_9prep"] = "Y" if m["de_prep_normalized"] in nine_preps else "N"
            m["strict_in"] = ""
    A.add("B:master<->extraction join failures", 0, join_failed)
    A.add("B:prep label mismatches", 0, prep_mismatch)

    u_master = [m for m in master if m["de_form"] == "uncontracted"]
    cls_counts = Counter(m["cls"] for m in u_master)
    A.add("B:master U rows", 668, len(u_master))
    A.add(
        "B:U 9-prep none (= outside 281 + in-list 0)",
        281,
        sum(1 for m in u_master if m["in_9prep"] == "Y" and m["cls"] == "none"),
    )
    A.add(
        "B:U 9-prep colloquial (= outside 105 + in-list 5)",
        110,
        sum(1 for m in u_master if m["in_9prep"] == "Y" and m["cls"] == "colloquial"),
    )
    A.add(
        "B:U 9-prep standard",
        121,
        sum(1 for m in u_master if m["in_9prep"] == "Y" and m["cls"] == "standard"),
    )
    A.add(
        "B:U 4-prep colloquial",
        100,
        sum(1 for m in u_master if m["in_9prep"] == "N" and m["cls"] == "colloquial"),
    )
    A.add(
        "B:U 4-prep none",
        56,
        sum(1 for m in u_master if m["in_9prep"] == "N" and m["cls"] == "none"),
    )
    A.add("B:U standard total (gate survivors of 668)", 121, cls_counts.get("standard", 0))
    A.notes.append(f"master U class split: {fmt_counter(cls_counts)}")
    u9_cls = Counter(m["cls"] for m in u_master if m["in_9prep"] == "Y")
    A.notes.append(f"master U 9-prep class split: {fmt_counter(u9_cls)}")
    u4_cls = Counter(m["cls"] for m in u_master if m["in_9prep"] == "N")
    A.notes.append(f"master U 4-prep class split: {fmt_counter(u4_cls)}")

    with open(PAIRS_CSV, encoding="utf-8-sig", newline="") as fh:
        pairs = list(csv.DictReader(fh))
    master_by_id = {m["datapoint_id"]: m for m in master}
    A.add("B:annotation CSV rows", 1303, len(pairs))
    A.add("B:CSV ids missing from master", 0, sum(1 for p in pairs if p["id"] not in master_by_id))
    A.add(
        "B:CSV form split",
        {"contracted": 680, "uncontracted": 623},
        dict(Counter(p["form"] for p in pairs)),
    )
    csv_u_cls = Counter(master_by_id[p["id"]]["cls"] for p in pairs if p["form"] == "uncontracted")
    A.notes.append(f"CSV U-row class split (623): {fmt_counter(csv_u_cls)}")

    # ---------------- Phase C: human review batch cross-tab ----------------
    with open(LOWCONF_CSV, encoding="utf-8-sig", newline="") as fh:
        lowconf_ids = {r["id"] for r in csv.DictReader(fh)}
    review_all = read_xlsx_rows(REVIEW_XLSX)
    id_col = (
        "id"
        if review_all and "id" in review_all[0]
        else next(
            (c for c in ("datapoint_id", "row_id") if review_all and c in review_all[0]), None
        )
    )
    if id_col is None:
        print(
            "ERROR: no id column in xlsx; headers:",
            sorted(review_all[0]) if review_all else "empty",
        )
        return 2
    review_all = [r for r in review_all if (r.get(id_col) or "").strip()]
    # The integrated xlsx is the FULL CSV (1,303 rows) with human cells filled
    # on the first 150 rows only — the review window is rows 1-150.
    companion_idx = [i for i, r in enumerate(review_all) if r[id_col] in lowconf_ids]
    A.add("C:companion rows spliced into full sheet", 1, len(companion_idx))
    spliced_ids = [r[id_col] for r in review_all]
    for i in reversed(companion_idx):
        del spliced_ids[i]
    A.add(
        "C:sheet ids == CSV ids after companion removal",
        True,
        spliced_ids == [p["id"] for p in pairs],
    )
    if companion_idx:
        A.notes.append(
            f"companion row spliced at data-row {companion_idx[0] + 1} "
            f"({review_all[companion_idx[0]][id_col]})"
        )
    review = review_all[:150]
    A.add(
        "C:review window rows match CSV order (1-150)",
        True,
        [r[id_col] for r in review] == [p["id"] for p in pairs[:150]],
    )
    A.add("C:review rows", 150, len(review))
    A.add(
        "C:companion rows inside 150-row window",
        0,
        sum(1 for r in review if r[id_col] in lowconf_ids),
    )

    def cat_of(rid: str) -> str:
        m = master_by_id.get(rid)
        if m is None:
            return "missing"
        if m["de_form"] == "contracted":
            return "C"
        return "U9" if m["in_9prep"] == "Y" else "U4"

    def val(r: dict, c: str) -> str:
        return (r.get(c) or "").strip()

    group_a = [r for r in review if val(r, "de_valid").lower() in {"include", "exclude"}]
    group_b = [r for r in review if val(r, "de_valid").lower() not in {"include", "exclude"}]
    A.add("C:Group A judged", 75, len(group_a))
    A.add("C:Group B pending", 75, len(group_b))
    A.add(
        "C:Group A decisions",
        {"include": 44, "exclude": 31},
        dict(Counter(val(r, "de_valid").lower() for r in group_a)),
    )
    A.add(
        "C:Group A exclusion reasons",
        {"not_definite": 25, "not_target_pp": 6},
        dict(
            Counter(
                val(r, "de_exclusion_reason").lower()
                for r in group_a
                if val(r, "de_valid").lower() == "exclude"
            )
        ),
    )
    hash_bad = sum(
        1
        for r in group_a
        if master_by_id.get(r[id_col], {}).get("source_row_sha256") != val(r, "row_hash")
    )
    A.add("C:Group A row_hash mismatches vs master", 0, hash_bad)

    ga_ct = Counter(
        (cat_of(r[id_col]), master_by_id[r[id_col]]["cls"], val(r, "de_valid").lower())
        for r in group_a
        if r[id_col] in master_by_id
    )
    gb_ct = Counter(
        (cat_of(r[id_col]), master_by_id[r[id_col]]["cls"])
        for r in group_b
        if r[id_col] in master_by_id
    )
    ga_cat = Counter((cat_of(r[id_col]), val(r, "de_valid").lower()) for r in group_a)
    A.add(
        "C:Group A category x decision",
        {
            ("C", "include"): 23,
            ("C", "exclude"): 28,
            ("U9", "include"): 17,
            ("U9", "exclude"): 2,
            ("U4", "include"): 4,
            ("U4", "exclude"): 1,
        },
        dict(ga_cat),
    )
    gb_cat = Counter(cat_of(r[id_col]) for r in group_b)
    A.add("C:Group B categories", {"C": 31, "U9": 29, "U4": 15}, dict(gb_cat))
    gb_u4 = Counter(
        master_by_id[r[id_col]]["de_prep_normalized"] for r in group_b if cat_of(r[id_col]) == "U4"
    )
    A.add("C:Group B U4 prep breakdown", {"aus": 7, "gegen": 5, "unter": 3}, dict(gb_u4))

    tilde = sum(1 for r in review for c in ("en_form", "zh_form") if val(r, c) in {"~", "～"})
    A.notes.append(f"'~'/'～' placeholder cells in en_form/zh_form: {tilde}")
    ga_u9_incl = {k[1]: v for k, v in ga_ct.items() if k[0] == "U9" and k[2] == "include"}
    A.notes.append(f"Group A U9 includes by class: {fmt_counter(ga_u9_incl)}")
    ga_u9_excl = {k[1]: v for k, v in ga_ct.items() if k[0] == "U9" and k[2] == "exclude"}
    A.notes.append(f"Group A U9 excludes by class: {fmt_counter(ga_u9_excl)}")
    gb_u9 = {k[1]: v for k, v in gb_ct.items() if k[0] == "U9"}
    A.notes.append(f"Group B U9 by class: {fmt_counter(gb_u9)}")
    incl_nostd = sum(
        v
        for k, v in ga_ct.items()
        if k[2] == "include" and k[0] != "C" and k[1] in {"none", "colloquial"}
    )
    A.notes.append(
        f"Group A includes that a morphological gate would drop (U non-standard): {incl_nostd}"
    )

    # ---------------- Phase D: deferred ch4+ minimal-pair gate (condition 3) ----------------
    c_master = [m for m in master if m["de_form"] == "contracted"]
    A.add("D:master C rows", 724, len(c_master))
    empty_group = sum(1 for m in master if not m["minimal_pair_group"])
    A.notes.append(f"master rows with empty minimal_pair_group: {empty_group}")
    u_groups = {m["minimal_pair_group"] for m in u_master if m["minimal_pair_group"]}
    c13 = [m for m in c_master if m["chapter_int"] <= 3]
    c4 = [m for m in c_master if m["chapter_int"] >= 4]
    paired = [m for m in c4 if m["minimal_pair_group"] in u_groups]
    A.add("D:C ch1-3 (kept by paper rule)", 131, len(c13))
    A.add("D:C ch4+ total", 593, len(c4))
    A.add("D:C ch4+ with U counterpart (gate keeps)", 60, len(paired))
    A.add("D:C ch4+ cut by gate", 533, len(c4) - len(paired))
    A.add("D:paired distinct types", 19, len({m["minimal_pair_group"] for m in paired}))

    # ---------------- Outputs ----------------
    per_row = args.outdir / "per_row_contractibility.tsv"
    cols = [
        "datapoint_id",
        "chapter",
        "de_form",
        "de_prep_normalized",
        "det",
        "det_xpos",
        "det_deprel",
        "cls",
        "in_9prep",
        "strict_in",
        "in_filter_raw",
        "author_resource_match",
        "minimal_pair_group",
    ]
    with open(per_row, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(master)

    flags = args.outdir / "review_row_flags.tsv"
    with open(flags, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(
            [
                "xlsx_row",
                "id",
                "batch_group",
                "category",
                "class",
                "de_valid",
                "de_exclusion_reason",
                "is_companion",
                "row_hash_ok",
            ]
        )
        for i, r in enumerate(review, start=2):
            rid = r[id_col]
            m = master_by_id.get(rid, {})
            grp = "A" if val(r, "de_valid").lower() in {"include", "exclude"} else "B"
            w.writerow(
                [
                    i,
                    rid,
                    grp,
                    cat_of(rid),
                    m.get("cls", ""),
                    val(r, "de_valid"),
                    val(r, "de_exclusion_reason"),
                    "Y" if rid in lowconf_ids else "N",
                    "" if m.get("source_row_sha256") != val(r, "row_hash") else "Y",
                ]
            )

    ga_path = args.outdir / "group_a_crosstab.csv"
    with open(ga_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "class", "de_valid", "count"])
        for (cat, cls, dv), n in sorted(ga_ct.items()):
            w.writerow([cat, cls, dv, n])

    gb_path = args.outdir / "group_b_advisory.csv"
    with open(gb_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["category", "class", "count"])
        for (cat, cls), n in sorted(gb_ct.items()):
            w.writerow([cat, cls, n])

    summary = args.outdir / "summary_stats.md"
    with open(summary, "w", encoding="utf-8") as fh:
        fh.write("# Contractibility audit — generated summary\n\n")
        fh.write(A.report())
        fh.write("\n\n## Key derived numbers\n\n")
        survivors = cls_counts.get("standard", 0)
        fh.write(
            f"- Morphological gate effect on master U pool: {len(u_master)} -> "
            f"{survivors} (drop {len(u_master) - survivors})\n"
        )
        fh.write(f"- Group A includes a gate would drop: {incl_nostd}\n")
        fh.write(f"- ch4+ gate: 593 -> {len(paired)} contracted rows\n")

    print(A.report())
    print(f"\nOutputs written to {args.outdir}")
    fails = A.failures()
    if fails:
        print(f"\nANCHOR FAILURES: {len(fails)}")
        for n, e, a in fails:
            print(f"  - {n}: expected {e}, got {a}")
        return 1
    print("\nAll anchors PASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
