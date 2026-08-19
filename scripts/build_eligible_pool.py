"""Build the formal German eligible pool after the German review.

This is the one production selection path (see ``docs/FULL_NOVEL_SAMPLING.md``).
It runs exactly once per completed German review and needs three inputs:

  * the **full extraction set** — all 34 per-chapter contracted +
    uncontracted TSVs from ``run_full_novel_german_extraction.py``
    (Ch.1–17, both forms; a missing file is a hard failure — there is
    no chapter-subset option);
  * the **machine master** (``full_novel_annotation_master.tsv``) — the
    source of each row's ``source_row_sha256``;
  * the **returned annotator CSV** (``annotation_pairs.csv`` from
    ``build_annotation_csv.py``, filled in by the annotators — see
    ``docs/ANNOTATION_CSV.md``). Its ``de_valid`` column is the final
    German decision; ``de_corrected_head_lemma`` the lemma correction.

Fail-closed validation of the returned file, all before anything is
written:

  * duplicate ``id`` in the returned CSV;
  * returned id not present in the master;
  * master row not covered by the returned CSV (incomplete review);
  * ``row_hash`` ≠ the master hash for the same id;
  * any ``de_valid`` other than ``include`` / ``exclude`` (blank or
    ``uncertain`` means the review is not finished).

Interpretation note: EN/ZH ``omitted`` marks are only a true omission
when the same side's alignment confidence is ``high``/``medium``/``low``;
``omitted`` + ``not_aligned`` means retrieval failed and the row needs
repair (see docs/ANNOTATION_CSV.md). Any analysis counting omissions
must apply that distinction — it is not applied here.

The pool rule itself (``hp_corpus.sampling.build_eligible_pool``):
13-item paired preposition inventory; uncontracted Ch.1–17 after the
ART+det structural gate and an ``include`` review; contracted Ch.1–3
after ``include``; contracted Ch.4–17 only when its
``(canonical_prep, head_lemma)`` pair — corrected lemma when non-blank,
else machine lemma — occurs in the reviewed-include uncontracted set.

Outputs (under ``<out-dir>``):

  * ``eligible_pool.tsv`` — the eligible rows only;
  * ``eligible_pool_summary.json`` — aggregate counts on the actual
    inputs: extracted, automatically excluded (inventory / structural),
    human included / excluded, and the paper-eligible breakdown.

Stdout discipline: aggregate counts only. Never prints lemmas,
datapoint IDs, surface forms, or sentence text.

Usage::

    uv run python scripts/build_eligible_pool.py \\
        --extraction-dir data/extracted/full_novel \\
        --master-tsv data/derived/step4/full_novel_annotation_master.tsv \\
        --review-csv <returned-annotation_pairs.csv> \\
        --out-dir data/derived/eligible_pool
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from hp_corpus.sampling import (
    FULL_NOVEL_CHAPTERS,
    REVIEW_DECISIONS,
    IncompleteReviewError,
    Occurrence,
    OccurrenceIdentityConflictError,
    StructuralMetadataMissingError,
    build_eligible_pool,
    canonical_preposition,
    in_paired_inventory,
)

EXIT_OK = 0
EXIT_INPUT_ERROR = 2

SUMMARY_SCHEMA_VERSION = "eligible-pool-v1"

# Extraction TSV columns this CLI depends on. A header missing any of
# these is an upstream schema break — fail closed up front.
_REQUIRED_EXTRACTION_COLUMNS = (
    "parse_block_id",
    "source_segment_id",
    "prep",
    "noun",
    "pp_token_start",
    "pp_token_end",
    "det_xpos",
    "det_deprel",
)

# REVIEW_DECISIONS is imported from hp_corpus.sampling — the single
# definition shared by the core rule, this CLI, and the annotation-CSV
# schema (hp_corpus.annotation_csv.DE_VALID_VALUES).

POOL_TSV_COLUMNS = [
    "datapoint_id",
    "chapter",
    "form",
    "canonical_prep",
    "head_lemma",
    "machine_head_lemma",
    "corrected_head_lemma",
    "pool_reason",
    "source_segment_id",
    "parse_block_id",
    "pp_token_start",
    "pp_token_end",
    "source_hash",
]


def _exit(rule: str, message: str) -> int:
    print(f"FAIL: rule={rule} {message}", file=sys.stderr)
    return EXIT_INPUT_ERROR


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_datapoint_id(chapter: int, parse_block_id: str, token_start: str, token_end: str) -> str:
    return f"dp_ch{chapter:02d}_{parse_block_id}_t{token_start}-{token_end}"


def _read_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader), list(reader.fieldnames or [])


def load_extraction(extraction_dir: Path) -> list[dict[str, str]]:
    """Read the FULL extraction set — every chapter 1–17, both forms.
    Any missing chapter/form file is a hard failure — the pool must be
    built on the complete input, never a subset."""
    rows: list[dict[str, str]] = []
    for ch in FULL_NOVEL_CHAPTERS:
        for kind in ("contracted", "uncontracted"):
            path = extraction_dir / f"hp1_de_ch{ch:02d}_{kind}.tsv"
            if not path.exists():
                raise FileNotFoundError(path)
            body, header = _read_tsv(path)
            missing = [c for c in _REQUIRED_EXTRACTION_COLUMNS if c not in header]
            if missing:
                raise ValueError(
                    f"{path}: extraction header is missing required columns: {missing}"
                )
            for r in body:
                r["_chapter"] = str(ch)
                r["_form"] = kind
            rows.extend(body)
    return rows


def load_master(path: Path) -> dict[str, dict[str, str]]:
    """Index the machine master by datapoint_id. A duplicate id fails —
    it would otherwise silently overwrite and leave the hash anchor
    ambiguous."""
    body, _ = _read_tsv(path)
    out: dict[str, dict[str, str]] = {}
    for row in body:
        dp = row.get("datapoint_id", "")
        if not dp:
            continue
        if dp in out:
            raise ValueError(f"duplicate datapoint_id in machine master: {dp}")
        out[dp] = row
    return out


def check_extraction_master_consistency(
    extraction_rows: list[dict[str, str]], master_index: dict[str, dict[str, str]]
) -> None:
    """Bidirectional ID consistency between the extraction set and the
    machine master.

    The master is the extraction set minus the inventory-ineligible
    rows, so the eligible extraction ids and the master ids must be the
    *same set*. A file's existence is not its content: an extraction
    regression that drops rows, or a master built from a different
    extraction, must fail rather than silently shrink (or grow) the
    pool.
    """
    expected = {
        _make_datapoint_id(
            int(r["_chapter"]),
            r.get("parse_block_id", ""),
            r.get("pp_token_start", ""),
            r.get("pp_token_end", ""),
        )
        for r in extraction_rows
        if in_paired_inventory(canonical_preposition(r.get("prep", "")))
    }
    master_ids = set(master_index)
    if expected != master_ids:
        missing_from_master = len(expected - master_ids)
        missing_from_extraction = len(master_ids - expected)
        raise ValueError(
            "extraction set and machine master do not correspond "
            f"(inventory-eligible extraction rows absent from master: "
            f"{missing_from_master}; master rows with no inventory-eligible "
            f"extraction row: {missing_from_extraction})"
        )


def load_review_csv(
    path: Path, master_index: dict[str, dict[str, str]]
) -> dict[str, dict[str, str]]:
    """Read + validate the returned annotator CSV as the final German
    review (the ``de_valid`` column is the decision; ``row_hash`` binds
    each row to the master). Every rule here fails closed; nothing
    downstream may see a partial review."""
    # utf-8-sig tolerates Excel round-trips with or without the BOM.
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        body = list(reader)
        header = list(reader.fieldnames or [])
    required = ["id", "row_hash", "de_valid", "de_corrected_lemma"]
    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(f"returned review CSV is missing columns: {missing}")

    review: dict[str, dict[str, str]] = {}
    for row in body:
        dp = (row.get("id", "") or "").strip()
        if not dp:
            raise ValueError("returned review CSV row with empty id")
        if dp in review:
            raise ValueError(f"duplicate id in returned review CSV: {dp}")
        master_row = master_index.get(dp)
        if master_row is None:
            raise ValueError(
                f"returned review CSV id not present in the machine master: {dp}"
            )
        row_hash = (row.get("row_hash", "") or "").strip()
        master_hash = master_row.get("source_row_sha256", "")
        if row_hash != master_hash or not row_hash:
            raise ValueError(f"row hash mismatch for {dp}")
        decision = (row.get("de_valid", "") or "").strip()
        if decision not in REVIEW_DECISIONS:
            raise ValueError(
                f"review not complete: de_valid for {dp} is {decision!r} "
                "(expected include/exclude)"
            )
        review[dp] = row

    unreviewed = [dp for dp in master_index if dp not in review]
    if unreviewed:
        raise ValueError(
            f"review incomplete: {len(unreviewed)} master row(s) have no "
            "entry in the returned CSV"
        )
    return review


def build_occurrences(
    extraction_rows: list[dict[str, str]],
    master_index: dict[str, dict[str, str]],
    review: dict[str, dict[str, str]],
) -> list[Occurrence]:
    out: list[Occurrence] = []
    for r in extraction_rows:
        block_id = r.get("parse_block_id", "")
        segment_id = r.get("source_segment_id", "")
        if not block_id or not segment_id:
            raise ValueError(
                "extraction row without parse_block_id / source_segment_id provenance columns"
            )
        ch = int(r["_chapter"])
        prep_surface = r.get("prep", "")
        machine_lemma = r.get("noun", "")
        token_start = r.get("pp_token_start", "")
        token_end = r.get("pp_token_end", "")
        dp = _make_datapoint_id(ch, block_id, token_start, token_end)
        canonical = canonical_preposition(prep_surface)
        eligible = in_paired_inventory(canonical)
        master_row = master_index.get(dp)
        # Extraction↔master correspondence is enforced up front by
        # check_extraction_master_consistency; an inventory-ineligible
        # row may legitimately have no master row.
        source_hash = master_row.get("source_row_sha256", "") if master_row else ""
        review_row = review.get(dp)
        out.append(
            Occurrence(
                datapoint_id=dp,
                chapter=ch,
                form=r["_form"],
                canonical_prep=canonical,
                machine_head_lemma=machine_lemma,
                decision=(review_row.get("de_valid", "") or "").strip() if review_row else "",
                corrected_head_lemma=(
                    review_row.get("de_corrected_lemma", "") if review_row else ""
                ),
                inventory_eligible=eligible,
                source_hash=source_hash,
                source_segment_id=segment_id,
                parse_block_id=block_id,
                pp_token_start=int(token_start) if token_start else -1,
                pp_token_end=int(token_end) if token_end else -1,
                det_xpos=r.get("det_xpos", ""),
                det_deprel=r.get("det_deprel", ""),
            )
        )
    return out


def write_pool_tsv(path: Path, result) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=POOL_TSV_COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for row in result.rows:
            if not row.eligible:
                continue
            o = row.occurrence
            w.writerow(
                {
                    "datapoint_id": o.datapoint_id,
                    "chapter": o.chapter,
                    "form": o.form,
                    "canonical_prep": o.canonical_prep,
                    "head_lemma": row.head_lemma,
                    "machine_head_lemma": o.machine_head_lemma,
                    "corrected_head_lemma": o.corrected_head_lemma,
                    "pool_reason": row.pool_reason,
                    "source_segment_id": o.source_segment_id,
                    "parse_block_id": o.parse_block_id,
                    "pp_token_start": o.pp_token_start,
                    "pp_token_end": o.pp_token_end,
                    "source_hash": o.source_hash,
                }
            )
    return path


def write_summary(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extraction-dir", type=Path, required=True)
    ap.add_argument("--master-tsv", type=Path, required=True)
    ap.add_argument("--review-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--force-output",
        action="store_true",
        help="Required if any output file already exists.",
    )
    args = ap.parse_args(argv)

    if not args.extraction_dir.exists():
        return _exit("EXTRACTION_DIR_NOT_FOUND", str(args.extraction_dir))
    if not args.master_tsv.exists():
        return _exit("MASTER_TSV_ABSENT", str(args.master_tsv))
    if not args.review_csv.exists():
        return _exit("REVIEW_CSV_ABSENT", str(args.review_csv))

    pool_path = args.out_dir / "eligible_pool.tsv"
    summary_path = args.out_dir / "eligible_pool_summary.json"
    for p in (pool_path, summary_path):
        if p.exists() and not args.force_output:
            return _exit("OUTPUT_EXISTS", f"{p}; pass --force-output to overwrite")

    # The formal run always reads the complete Ch.1-17 extraction set —
    # there is no chapter subset to select, by design.
    try:
        extraction_rows = load_extraction(args.extraction_dir)
    except FileNotFoundError as exc:
        return _exit("MISSING_EXTRACTION_INPUT", str(exc))
    except ValueError as exc:
        return _exit("EXTRACTION_SCHEMA_MISMATCH", str(exc))

    try:
        master_index = load_master(args.master_tsv)
    except ValueError as exc:
        return _exit("MASTER_DUPLICATE_ID", str(exc))
    try:
        check_extraction_master_consistency(extraction_rows, master_index)
    except ValueError as exc:
        return _exit("EXTRACTION_MASTER_MISMATCH", str(exc))
    try:
        review = load_review_csv(args.review_csv, master_index)
    except ValueError as exc:
        # Every returned-file rule fails closed: duplicate ids,
        # id-not-in-master, master-row-not-reviewed, row-hash mismatch,
        # incomplete decisions.
        return _exit("REVIEW_OVERLAY_INVALID", str(exc))

    try:
        occurrences = build_occurrences(extraction_rows, master_index, review)
    except ValueError as exc:
        return _exit("EXTRACTION_MASTER_MISMATCH", str(exc))
    if not occurrences:
        return _exit("NO_OCCURRENCES", "extraction set is empty")

    try:
        result = build_eligible_pool(occurrences)
    except OccurrenceIdentityConflictError as exc:
        return _exit("OCCURRENCE_IDENTITY_CONFLICT", str(exc))
    except StructuralMetadataMissingError as exc:
        return _exit("STRUCTURAL_METADATA_MISSING", str(exc))
    except IncompleteReviewError as exc:
        return _exit("INCOMPLETE_REVIEW", str(exc))

    # Input hashes for reproducibility (file names only, no content).
    extraction_files = sorted(args.extraction_dir.glob("hp1_de_ch*.tsv"))
    summary = {
        **result.summary,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "input_hashes": {
            "extraction": {p.name: _sha256(p) for p in extraction_files},
            "master": _sha256(args.master_tsv),
            "review_csv": _sha256(args.review_csv),
        },
    }
    write_pool_tsv(pool_path, result)
    write_summary(summary_path, summary)

    # Aggregate stdout — no lemmas, no IDs, no surface forms.
    s = result.summary
    print(f"extracted_total: {s['extracted_total']}")
    print(f"duplicate_rows_collapsed: {s['duplicate_rows_collapsed']}")
    auto = s["automatically_excluded"]
    print(
        f"automatically_excluded: inventory={auto['outside_paired_inventory']} "
        f"structural={auto['failed_structural_gate']}"
    )
    review = s["human_review"]
    print(f"human_review: included={review['included']} excluded={review['excluded']}")
    ep = s["eligible_pool"]
    print(
        f"eligible_pool: uncontracted_all_chapters={ep['uncontracted_all_chapters']} "
        f"contracted_ch1_3={ep['contracted_ch1_3']} "
        f"contracted_ch4_17_pair_matched={ep['contracted_ch4_17_pair_matched']} "
        f"eligible_total={ep['eligible_total']}"
    )
    no_pair = s["contracted_ch4_17_no_uncontracted_counterpart"]
    print(f"contracted_ch4_17_no_uncontracted_counterpart: {no_pair}")
    print(f"pool: {pool_path}")
    print(f"summary: {summary_path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
