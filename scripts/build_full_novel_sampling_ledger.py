"""Build the full-novel sampling ledger from extraction + master TSVs.

Reads the per-chapter German PP extraction TSVs produced by
``run_full_novel_german_extraction.py`` (Ch.1-17), joins each occurrence
to the master annotation TSV (for ``de_candidate_decision`` and an
optional ``de_reviewed_lemma`` column), runs the pure sampling rule from
``hp_corpus.sampling``, and writes:

  * ``<out-dir>/full_novel_ledger.tsv``  — every extracted occurrence,
    one row, with sampling reason / status / effective lemma /
    supporting C_late IDs.
  * ``<out-dir>/full_novel_target.tsv``  — the selected rows joined back
    to the master annotation TSV's ``ALL_TSV_COLUMNS`` shape, ready to
    hand to annotators. (If the master TSV is absent, this file is
    skipped with a stderr message — the ledger is still written.)
  * ``<out-dir>/full_novel_summary.json`` — aggregate counts.

Methodological note: the C_late expansion uses the head-noun lemma
ALONE, not ``canonical_preposition + lemma``. See
``docs/FULL_NOVEL_SAMPLING.md`` for the rationale and the
``manual_lemma_override`` mechanism for parser-lemma corrections.

Stdout discipline: aggregate counts only. Never prints lemmas,
datapoint IDs, surface forms, or sentence text.

Usage::

    uv run python scripts/build_full_novel_sampling_ledger.py \\
        --extraction-dir data/extracted/full_novel \\
        --master-tsv data/derived/step4/ch1_3_full_annotation.tsv \\
        --chapters 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 \\
        --out-dir data/derived/sampling
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from hp_corpus.sampling import (
    CHAPTER_RANGE_FULL_NOVEL,
    Occurrence,
    SamplingRow,
    canonical_preposition,
    is_inventory_eligible,
    select_sample,
)
from hp_corpus.step4 import ALL_TSV_COLUMNS

EXIT_OK = 0
EXIT_INPUT_ERROR = 2

LEDGER_COLUMNS = [
    "datapoint_id",
    "chapter",
    "form",
    "canonical_prep",
    "machine_head_lemma",
    "reviewed_head_lemma",
    "effective_matching_lemma",
    "german_candidate_decision",
    "inventory_eligible",
    "sampling_selected",
    "sampling_reason",
    "sampling_status",
    "supports_late_contracted_ids",
    "manual_lemma_override",
    "source_hash",
]


def _exit(rule: str, message: str) -> int:
    print(f"FAIL: rule={rule} {message}", file=sys.stderr)
    return EXIT_INPUT_ERROR


def _make_datapoint_id(chapter: int, sentence_id: str, token_start: str, token_end: str) -> str:
    return f"dp_ch{chapter:02d}_{sentence_id}_t{token_start}-{token_end}"


def _load_extraction_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _load_master_index(path: Path) -> dict[str, dict[str, str]]:
    """Index the master TSV by datapoint_id. Tolerates a missing file
    (returns empty dict) so the CLI can run on pure extraction data."""
    if not path or not path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            dp = row.get("datapoint_id", "")
            if dp:
                out[dp] = row
    return out


def build_occurrences(
    extraction_dir: Path,
    chapters: list[int],
    master_index: dict[str, dict[str, str]],
) -> list[Occurrence]:
    """Read each chapter's contracted + uncontracted extraction TSVs and
    produce Occurrence records, joining to the master for German review
    decisions and an optional reviewed lemma."""
    out: list[Occurrence] = []
    for ch in chapters:
        for form, kind in (("contracted", "contracted"), ("uncontracted", "uncontracted")):
            path = extraction_dir / f"hp1_de_ch{ch:02d}_{kind}.tsv"
            try:
                rows = _load_extraction_tsv(path)
            except FileNotFoundError:
                continue
            for r in rows:
                sentence_id = r.get("sentence_id", "")
                prep_surface = r.get("prep", "")
                machine_lemma = r.get("noun", "")
                token_start = r.get("pp_token_start", "")
                token_end = r.get("pp_token_end", "")
                dp = _make_datapoint_id(ch, sentence_id, token_start, token_end)
                master_row = master_index.get(dp, {})
                reviewed_lemma = master_row.get("de_reviewed_lemma", "")
                manual_override = master_row.get("manual_lemma_override", "")
                decision = master_row.get("de_candidate_decision", "")
                # Source hash — prefer the master's hash if the row is in
                # the master; otherwise compute a stable hash from the
                # extraction coordinates.
                source_hash = master_row.get("source_row_sha256", "")
                if not source_hash:
                    source_hash = _extraction_hash(
                        dp, sentence_id, prep_surface, machine_lemma, token_start, token_end
                    )
                canonical = canonical_preposition(prep_surface)
                out.append(
                    Occurrence(
                        datapoint_id=dp,
                        chapter=ch,
                        form=form,
                        canonical_prep=canonical,
                        machine_head_lemma=machine_lemma,
                        reviewed_head_lemma=reviewed_lemma,
                        german_candidate_decision=decision,
                        inventory_eligible=is_inventory_eligible(canonical),
                        source_hash=source_hash,
                        manual_lemma_override=manual_override,
                    )
                )
    return out


def _extraction_hash(*parts: str) -> str:
    import hashlib

    blob = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_ledger(path: Path, result) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=LEDGER_COLUMNS, delimiter="\t", lineterminator="\n"
        )
        w.writeheader()
        for row in result.ledger:
            w.writerow(_ledger_row_to_dict(row))
    return path


def _ledger_row_to_dict(row: SamplingRow) -> dict[str, Any]:
    o = row.occurrence
    return {
        "datapoint_id": o.datapoint_id,
        "chapter": o.chapter,
        "form": o.form,
        "canonical_prep": o.canonical_prep,
        "machine_head_lemma": o.machine_head_lemma,
        "reviewed_head_lemma": o.reviewed_head_lemma,
        "effective_matching_lemma": row.effective_matching_lemma,
        "german_candidate_decision": o.german_candidate_decision,
        "inventory_eligible": "true" if o.inventory_eligible else "false",
        "sampling_selected": "true" if row.sampling_selected else "false",
        "sampling_reason": row.sampling_reason,
        "sampling_status": row.sampling_status,
        "supports_late_contracted_ids": json.dumps(
            list(row.supports_late_contracted_ids), ensure_ascii=False
        ),
        "manual_lemma_override": o.manual_lemma_override,
        "source_hash": o.source_hash,
    }


def write_target(
    path: Path,
    result,
    master_index: dict[str, dict[str, str]],
) -> Path | None:
    """Write the analysis-target TSV in ALL_TSV_COLUMNS shape. Rows that
    aren't in the master TSV are skipped (we don't have their full source
    columns). Returns the path, or None if no selected rows were in the
    master."""
    selected_in_master = [
        r
        for r in result.ledger
        if r.sampling_selected and r.occurrence.datapoint_id in master_index
    ]
    if not selected_in_master:
        return None
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
        for r in selected_in_master:
            master_row = master_index[r.occurrence.datapoint_id]
            w.writerow({col: master_row.get(col, "") for col in ALL_TSV_COLUMNS})
    return path


def write_summary(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extraction-dir", type=Path, required=True)
    ap.add_argument("--master-tsv", type=Path, default=None)
    ap.add_argument(
        "--chapters",
        type=int,
        nargs="+",
        default=list(CHAPTER_RANGE_FULL_NOVEL),
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--use-machine-lemma",
        action="store_true",
        help="Build a provisional ledger using machine lemma even when "
        "reviewed lemma is available (default: prefer reviewed lemma).",
    )
    ap.add_argument(
        "--force-output",
        action="store_true",
        help="Required if any output file already exists.",
    )
    args = ap.parse_args(argv)

    if not args.extraction_dir.exists():
        return _exit("EXTRACTION_DIR_NOT_FOUND", str(args.extraction_dir))
    if any(ch not in CHAPTER_RANGE_FULL_NOVEL for ch in args.chapters):
        return _exit(
            "CHAPTER_OUT_OF_RANGE",
            "chapters must be in 1..17",
        )

    ledger_path = args.out_dir / "full_novel_ledger.tsv"
    target_path = args.out_dir / "full_novel_target.tsv"
    summary_path = args.out_dir / "full_novel_summary.json"
    for p in (ledger_path, summary_path):
        if p.exists() and not args.force_output:
            return _exit("OUTPUT_EXISTS", f"{p}; pass --force-output to overwrite")

    master_index = _load_master_index(args.master_tsv) if args.master_tsv else {}
    occurrences = build_occurrences(args.extraction_dir, args.chapters, master_index)
    if not occurrences:
        return _exit(
            "NO_OCCURRENCES",
            f"no extraction TSVs found under {args.extraction_dir} for chapters {args.chapters}",
        )

    result = select_sample(occurrences, use_reviewed_lemma=not args.use_machine_lemma)

    write_ledger(ledger_path, result)
    target_written = write_target(target_path, result, master_index) if master_index else None
    summary = {
        **result.summary,
        "use_reviewed_lemma": not args.use_machine_lemma,
        "master_tsv_present": bool(master_index),
        "selected_in_master": sum(
            1
            for r in result.ledger
            if r.sampling_selected and r.occurrence.datapoint_id in master_index
        ),
    }
    write_summary(summary_path, summary)

    # Aggregate stdout — no lemmas, no IDs, no surface forms.
    print(f"occurrence_total: {result.summary['occurrence_total']}")
    print(f"selected_total: {result.summary['selected_total']}")
    print(f"u_lemma_count: {result.summary['u_lemma_count']}")
    print("by_status:")
    for k in sorted(result.summary["by_status"]):
        print(f"  {k}: {result.summary['by_status'][k]}")
    print("by_reason:")
    for k in sorted(result.summary["by_reason"]):
        print(f"  {k}: {result.summary['by_reason'][k]}")
    print(f"ledger: {ledger_path}")
    if target_written is not None:
        print(f"target: {target_written}")
    else:
        print("target: skipped (master TSV absent or no selected rows in master)")
    print(f"summary: {summary_path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
