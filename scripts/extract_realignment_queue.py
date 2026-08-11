"""Extract rows flagged for sentence-alignment review into a queue TSV.

Scans an annotated Step 4 TSV and emits every row where either EN or ZH
``alignment_qc`` is ``incorrect`` or ``uncertain``. The output is the
worklist for a source-data alignment fix; it is **not** an editable-cell
correction. Correcting an alignment changes the source-side
``{lang}_aligned_text`` and ``source_row_sha256`` on every affected row,
so the path forward is:

  1. Annotator flags bad-alignment rows via ``{lang}_alignment_qc``.
  2. This script collects them into ``realignment_queue.tsv``.
  3. Human fixes the underlying alignment (out of scope here — the
     sentence-alignment algorithm in ``src/hp_corpus/align.py`` is
     untouched in this pass).
  4. ``build_ch1_3_full_annotation.py`` is re-run after the fix,
     producing a new annotation-target TSV with the same
     ``datapoint_id``s but updated alignment fields and source hashes.
  5. A follow-up migration script (out of scope here; see
     ``docs/STEP4_ANNOTATION.md``) carries over annotation fields from
     the old TSV to the new one by ``datapoint_id``, skipping the side
     whose alignment changed.

The queue TSV is written under ``data/derived/step4/`` (gitignored —
contains novel text). Stdout prints aggregate counts only (rows queued,
by chapter, by QC state) so it is safe for terminal/CI output.

Usage:
    uv run python scripts/extract_realignment_queue.py \\
        data/derived/step4/ch1_3_full_annotation.tsv \\
        [--output data/derived/step4/realignment_queue.tsv]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from hp_corpus.step4 import ALL_TSV_COLUMNS

# Columns written to the queue TSV. Carries enough context for a human to
# investigate the alignment issue (DE sentence id, current target ids and
# aligned text, notes) without re-opening the source TSV. Output is
# gitignored — contains novel text.
QUEUE_COLUMNS = [
    "datapoint_id",
    "chapter",
    "de_sentence_id",
    "de_pp_surface",
    "de_form",
    "de_candidate_decision",
    "en_alignment_qc",
    "en_sentence_ids",
    "en_aligned_text",
    "en_alignment_notes",
    "zh_alignment_qc",
    "zh_sentence_ids",
    "zh_aligned_text",
    "zh_alignment_notes",
    "general_notes",
]

# QC values that route a row to the realignment queue.
_QUEUE_QC_VALUES = frozenset({"incorrect", "uncertain"})


def _row_is_queued(row: dict[str, str]) -> tuple[bool, list[str]]:
    """Return (queued, sides_flagged). A row is queued iff any EN/ZH
    alignment_qc is in {incorrect, uncertain}."""
    sides: list[str] = []
    for lang in ("en", "zh"):
        qc = row.get(f"{lang}_alignment_qc", "").strip()
        if qc in _QUEUE_QC_VALUES:
            sides.append(lang)
    return (bool(sides), sides)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tsv", type=Path)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/step4/realignment_queue.tsv"),
    )
    args = ap.parse_args(argv)

    if not args.tsv.exists():
        print(f"FAIL: input TSV not found: {args.tsv}", file=sys.stderr)
        return 2

    with open(args.tsv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", fieldnames=list(ALL_TSV_COLUMNS))
        header = next(reader, None)
        if header is None:
            print("FAIL: TSV is empty", file=sys.stderr)
            return 2
        rows = list(reader)

    queued_rows: list[dict[str, str]] = []
    by_chapter: Counter[str] = Counter()
    by_side: Counter[str] = Counter()
    by_qc_value: Counter[str] = Counter()
    for r in rows:
        is_queued, sides = _row_is_queued(r)
        if not is_queued:
            continue
        queued_rows.append({col: r.get(col, "") for col in QUEUE_COLUMNS})
        by_chapter[str(r.get("chapter", "?"))] += 1
        for s in sides:
            by_side[s] += 1
            qc = r.get(f"{s}_alignment_qc", "").strip()
            if qc:
                by_qc_value[f"{s}:{qc}"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=QUEUE_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        w.writeheader()
        for r in queued_rows:
            w.writerow(r)

    # Stdout: aggregate counts only.
    print(f"input rows: {len(rows)}")
    print(f"queued rows: {len(queued_rows)}")
    print(f"  by chapter: {dict(sorted(by_chapter.items()))}")
    print(f"  by side flagged: {dict(sorted(by_side.items()))}")
    print(f"  by side:qc: {dict(sorted(by_qc_value.items()))}")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
