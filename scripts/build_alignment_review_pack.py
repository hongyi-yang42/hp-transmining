"""Build a gitignored manual alignment review TSV for the Step 4 pilot 20.

For each of the 20 pilot datapoints, emits one row containing:
  * datapoint_id, chapter, de_form
  * de_sentence_ids, de_text, de_token_range
  * en_sentence_ids, en_text, en_cardinality, en_confidence
  * zh_sentence_ids, zh_text, zh_cardinality, zh_confidence
  * one preceding + one following context sentence per language (de/en/zh)

Plus blank human-check columns (never auto-filled):
  en_scene_match, zh_scene_match, en_counterpart_locatable,
  zh_counterpart_locatable, alignment_issue, review_notes

Stdout prints aggregate counts only — never novel text. The TSV itself is
written under ``data/derived/`` (gitignored) so it can carry source text for
the human reviewer without entering git.

Usage:
    uv run python scripts/build_alignment_review_pack.py \\
        [--pilot-tsv data/derived/step4/ch1_3_pilot_20.tsv] \\
        [--segmented-dir data/segmented] \\
        [--output data/derived/alignment_v2/manual_review_pack.tsv]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from hp_corpus.schema import Segment

SEGMENTED_DIR = Path("data/segmented")
DEFAULT_PILOT = Path("data/derived/step4/ch1_3_pilot_20.tsv")
DEFAULT_OUTPUT = Path("data/derived/alignment_v2/manual_review_pack.tsv")

CONTEXT = 1  # one preceding + one following sentence per language


def _load_segments_chapter(lang: str, chapter: int, segmented_dir: Path) -> list[Segment]:
    path = segmented_dir / f"hp1_{lang}_ch{chapter:02d}.jsonl"
    if not path.exists():
        return []
    out: list[Segment] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(Segment.model_validate_json(line))
    return out


def _build_index(segs: list[Segment]) -> tuple[dict[str, int], list[str]]:
    """Return (id → position index, ordered list of IDs by file order).
    The file order is assumed to be chapter-monotonic (QC confirms this)."""
    id_to_pos = {s.id: i for i, s in enumerate(segs)}
    return id_to_pos, [s.id for s in segs]


def _join_text(segs: list[Segment], ids: list[str]) -> str:
    """Concatenate text for a list of segment IDs in given order, separated
    by ' / ' so multi-sentence alignments stay readable in a TSV cell."""
    if not ids:
        return ""
    id_to_seg = {s.id: s for s in segs}
    parts = [id_to_seg[i].text for i in ids if i in id_to_seg]
    return " / ".join(parts)


def _context(segs: list[Segment], ids: list[str], n: int) -> tuple[str, str, str]:
    """Return (prev_text, curr_marker, next_text) — the n preceding and
    following sentences relative to the first/last id in ``ids``.

    Returns ("", "", "") if no ids or no neighbors."""
    if not ids or not segs:
        return "", "", ""
    id_to_pos, _ = _build_index(segs)
    positions = [id_to_pos[i] for i in ids if i in id_to_pos]
    if not positions:
        return "", "", ""
    lo = min(positions)
    hi = max(positions)
    prev_ids = [segs[i].id for i in range(max(0, lo - n), lo)]
    next_ids = [segs[i].id for i in range(hi + 1, min(len(segs), hi + 1 + n))]
    prev_text = _join_text(segs, prev_ids)
    next_text = _join_text(segs, next_ids)
    return prev_text, "«" + _join_text(segs, ids) + "»", next_text


def _cardinality_from_alignment(ids_src: list[str], ids_tgt: list[str]) -> str:
    return f"{len(ids_src)}:{len(ids_tgt)}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot-tsv", type=Path, default=DEFAULT_PILOT)
    ap.add_argument("--segmented-dir", type=Path, default=SEGMENTED_DIR)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args(argv)

    if not args.pilot_tsv.exists():
        print(f"FAIL: pilot TSV not found: {args.pilot_tsv}", file=sys.stderr)
        return 2

    # Load pilot rows.
    with open(args.pilot_tsv, encoding="utf-8") as f:
        pilot_rows = list(csv.DictReader(f, delimiter="\t"))
    if not pilot_rows:
        print("FAIL: pilot TSV has no rows", file=sys.stderr)
        return 2

    # Pre-load segmented files per (lang, chapter). Chapters in pilot may be 1-3.
    chapters_needed = sorted({int(r["chapter"]) for r in pilot_rows})
    segs_cache: dict[tuple[str, int], list[Segment]] = {}
    for ch in chapters_needed:
        for lang in ("de", "en", "zh"):
            segs_cache[(lang, ch)] = _load_segments_chapter(lang, ch, args.segmented_dir)

    # Output columns.
    headers = [
        "datapoint_id",
        "chapter",
        "de_form",
        "de_sentence_ids",
        "de_token_range",
        "de_text",
        "de_context_prev",
        "de_context_next",
        "en_sentence_ids",
        "en_text",
        "en_cardinality",
        "en_confidence",
        "en_context_prev",
        "en_context_next",
        "zh_sentence_ids",
        "zh_text",
        "zh_cardinality",
        "zh_confidence",
        "zh_context_prev",
        "zh_context_next",
        # ---- blank human-check columns (annotator fills) ----
        "en_scene_match",
        "zh_scene_match",
        "en_counterpart_locatable",
        "zh_counterpart_locatable",
        "alignment_issue",
        "review_notes",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(headers)
        for r in pilot_rows:
            ch = int(r["chapter"])
            # de_sentence_id is singular in the pilot TSV (one DE sentence per
            # PP datapoint). en/zh_sentence_ids are JSON lists.
            de_ids = [r["de_sentence_id"]] if r.get("de_sentence_id") else []
            en_ids = json.loads(r["en_sentence_ids"]) if r.get("en_sentence_ids") else []
            zh_ids = json.loads(r["zh_sentence_ids"]) if r.get("zh_sentence_ids") else []

            de_segs = segs_cache.get(("de", ch), [])
            en_segs = segs_cache.get(("en", ch), [])
            zh_segs = segs_cache.get(("zh", ch), [])

            de_prev, _, de_next = _context(de_segs, de_ids, CONTEXT)
            en_prev, _, en_next = _context(en_segs, en_ids, CONTEXT)
            zh_prev, _, zh_next = _context(zh_segs, zh_ids, CONTEXT)

            w.writerow([
                r["datapoint_id"],
                r["chapter"],
                r["de_form"],
                r["de_sentence_id"],
                f"{r['de_token_start']}-{r['de_token_end']}",
                r.get("de_sentence_text", ""),
                de_prev,
                de_next,
                r.get("en_sentence_ids", ""),
                r.get("en_aligned_text", ""),
                r["en_alignment_cardinality"],
                r["en_alignment_confidence"],
                en_prev,
                en_next,
                r.get("zh_sentence_ids", ""),
                r.get("zh_aligned_text", ""),
                r["zh_alignment_cardinality"],
                r["zh_alignment_confidence"],
                zh_prev,
                zh_next,
                "",  # en_scene_match
                "",  # zh_scene_match
                "",  # en_counterpart_locatable
                "",  # zh_counterpart_locatable
                "",  # alignment_issue
                "",  # review_notes
            ])

    # Aggregate stdout — no novel text.
    by_chapter = {}
    for r in pilot_rows:
        ch = r["chapter"]
        by_chapter[ch] = by_chapter.get(ch, 0) + 1
    by_form = {}
    for r in pilot_rows:
        by_form[r["de_form"]] = by_form.get(r["de_form"], 0) + 1

    print(f"rows: {len(pilot_rows)}")
    print(f"by_chapter: {by_chapter}")
    print(f"by_form: {by_form}")
    print(f"output: {args.output}")
    print(f"  ({len(headers)} columns; {6} blank human-check columns at end)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
