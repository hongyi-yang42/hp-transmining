"""Generic annotation context-pack builder.

Reads an annotation-target TSV (one row per German PP occurrence, with
the aligned EN/ZH sentence ids already populated by upstream steps) and
emits an enlarged TSV that adds, per language: one preceding + following
context sentence drawn from the segmented JSONL. The output is the
``manual_review_pack`` shape (see :data:`OUTPUT_COLUMNS`).

Stdout discipline is critical: the output TSV carries source novel text
(it lives under gitignored ``data/derived/``), but stdout must never
echo it. Only aggregate counts are printed:

    rows: <N>
    by_chapter: {1: K, 2: K, ...}
    by_form: {contracted: K, uncontracted: K}
    output: <path>
      (<N> columns; 6 blank human-check columns at end)

Failure modes (all exit 2, all messages aggregate-only):

    MISSING_REQUIRED_COLUMN   — input header missing a required column
    MALFORMED_DE_SENTENCE_ID  — de_sentence_id not canonical
    UNRESOLVED_SEGMENT_ID     — referenced segment id not in JSONL
    MALFORMED_SENTENCE_IDS_JSON — en/zh_sentence_ids cell not a JSON list

Usage::

    uv run python scripts/build_annotation_context_pack.py \\
        --input-tsv   data/derived/step4/ch1_3_pilot_20.tsv \\
        --segmented-dir data/segmented \\
        --output      data/derived/alignment_v2/manual_review_pack.tsv \\
        [--context-size 1] [--force-output]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# Allow running this script from a checkout without installing the
# package (mirrors the existing pilot script's implicit assumption).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hp_corpus.annotation_workflow import (  # noqa: E402
    N_BLANK_HUMAN_CHECK_COLUMNS,
    OUTPUT_COLUMNS,
    REQUIRED_INPUT_COLUMNS,
    ContextPackError,
    assert_de_sentence_id_wellformed,
    assert_ids_resolved,
    context_around,
    load_segments_chapter,
    parse_sentence_ids,
)

DEFAULT_SEGMENTED_DIR = Path("data/segmented")
DEFAULT_CONTEXT_SIZE = 1


def _die(rule: str, row: int | None, *, detail: str = "") -> int:
    """Emit a single stderr aggregate line and return exit code 2.
    Never includes source text or segment ids."""
    if row is None:
        msg = f"FAIL: rule={rule}"
    else:
        msg = f"FAIL: rule={rule} row={row}"
    if detail:
        msg += f" {detail}"
    print(msg, file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-tsv", type=Path, required=True)
    ap.add_argument(
        "--segmented-dir", type=Path, default=DEFAULT_SEGMENTED_DIR
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--context-size", type=int, default=DEFAULT_CONTEXT_SIZE
    )
    ap.add_argument(
        "--force-output",
        action="store_true",
        help="overwrite an existing --output (default: refuse)",
    )
    args = ap.parse_args(argv)

    if args.context_size < 0:
        return _die("NEGATIVE_CONTEXT_SIZE", None)

    if not args.input_tsv.exists():
        return _die("INPUT_TSV_MISSING", None)

    if args.output.exists() and not args.force_output:
        return _die(
            "OUTPUT_EXISTS",
            None,
            detail=f"(pass --force-output to overwrite {args.output})",
        )

    # ----- read input header + body -----
    with open(args.input_tsv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        header: list[str] = reader.fieldnames or []
        rows: list[dict[str, str]] = [dict(r) for r in reader]

    # Required-column check happens against the header, before any row
    # is touched, so row=None on the error path.
    for col in REQUIRED_INPUT_COLUMNS:
        if col not in header:
            return _die("MISSING_REQUIRED_COLUMN", None, detail=f"column={col}")

    # ----- pre-cache segmented files per (lang, chapter) seen in input -----
    chapters_seen: set[int] = set()
    for r in rows:
        try:
            chapters_seen.add(int(r["chapter"]))
        except (KeyError, ValueError):
            # Malformed chapter cell — surface as malformed-id rule but
            # without leaking text. Report the body row index.
            return _die("MALFORMED_CHAPTER", rows.index(r))

    segs_cache: dict[tuple[str, int], list] = {}
    for ch in sorted(chapters_seen):
        for lang in ("de", "en", "zh"):
            segs_cache[(lang, ch)] = load_segments_chapter(lang, ch, args.segmented_dir)

    # ----- iterate rows, building output rows -----
    output_rows: list[list[str]] = []
    for idx, r in enumerate(rows):
        try:
            ch = int(r["chapter"])
            de_sid = r["de_sentence_id"]

            # 1) DE id must be canonical before we look anything up.
            assert_de_sentence_id_wellformed(de_sid, row=idx)

            # 2) Decode en/zh sentence_ids lists.
            en_ids = parse_sentence_ids(
                r.get("en_sentence_ids", ""), row=idx, field="en_sentence_ids"
            )
            zh_ids = parse_sentence_ids(
                r.get("zh_sentence_ids", ""), row=idx, field="zh_sentence_ids"
            )
            de_ids = [de_sid]

            # 3) All referenced ids must resolve in the corresponding
            # (lang, chapter) JSONL.
            de_segs = segs_cache.get(("de", ch), [])
            en_segs = segs_cache.get(("en", ch), [])
            zh_segs = segs_cache.get(("zh", ch), [])

            assert_ids_resolved(de_ids, de_segs, row=idx, lang="de", chapter=ch)
            assert_ids_resolved(en_ids, en_segs, row=idx, lang="en", chapter=ch)
            assert_ids_resolved(zh_ids, zh_segs, row=idx, lang="zh", chapter=ch)

            # 4) Build context triples. The "curr" cell text is recomputed
            # from the segmented JSONL (not the input TSV's aligned_text
            # cell) so multi-sentence alignments get the canonical
            # " / " join and so the builder is the single source of truth
            # for the text the annotator sees.
            de_prev, de_curr, de_next = context_around(de_segs, de_ids, args.context_size)
            en_prev, en_curr, en_next = context_around(en_segs, en_ids, args.context_size)
            zh_prev, zh_curr, zh_next = context_around(zh_segs, zh_ids, args.context_size)

            # DE text gets the «» markers so the annotator can see which
            # sentences are current when DE is multi-sentence.
            de_text_cell = f"«{de_curr}»" if de_curr else ""

            output_rows.append([
                r["datapoint_id"],
                r["chapter"],
                r["de_form"],
                de_sid,
                f"{r['de_token_start']}-{r['de_token_end']}",
                de_text_cell,
                de_prev,
                de_next,
                r.get("en_sentence_ids", ""),
                en_curr,
                r["en_alignment_cardinality"],
                r["en_alignment_confidence"],
                en_prev,
                en_next,
                r.get("zh_sentence_ids", ""),
                zh_curr,
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
        except ContextPackError as e:
            return _die(e.rule, e.row, detail=e.detail or "")

    # ----- write output -----
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(list(OUTPUT_COLUMNS))
        for row in output_rows:
            w.writerow(row)

    # ----- aggregate stdout (no novel text) -----
    by_chapter: dict[int, int] = defaultdict(int)
    by_form: dict[str, int] = defaultdict(int)
    for r in rows:
        by_chapter[int(r["chapter"])] += 1
        by_form[r["de_form"]] += 1

    print(f"rows: {len(rows)}")
    # Emit chapters in ascending order for deterministic stdout.
    sorted_chapters = {str(k): by_chapter[k] for k in sorted(by_chapter)}
    print(f"by_chapter: {sorted_chapters}")
    print(f"by_form: {dict(by_form)}")
    print(f"output: {args.output}")
    print(
        f"  ({len(OUTPUT_COLUMNS)} columns; "
        f"{N_BLANK_HUMAN_CHECK_COLUMNS} blank human-check columns at end)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
