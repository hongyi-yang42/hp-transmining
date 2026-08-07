"""Build the Step 4 cross-lingual annotation pack for Ch.1–3.

Reads the German PP extraction TSVs (with occurrence coordinates) plus
the three-language segmented JSONL and DE↔EN / DE↔ZH alignment JSONL,
emits the candidate JSONL and a 10+10 pilot TSV for human annotation.

Stdout prints aggregate counts only — no surface forms, lemmas, segment
IDs, or sentence text from the source novels. Detailed output goes to
``data/derived/step4/`` which is gitignored.

Usage:
    uv run python scripts/build_step4_annotation_pack.py \
        [--extraction-dir data/extracted] \
        [--segmented-dir data/segmented] \
        [--aligned-dir data/aligned] \
        [--output-dir data/derived/step4] \
        [--chapters 1 2 3] \
        [--n-contracted 10] \
        [--n-uncontracted 10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hp_corpus.step4 import (
    InsufficientCandidatesError,
    build_candidates,
    select_pilot,
    summarize_candidates,
    write_candidates_jsonl,
    write_pilot_tsv,
    write_summary_json,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extraction-dir", type=Path, default=Path("data/extracted"))
    ap.add_argument("--segmented-dir", type=Path, default=Path("data/segmented"))
    ap.add_argument("--aligned-dir", type=Path, default=Path("data/aligned"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/derived/step4"))
    ap.add_argument("--chapters", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n-contracted", type=int, default=10)
    ap.add_argument("--n-uncontracted", type=int, default=10)
    args = ap.parse_args(argv)

    candidates = build_candidates(
        extraction_dir=args.extraction_dir,
        segmented_dir=args.segmented_dir,
        aligned_dir=args.aligned_dir,
        chapters=args.chapters,
    )
    candidates_jsonl = args.output_dir / "ch1_3_all_candidates.jsonl"

    try:
        selected, pilot_summary = select_pilot(
            candidates,
            n_contracted=args.n_contracted,
            n_uncontracted=args.n_uncontracted,
        )
    except InsufficientCandidatesError as e:
        # select_pilot mutates candidates in-place (pilot_selected=True on
        # chosen items) only on success; on failure, no item is flipped.
        # Write candidates reflecting the unselected state, plus a summary.
        write_candidates_jsonl(candidates, candidates_jsonl)
        full_summary = summarize_candidates(candidates, [], e.summary)
        write_summary_json(full_summary, args.output_dir / "ch1_3_pilot_20.summary.json")
        print(f"INSUFFICIENT CANDIDATES for pilot: {e}", file=sys.stderr)
        print(
            f"  wrote {candidates_jsonl} ({len(candidates)} candidates) "
            f"and summary; pilot TSV not written.",
            file=sys.stderr,
        )
        return 2

    # On success: write candidates AFTER select_pilot so the JSONL reflects
    # which 20 items are pilot_selected=True.
    write_candidates_jsonl(candidates, candidates_jsonl)

    # Selected arrives in selection order; the TSV is more useful to a human
    # annotator when grouped by chapter and ordered by sentence, so re-sort.
    selected_sorted = sorted(
        selected,
        key=lambda c: (int(c["chapter"]), str(c["de_sentence_id"]), int(c["de_token_start"])),
    )
    pilot_tsv = args.output_dir / "ch1_3_pilot_20.tsv"
    write_pilot_tsv(selected_sorted, pilot_tsv)

    full_summary = summarize_candidates(candidates, selected_sorted, pilot_summary)
    summary_path = args.output_dir / "ch1_3_pilot_20.summary.json"
    write_summary_json(full_summary, summary_path)

    # ----- stdout: aggregate counts only -----
    print(f"candidates: {full_summary['candidate_total']}")
    print(f"  contracted:   {full_summary['by_form']['contracted']}")
    print(f"  uncontracted: {full_summary['by_form']['uncontracted']}")
    print(f"  author_resource_match=True:  {full_summary['by_author_match']['true']}")
    print(f"  author_resource_match=False: {full_summary['by_author_match']['false']}")
    print(
        f"  minimal-pair groups: {full_summary['minimal_pair_group_count']} "
        f"({full_summary['minimal_pair_groups_with_both_forms']} with both forms)"
    )
    print(
        f"  missing alignment: en={full_summary['missing_alignment']['en']}, "
        f"zh={full_summary['missing_alignment']['zh']}, "
        f"either={full_summary['missing_alignment']['either']}"
    )
    multi = full_summary["multi_sentence_alignment"]
    print(
        f"  multi-sentence alignment: en_1_to_n={multi['en_1_to_n']}, "
        f"zh_1_to_n={multi['zh_1_to_n']}"
    )
    print(f"pilot selected: {full_summary['pilot_selected_total']}")
    if pilot_summary:
        print(f"  by reason: {pilot_summary['by_reason']}")
    print("outputs:")
    print(f"  {candidates_jsonl}")
    print(f"  {pilot_tsv}")
    print(f"  {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
