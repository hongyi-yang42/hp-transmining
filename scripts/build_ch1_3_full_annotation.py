"""Build the full Ch.1-3 Bremmers-sample annotation TSV.

Applies the paper's preposition-inventory filter (Bremmers et al. 2022,
§2.2.1) to every extracted German PP from Ch.1-3 and writes the full
selected set as a human-annotation TSV. The 10+10 method pilot produced
by ``build_step4_annotation_pack.py`` is a strict subset of this file.

This script is additive to ``build_step4_annotation_pack.py``: that
script emits ``all_candidates.jsonl`` + the 20-row pilot TSV; this one
emits the full annotation target. The two coexist — annotate the full
TSV going forward, keep the pilot TSV for method-record purposes.

Stdout carries aggregate counts only — no surface forms, lemmas,
segment IDs, or sentence text. Detailed output goes to
``data/derived/step4/`` which is gitignored.

Usage:
    uv run python scripts/build_ch1_3_full_annotation.py \\
        [--extraction-dir data/extracted] \\
        [--segmented-dir data/segmented] \\
        [--aligned-dir data/aligned] \\
        [--output-dir data/derived/step4] \\
        [--chapters 1 2 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hp_corpus.step4 import (
    PAPER_TABLE_A,
    build_candidates,
    select_paper_sample,
    summarize_candidates,
    write_pilot_tsv,
    write_summary_json,
)

# Scope label written into every row of the full TSV. ``DATASET_SCOPE``
# from step4.py is "ch1_3_method_pilot"; we override it here because the
# full TSV is no longer a pilot. The override is applied inside
# ``write_pilot_tsv`` (alongside the existing source_row_sha256 mutation
# pattern) so the script does not reach into candidate dicts directly.
FULL_SAMPLE_SCOPE = "ch1_3_paper_sample"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extraction-dir", type=Path, default=Path("data/extracted"))
    ap.add_argument("--segmented-dir", type=Path, default=Path("data/segmented"))
    ap.add_argument("--aligned-dir", type=Path, default=Path("data/aligned"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/derived/step4"))
    ap.add_argument("--chapters", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args(argv)

    candidates = build_candidates(
        extraction_dir=args.extraction_dir,
        segmented_dir=args.segmented_dir,
        aligned_dir=args.aligned_dir,
        chapters=args.chapters,
    )
    selected, sample_summary = select_paper_sample(candidates)

    # ``selected`` arrives in build_candidates' stable-sort order
    # (chapter, sentence_id, token_start, token_end); select_paper_sample
    # preserves that order, so no re-sort is needed.
    out_tsv = args.output_dir / "ch1_3_full_annotation.tsv"
    out_summary = args.output_dir / "ch1_3_full_annotation.summary.json"
    write_pilot_tsv(selected, out_tsv, scope_override=FULL_SAMPLE_SCOPE)

    # Reuse summarize_candidates for the candidate-level stats (missing
    # alignment, multi-sentence alignment, minimal-pair groups), then
    # attach the paper-sample view and the Table A comparison.
    cand_summary = summarize_candidates(candidates, None, None)
    n_c = sample_summary["by_form"]["contracted"]
    n_u = sample_summary["by_form"]["uncontracted"]
    total = sample_summary["selected_total"]
    summary = {
        "dataset_scope": FULL_SAMPLE_SCOPE,
        "candidate_total": sample_summary["candidate_total"],
        "selected_total": total,
        "dropped_total": sample_summary["dropped_total"],
        "dropped_by_form": sample_summary["dropped_by_form"],
        "by_form_selected": sample_summary["by_form"],
        "by_chapter_selected": sample_summary["by_chapter"],
        "shared_prepositions": sample_summary["shared_prepositions"],
        "minimal_pair_groups_in_sample": sample_summary["minimal_pair_groups_in_sample"],
        "minimal_pair_groups_with_both_forms": (
            sample_summary["minimal_pair_groups_with_both_forms"]
        ),
        "alignment_quality": {
            "missing_alignment": cand_summary["missing_alignment"],
            "multi_sentence_alignment": cand_summary["multi_sentence_alignment"],
        },
        "table_a": {
            "ours_ch1_3_subset": {
                "contracted": n_c,
                "uncontracted": n_u,
                "total": total,
                "contracted_pct": round(100 * n_c / total, 1) if total else 0.0,
                "uncontracted_pct": round(100 * n_u / total, 1) if total else 0.0,
            },
            "paper_full_novel": PAPER_TABLE_A,
            "note": (
                "Ch.1-3 is a strict subset of the paper's full-novel sample; "
                "the count difference reflects chapters outside our current "
                "scope (Ch.4-17), not methodology drift."
            ),
        },
    }
    write_summary_json(summary, out_summary)

    # ----- stdout: aggregate counts only -----
    print(f"candidates total: {sample_summary['candidate_total']}")
    print(
        f"  dropped (canonical prep not in both inventories): "
        f"{sample_summary['dropped_total']} {sample_summary['dropped_by_form']}"
    )
    print(f"selected (Ch.1-3 paper sample): {total}")
    print(f"  contracted:   {n_c}")
    print(f"  uncontracted: {n_u}")
    print(f"  by chapter: {sample_summary['by_chapter']}")
    print(
        f"  minimal-pair groups in sample: "
        f"{sample_summary['minimal_pair_groups_in_sample']} "
        f"({sample_summary['minimal_pair_groups_with_both_forms']} with both forms)"
    )
    print("Table A (German form distribution, % of Ch.1-3 subset):")
    pct_c = round(100 * n_c / total, 1) if total else 0.0
    pct_u = round(100 * n_u / total, 1) if total else 0.0
    print(f"  contracted:   {n_c} ({pct_c}%)")
    print(f"  uncontracted: {n_u} ({pct_u}%)")
    print("Paper full-novel Table A (for comparison):")
    print(
        f"  contracted:   {PAPER_TABLE_A['contracted']} "
        f"({PAPER_TABLE_A['contracted_pct']}%)"
    )
    print(
        f"  uncontracted: {PAPER_TABLE_A['uncontracted']} "
        f"({PAPER_TABLE_A['uncontracted_pct']}%)"
    )
    print("outputs:")
    print(f"  {out_tsv}")
    print(f"  {out_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
