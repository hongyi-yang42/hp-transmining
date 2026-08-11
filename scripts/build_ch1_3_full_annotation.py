"""Build the Ch.1–3 annotation-target TSV.

Applies the paper's preposition-inventory filter (Bremmers et al. 2022,
§2.2.1) to every extracted German PP from Ch.1–3 and writes the surviving
pool as a human-annotation TSV. The 10+10 method pilot produced by
``build_step4_annotation_pack.py`` is a strict subset of this file.

This script is additive to ``build_step4_annotation_pack.py``: that
script emits ``all_candidates.jsonl`` + the 20-row pilot TSV; this one
emits the annotation target. The two coexist — annotate the target TSV
going forward, keep the pilot TSV for method-record purposes.

**The output is a Ch.1–3 paper-eligible annotation pool, not the paper's
final 96 trilingual contexts.** The 96 are hand-selected from the full
novel; this pool is every Ch.1–3 occurrence whose canonical preposition
is in the paper's 13-item paired inventory.

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
    select_ch1_3_annotation_pool,
    summarize_candidates,
    write_pilot_tsv,
    write_summary_json,
)

# Scope label written into every row of the annotation-target TSV.
# This is the operational pool label; ``paper_final_sample`` on every row
# stays False (the pool is not the paper's hand-picked 96).
ANNOTATION_TARGET_SCOPE = "ch1_3_annotation_target"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extraction-dir", type=Path, default=Path("data/extracted"))
    ap.add_argument("--segmented-dir", type=Path, default=Path("data/segmented"))
    ap.add_argument("--aligned-dir", type=Path, default=Path("data/aligned"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/derived/step4"))
    ap.add_argument("--chapters", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args(argv)

    if sorted(args.chapters) != [1, 2, 3]:
        ap.error(
            "this builder is methodologically restricted to Ch.1–3; "
            f"got --chapters {' '.join(str(c) for c in args.chapters)!r}"
        )

    candidates = build_candidates(
        extraction_dir=args.extraction_dir,
        segmented_dir=args.segmented_dir,
        aligned_dir=args.aligned_dir,
        chapters=args.chapters,
    )
    pool, pool_summary = select_ch1_3_annotation_pool(candidates)

    # ``pool`` arrives in build_candidates' stable-sort order
    # (chapter, sentence_id, token_start, token_end); the selector preserves
    # that order, so no re-sort is needed.
    out_tsv = args.output_dir / "ch1_3_full_annotation.tsv"
    out_summary = args.output_dir / "ch1_3_full_annotation.summary.json"
    write_pilot_tsv(pool, out_tsv, scope_override=ANNOTATION_TARGET_SCOPE)

    # Reuse summarize_candidates for the candidate-level stats (missing
    # alignment, multi-sentence alignment, minimal-pair groups), then
    # attach the pool view and the Table A comparison.
    cand_summary = summarize_candidates(candidates, None, None)
    n_c = pool_summary["by_form"]["contracted"]
    n_u = pool_summary["by_form"]["uncontracted"]
    total = pool_summary["pool_total"]
    summary = {
        "dataset_scope": ANNOTATION_TARGET_SCOPE,
        "candidate_total": pool_summary["candidate_total"],
        "pool_total": total,
        "ineligible_total": pool_summary["ineligible_total"],
        "ineligible_by_form": pool_summary["dropped_by_form"],
        "by_form_in_pool": pool_summary["by_form"],
        "by_chapter_in_pool": pool_summary["by_chapter"],
        "shared_prepositions": pool_summary["shared_prepositions"],
        "minimal_pair_groups_in_pool": pool_summary["minimal_pair_groups_in_sample"],
        "minimal_pair_groups_with_both_forms": (
            pool_summary["minimal_pair_groups_with_both_forms"]
        ),
        "alignment_quality": {
            "missing_alignment": cand_summary["missing_alignment"],
            "multi_sentence_alignment": cand_summary["multi_sentence_alignment"],
        },
        "table_a": {
            "ch1_3_pool": {
                "contracted": n_c,
                "uncontracted": n_u,
                "total": total,
                "contracted_pct": round(100 * n_c / total, 1) if total else 0.0,
                "uncontracted_pct": round(100 * n_u / total, 1) if total else 0.0,
            },
            "paper_full_novel": PAPER_TABLE_A,
            "note": (
                "Ch.1–3 paper-eligible annotation pool; not the paper's final "
                "96. The paper's 96 is hand-selected from the full novel; this "
                "pool is every Ch.1–3 occurrence whose canonical preposition is "
                "in the paper's 13-item paired inventory."
            ),
        },
    }
    write_summary_json(summary, out_summary)

    # ----- stdout: aggregate counts only -----
    print(f"candidates total: {pool_summary['candidate_total']}")
    print(
        f"  ineligible (canonical prep not in both inventories): "
        f"{pool_summary['ineligible_total']} {pool_summary['dropped_by_form']}"
    )
    print(f"pool (Ch.1–3 annotation target): {total}")
    print(f"  contracted:   {n_c}")
    print(f"  uncontracted: {n_u}")
    print(f"  by chapter: {pool_summary['by_chapter']}")
    print(
        f"  minimal-pair groups in pool: "
        f"{pool_summary['minimal_pair_groups_in_sample']} "
        f"({pool_summary['minimal_pair_groups_with_both_forms']} with both forms)"
    )
    print("Table A (German form distribution, % of Ch.1–3 pool):")
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
