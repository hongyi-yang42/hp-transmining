"""Build the full-novel (Ch.1-17) annotation master TSV.

Generalises ``scripts/build_ch1_3_full_annotation.py`` past Ch.3. The
Ch.1-3 builder is methodologically pinned to chapters [1, 2, 3]; this
script takes its chapter list from the extraction manifest instead, so
it covers the full novel the moment every requested chapter has been
extracted AND aligned.

What it emits — same shape and discipline as the Ch.1-3 master:

  * ``full_novel_annotation_master.tsv`` — every inventory-eligible
    German PP occurrence for the requested chapters, joined to its EN/ZH
    aligned context. Source columns are machine-written and hashed
    (``source_row_sha256``); every human-editable column is blank except
    the builder defaults (``{en,zh}_alignment_qc = assumed_ok``). No
    German decisions, no EN/ZH forms or spans, no gold labels.
  * ``full_novel_annotation_master.summary.json`` — aggregate counts.

Fail-closed rules (the manifest is the operational record of what ran):

  * Missing manifest file → ``MISSING_MANIFEST``.
  * A requested (chapter, form) with no manifest entry, or an entry whose
    status is not ``ok`` / ``zero_hits_ok`` (e.g. ``missing_input``,
    ``empty_input``, ``extraction_error``) → ``EXTRACTION_NOT_READY``.
    No master is written; stale or partial extraction dirs must not
    silently produce a partial-corpus master.
  * A ``zero_hits_ok`` (chapter, form) pair may have a header-only TSV;
    the pair is passed to ``build_candidates`` so the header-only file is
    accepted. Zero-byte files still fail.
  * Missing / empty segmented or alignment inputs → ``MissingInputsError``
    from ``hp_corpus.step4`` (re-reported as ``MISSING_INPUTS``).

Stdout carries aggregate counts only — never sentence text, lemmas,
surface forms, segment IDs, or datapoint IDs.

Usage::

    uv run python scripts/build_full_novel_annotation.py \\
        [--extraction-dir data/extracted/full_novel] \\
        [--manifest data/extracted/full_novel/manifest.json] \\
        [--segmented-dir data/segmented] \\
        [--aligned-dir data/aligned] \\
        [--output-dir data/derived/step4] \\
        [--chapters 1 ... 17] [--force-output]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Make the src package importable when this script is run directly via
# ``uv run python scripts/...``.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hp_corpus.step4 import (  # noqa: E402
    MissingInputsError,
    build_candidates,
    select_ch1_3_annotation_pool,
    write_pilot_tsv,
    write_summary_json,
)

EXIT_OK = 0
EXIT_INPUT_ERROR = 2

MIN_CHAPTER = 1
MAX_CHAPTER = 17
OK_STATUSES = frozenset({"ok", "zero_hits_ok"})

ANNOTATION_MASTER_SCOPE = "full_novel_annotation_master"

MASTER_TSV_NAME = "full_novel_annotation_master.tsv"
MASTER_SUMMARY_NAME = "full_novel_annotation_master.summary.json"


def _fail(rule: str, message: str) -> int:
    print(f"FAIL: rule={rule} {message}", file=sys.stderr)
    return EXIT_INPUT_ERROR


def load_manifest_statuses(
    manifest_path: Path,
) -> dict[tuple[int, str], str]:
    """Return {(chapter, form): status} from the extraction manifest."""
    import json

    with open(manifest_path, encoding="utf-8") as f:
        entries = json.load(f)
    return {(int(e["chapter"]), str(e["form"])): str(e["status"]) for e in entries}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extraction-dir", type=Path, default=Path("data/extracted/full_novel"))
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Extraction manifest JSON (default: <extraction-dir>/manifest.json).",
    )
    ap.add_argument("--segmented-dir", type=Path, default=Path("data/segmented"))
    ap.add_argument("--aligned-dir", type=Path, default=Path("data/aligned"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/derived/step4"))
    ap.add_argument(
        "--chapters",
        type=int,
        nargs="+",
        default=list(range(MIN_CHAPTER, MAX_CHAPTER + 1)),
    )
    ap.add_argument("--force-output", action="store_true")
    args = ap.parse_args(argv)

    chapters = sorted(set(args.chapters))
    if any(ch not in range(MIN_CHAPTER, MAX_CHAPTER + 1) for ch in chapters):
        return _fail("CHAPTER_OUT_OF_RANGE", f"chapters must be in {MIN_CHAPTER}..{MAX_CHAPTER}")

    manifest_path = args.manifest or (args.extraction_dir / "manifest.json")
    if not manifest_path.exists():
        return _fail("MISSING_MANIFEST", str(manifest_path))
    statuses = load_manifest_statuses(manifest_path)

    # --- Manifest gate -----------------------------------------------------
    not_ready: list[tuple[int, str, str]] = []
    zero_hits_ok: set[tuple[int, str]] = set()
    for ch in chapters:
        for form in ("contracted", "uncontracted"):
            status = statuses.get((ch, form))
            if status is None:
                not_ready.append((ch, form, "no_manifest_entry"))
            elif status not in OK_STATUSES:
                not_ready.append((ch, form, status))
            elif status == "zero_hits_ok":
                zero_hits_ok.add((ch, form))
    if not_ready:
        by_rule: Counter[str] = Counter(rule for _, _, rule in not_ready)
        print(
            f"FAIL: rule=EXTRACTION_NOT_READY {len(not_ready)} (chapter, form) pairs "
            f"not extractable: {dict(by_rule)}",
            file=sys.stderr,
        )
        for ch, form, rule in not_ready[:10]:
            print(f"  ch{ch:02d} {form}: {rule}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    # --- Output guard ------------------------------------------------------
    out_tsv = args.output_dir / MASTER_TSV_NAME
    out_summary = args.output_dir / MASTER_SUMMARY_NAME
    if out_tsv.exists() and not args.force_output:
        return _fail("OUTPUT_EXISTS", f"{out_tsv}; pass --force-output to overwrite")

    # --- Build -------------------------------------------------------------
    try:
        candidates = build_candidates(
            extraction_dir=args.extraction_dir,
            segmented_dir=args.segmented_dir,
            aligned_dir=args.aligned_dir,
            chapters=chapters,
            zero_hits_ok=frozenset(zero_hits_ok),
        )
    except MissingInputsError as exc:
        return _fail("MISSING_INPUTS", f"{len(exc.missing)} input file(s) {exc.kind} or malformed")
    except Exception as exc:  # noqa: BLE001 — fail closed on any builder error
        return _fail("BUILD_FAILED", type(exc).__name__)

    pool, pool_summary = select_ch1_3_annotation_pool(candidates)
    write_pilot_tsv(pool, out_tsv, scope_override=ANNOTATION_MASTER_SCOPE)

    by_status = Counter(
        statuses[(ch, form)] for ch in chapters for form in ("contracted", "uncontracted")
    )
    summary = {
        "dataset_scope": ANNOTATION_MASTER_SCOPE,
        "chapters": chapters,
        "manifest_by_status": dict(sorted(by_status.items())),
        "zero_hits_ok_pairs": sorted(f"ch{ch:02d}_{form}" for ch, form in zero_hits_ok),
        "candidate_total": pool_summary["candidate_total"],
        "pool_total": pool_summary["pool_total"],
        "ineligible_total": pool_summary["ineligible_total"],
        "ineligible_by_form": pool_summary["dropped_by_form"],
        "by_form_in_pool": pool_summary["by_form"],
        "by_chapter_in_pool": pool_summary["by_chapter"],
        "shared_prepositions": pool_summary["shared_prepositions"],
        "minimal_pair_groups_in_pool": pool_summary["minimal_pair_groups_in_sample"],
        "minimal_pair_groups_with_both_forms": pool_summary["minimal_pair_groups_with_both_forms"],
        "note": (
            "Full-novel annotation master: every inventory-eligible occurrence "
            "with blank editable columns. German review (de_candidate_decision) "
            "and EN/ZH annotation are downstream human work; the sampling "
            "ledger (build_full_novel_sampling_ledger.py) joins back to this "
            "file by datapoint_id."
        ),
    }
    write_summary_json(summary, out_summary)

    # --- stdout: aggregate counts only --------------------------------------
    print(f"chapters: {chapters}")
    print(f"manifest by_status: {dict(sorted(by_status.items()))}")
    print(f"zero_hits_ok (chapter, form) pairs: {len(zero_hits_ok)}")
    print(f"candidates total: {pool_summary['candidate_total']}")
    print(
        f"  ineligible (canonical prep not in both inventories): "
        f"{pool_summary['ineligible_total']} {pool_summary['dropped_by_form']}"
    )
    print(f"master pool (full-novel annotation target): {pool_summary['pool_total']}")
    print(f"  contracted:   {pool_summary['by_form']['contracted']}")
    print(f"  uncontracted: {pool_summary['by_form']['uncontracted']}")
    print(f"  by chapter: {pool_summary['by_chapter']}")
    print("outputs:")
    print(f"  {out_tsv}")
    print(f"  {out_summary}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
