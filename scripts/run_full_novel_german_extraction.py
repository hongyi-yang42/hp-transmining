"""Full-novel German PP extraction with chapter manifests.

SUPERADES ``scripts/run_paper_extractor.py`` for full-novel (Ch.1–17)
extraction. The older script stays in the tree for ongoing Ch.1–3 work,
but it has two full-novel blockers that make it unsafe past Ch.9:

1. **Filename bug** (``run_paper_extractor.py:202``): the format string
   ``f"hp1_de_ch0{ch}_nomwt.conllu"`` produces ``hp1_de_ch010_nomwt.conllu``
   for ``ch >= 10`` instead of ``hp1_de_ch10_nomwt.conllu``. Any chapter
   ≥ 10 silently fails to be found.
2. **Silent skip of missing inputs** (``run_paper_extractor.py:204``):
   a missing .conllu file is logged to stderr then ``continue``'d, so
   an upstream pipeline failure (forgotten parse step, partial run)
   produces an empty TSV without any error.

This script fixes both:

* All chapter paths use ``f"hp1_de_ch{ch:02d}_..."`` zero-padding.
* Missing inputs raise ``MissingParsedInputError`` by default; pass
  ``--allow-missing`` to record ``status="missing_input"`` in the
  manifest and continue. Empty / corrupt inputs always fail closed.

The extraction ALGORITHM is identical to ``run_paper_extractor.extract``
(see :func:`hp_corpus.german_extraction.extract_chapter`). What's new is
the I/O wrapper, the per-chapter manifest, and the fail-closed rules.

The manifest is a JSON array with one entry per (chapter, form) pair.
Each entry captures the parsed-file SHA-256, size, sentence count, the
extractor version, and the extraction status so downstream consumers
can prove which input produced which TSV.

Stdout carries aggregate counts only — chapter number, hit count,
filter-match count, and the output filename. No noun lemmas, surface
forms, sentence IDs, or preposition forms are ever printed.

Usage::

    uv run python scripts/run_full_novel_german_extraction.py \\
        --chapters 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 \\
        [--parsed-dir data/parsed] \\
        [--output-dir data/extracted/full_novel] \\
        [--manifest <path>] \\
        [--allow-missing]

Vendor prerequisite: clone ``time-in-translation/conll-extractor`` into
``vendor/conll-extractor/`` before running on real data. Unit tests
don't need the vendor.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Make the src package importable when this script is run directly via
# ``uv run python scripts/...``. The CLI subcommands in hp_corpus.cli
# already do this via the installed entry point; standalone scripts that
# import from hp_corpus need to wire it themselves.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Inject the vendored conll-extractor into sys.path so the filter-list
# import inside hp_corpus.german_extraction succeeds in production.
_VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "conll-extractor"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from hp_corpus.german_extraction import (  # noqa: E402
    EXTRACTOR_VERSION,
    RULE_CHAPTER_RANGE,
    RULE_EMPTY,
    RULE_FAILED,
    RULE_MISSING,
    EmptyParsedInputError,
    ExtractionFailedError,
    GermanExtractionError,
    MissingParsedInputError,
    chapter_manifest,
    extract_chapter,
    validate_against_filters,
    write_chapter_tsv,
    write_manifest_json,
)

MIN_CHAPTER = 1
MAX_CHAPTER = 17


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Full-novel German PP extraction (Ch.1-17) with manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--chapters",
        type=int,
        nargs="+",
        required=True,
        help=f"Chapter numbers to process (each in {MIN_CHAPTER}..{MAX_CHAPTER}).",
    )
    ap.add_argument(
        "--parsed-dir",
        type=Path,
        default=Path("data/parsed"),
        help="Directory containing hp1_de_chNN_nomwt.conllu files.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/extracted/full_novel"),
        help="Directory to write per-chapter TSV outputs.",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest JSON path. Defaults to <output-dir>/manifest.json.",
    )
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Skip chapters whose .conllu input is missing instead of "
            "exiting non-zero. The missing chapter still gets a "
            "manifest entry with status='missing_input'. Empty/corrupt "
            "inputs always fail closed."
        ),
    )
    return ap


def _exit_with(rule: str, message: str) -> int:
    """Print a one-line rule message to stderr and return exit code 1.

    The message never includes the chapter's path contents — only the
    rule name and chapter number, to preserve the stdout/stderr privacy
    guarantee.
    """
    print(message, file=sys.stderr)
    print(f"rule: {rule}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    # --- Validate chapter range -------------------------------------------------
    out_of_range = [c for c in args.chapters if not (MIN_CHAPTER <= c <= MAX_CHAPTER)]
    if out_of_range:
        print(
            f"chapters out of range [{MIN_CHAPTER}..{MAX_CHAPTER}]: {out_of_range}",
            file=sys.stderr,
        )
        print(f"rule: {RULE_CHAPTER_RANGE}", file=sys.stderr)
        return 2

    parsed_dir: Path = args.parsed_dir
    output_dir: Path = args.output_dir
    manifest_path: Path = args.manifest or (output_dir / "manifest.json")
    allow_missing: bool = args.allow_missing

    manifest: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-chapter flow -------------------------------------------------------
    for chapter in args.chapters:
        parsed_path = parsed_dir / f"hp1_de_ch{chapter:02d}_nomwt.conllu"

        for contracted in (True, False):
            kind = "contracted" if contracted else "uncontracted"
            filt_name = (
                "FILTER_CONTRACTED_123" if contracted else "FILTER_PP"
            )
            tsv_name = f"hp1_de_ch{chapter:02d}_{kind}.tsv"
            tsv_path = output_dir / tsv_name

            try:
                hits = extract_chapter(parsed_path, contracted)
            except MissingParsedInputError:
                if allow_missing:
                    print(f"Ch.{chapter}: {RULE_MISSING}")
                    manifest.append(
                        chapter_manifest(
                            chapter=chapter,
                            contracted=contracted,
                            parsed_path=parsed_path,
                            hits=None,
                            status="missing_input",
                            error=RULE_MISSING,
                        )
                    )
                    continue
                print(f"Ch.{chapter}: {RULE_MISSING}", file=sys.stderr)
                print(f"rule: {RULE_MISSING}", file=sys.stderr)
                return 1
            except EmptyParsedInputError:
                print(f"Ch.{chapter}: {RULE_EMPTY}", file=sys.stderr)
                print(f"rule: {RULE_EMPTY}", file=sys.stderr)
                return 1
            except ExtractionFailedError as exc:
                print(f"Ch.{chapter}: {RULE_FAILED}", file=sys.stderr)
                print(f"rule: {RULE_FAILED} ({exc})", file=sys.stderr)
                return 1
            except GermanExtractionError as exc:
                # Defensive: any other typed error from this module.
                print(f"Ch.{chapter}: {RULE_FAILED}", file=sys.stderr)
                print(f"rule: {RULE_FAILED} ({exc})", file=sys.stderr)
                return 1

            n_match, n_total, matched = validate_against_filters(hits, contracted)
            write_chapter_tsv(tsv_path, hits, matched)

            status = "ok" if hits else "zero_hits_ok"
            manifest.append(
                chapter_manifest(
                    chapter=chapter,
                    contracted=contracted,
                    parsed_path=parsed_path,
                    hits=hits,
                    status=status,
                )
            )

            print(
                f"Ch.{chapter} {kind}: {n_total} extracted, "
                f"{n_match} match {filt_name} → {tsv_name}"
            )

    # --- Write manifest + summary ----------------------------------------------
    write_manifest_json(manifest_path, manifest)

    by_status: Counter[str] = Counter(m["status"] for m in manifest)
    print(f"manifest: {manifest_path}")
    print(f"total_chapter_entries: {len(manifest)}")
    status_dict = {k: by_status.get(k, 0) for k in sorted(by_status)}
    print(f"by_status: {status_dict}")
    print(f"extractor_version: {EXTRACTOR_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
