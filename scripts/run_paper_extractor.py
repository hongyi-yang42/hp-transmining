"""Run the paper's PP extractor on DE CoNLL-U files for Ch.1-3.

Uses time-in-translation/conll-extractor's logic verbatim:
- contracted=True → finds tokens in CONTRACTED list, extracts (prep, noun)
- contracted=False → finds PREPOSITIONS tokens with same-head ART in
  DETERMINERS (das/dem/der), extracts (prep, det, noun)

Then validates against FILTER_CONTRACTED_123 / FILTER_PP (the Ch.1-3
subset the paper annotated) to see how many of our hits match the paper's
data. Outputs per-chapter TSVs + prints combined stats.

Stdout carries aggregate counts only — never noun lemmas, forms, or any
other token-level data from the source text. Per-PP detail goes only to
the TSV files under --output-dir (gitignored).

Usage:
    uv run python scripts/run_paper_extractor.py [--chapters 1 2 3] \
        [--parsed-dir data/parsed] [--output-dir data/extracted]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pyconll

sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "conll-extractor"))
from conll_extractor.prepositions.data import (  # type: ignore
    CONTRACTED,
    DETERMINERS,
    FILTER_CONTRACTED_123,
    FILTER_PP,
    PREPOSITIONS,
)

POS_ARTICLE = "ART"
POS_NOUN = "NN"


def extract(in_file: Path, contracted: bool) -> list[dict]:
    """Adapted from conll_extractor.prepositions.extract.process_single."""
    out = []
    sentences = pyconll.load_from_file(str(in_file))
    forms = CONTRACTED if contracted else PREPOSITIONS
    needs_determiner = not contracted

    for sentence in sentences:
        current_head = None
        current_det = None
        current_token = None

        for token in sentence:
            if token.form is None:
                continue
            if token.form in forms:
                current_token = token.form
                current_head = token.head

            if needs_determiner and current_head is not None:
                if (
                    token.head == current_head
                    and token.xpos == POS_ARTICLE
                    and token.form in DETERMINERS
                ):
                    current_det = token.form

            current_det_filled = current_det is not None or not needs_determiner
            current_head_filled = (
                current_head is not None and token.id == current_head and token.xpos == POS_NOUN
            )

            if current_head_filled and current_det_filled:
                out.append(
                    {
                        "prep": current_token,
                        "det": current_det,
                        "noun": token.lemma,
                        "sentence_id": sentence.id,
                    }
                )
                current_head = None
                current_det = None
                current_token = None
    return out


def validate(hits: list[dict], contracted: bool) -> tuple[int, int, list[dict]]:
    """Return (matched_count, total_count, matched_hits)."""
    filt = FILTER_CONTRACTED_123 if contracted else FILTER_PP
    matched = [h for h in hits if h["prep"] in filt and h["noun"] in filt[h["prep"]]]
    return len(matched), len(hits), matched


def write_tsv(path: Path, hits: list[dict], matched: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["sentence_id", "prep", "det", "noun", "in_filter"])
        for h in hits:
            in_filter = "Y" if h in matched else "N"
            w.writerow([h["sentence_id"], h["prep"], h["det"] or "-", h["noun"], in_filter])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chapters", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--parsed-dir", type=Path, default=Path("data/parsed"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/extracted"))
    args = ap.parse_args(argv)

    parsed_dir = args.parsed_dir
    out_dir = args.output_dir

    total_contracted: list[dict] = []
    total_uncontracted: list[dict] = []
    total_contracted_matched: list[dict] = []
    total_uncontracted_matched: list[dict] = []

    for ch in args.chapters:
        in_file = parsed_dir / f"hp1_de_ch0{ch}_nomwt.conllu"
        if not in_file.exists():
            print(f"SKIP Ch.{ch}: {in_file} not found (run normalize_conllu_mwt.py first)")
            continue

        print(f"\n=== Chapter {ch} ({in_file.name}) ===")
        for contracted in (True, False):
            kind = "contracted" if contracted else "uncontracted"
            hits = extract(in_file, contracted)
            n_match, n_total, matched = validate(hits, contracted)
            out_path = out_dir / f"hp1_de_ch0{ch}_{kind}.tsv"
            write_tsv(out_path, hits, matched)
            filt_name = "FILTER_CONTRACTED_123" if contracted else "FILTER_PP"
            print(f"  {kind}: {n_total} extracted, {n_match} match {filt_name} → {out_path.name}")

            if contracted:
                total_contracted.extend(hits)
                total_contracted_matched.extend(matched)
            else:
                total_uncontracted.extend(hits)
                total_uncontracted_matched.extend(matched)

    # Combined summary
    c_pct = 100 * len(total_contracted_matched) / max(1, len(total_contracted))
    u_pct = 100 * len(total_uncontracted_matched) / max(1, len(total_uncontracted))
    print("\n=== COMBINED Ch.1-3 ===")
    print(
        f"  contracted:   {len(total_contracted_matched):3d} / {len(total_contracted):3d} "
        f"match FILTER_CONTRACTED_123 ({c_pct:.0f}%)"
    )
    print(
        f"  uncontracted: {len(total_uncontracted_matched):3d} / {len(total_uncontracted):3d} "
        f"match FILTER_PP ({u_pct:.0f}%)"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
