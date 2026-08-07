"""Run the paper's PP extractor on our DE CoNLL-U.

Uses time-in-translation/conll-extractor's logic verbatim:
- contracted=True → finds tokens in CONTRACTED list, extracts (prep, noun)
- contracted=False → finds PREPOSITIONS tokens with same-head ART in
  DETERMINERS (das/dem/der), extracts (prep, det, noun)

Then validates against FILTER_CONTRACTED_123 (the Ch.1-3 subset the paper
annotated) to see how many of our Ch.1 hits match the paper's data.
"""

from __future__ import annotations

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
    """Adapted from conll_extractor.prepositions.extract.process_single.
    Returns list of dicts instead of writing CSV directly."""
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
    """Return (matched_count, total_count, matched_hits).
    For contracted: match against FILTER_CONTRACTED_123 (Ch.1-3 paper set).
    For uncontracted: match against FILTER_PP."""
    filt = FILTER_CONTRACTED_123 if contracted else FILTER_PP
    matched = [h for h in hits if h["prep"] in filt and h["noun"] in filt[h["prep"]]]
    return len(matched), len(hits), matched


def main() -> int:
    in_file = Path("data/parsed/hp1_de_ch01_nomwt.conllu")
    out_dir = Path("data/extracted")
    out_dir.mkdir(parents=True, exist_ok=True)

    for contracted in (True, False):
        kind = "contracted" if contracted else "uncontracted"
        hits = extract(in_file, contracted)
        n_match, n_total, matched = validate(hits, contracted)
        out_path = out_dir / f"hp1_de_ch01_{kind}.tsv"
        with open(out_path, "w", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["sentence_id", "prep", "det", "noun", "in_filter"])
            for h in hits:
                in_filter = "Y" if h in matched else "N"
                w.writerow([h["sentence_id"], h["prep"], h["det"] or "-", h["noun"], in_filter])
        filt_name = "FILTER_CONTRACTED_123" if contracted else "FILTER_PP"
        print(f"{kind}: extracted {n_total} PPs, {n_match} match {filt_name}")
        print(f"  → {out_path}")
        if contracted:
            # Break down which prepositions
            by_prep = {}
            for h in matched:
                by_prep.setdefault(h["prep"], []).append(h["noun"])
            for p in sorted(by_prep):
                print(f"    {p}: {len(by_prep[p])} hits — {by_prep[p][:5]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
