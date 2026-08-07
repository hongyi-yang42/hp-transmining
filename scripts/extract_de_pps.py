"""Extract German PPs with definite articles for Bremmers et al. (2021) reproduction.

Two flavors of extraction:
  - Contracted: preposition+article fused into one token (im, am, zum, ins, ...).
    These are unambiguous and regex-extractable without a POS tagger.
  - Uncontracted: preposition followed by a separate definite article token
    (in dem, an dem, zu dem, ...). Regex-based; will catch most cases but may
    produce false positives when "der/die/das" is a relative pronoun rather
    than an article. Manual review needed.

Output: TSV with columns
    segment_id, kind, prep, article, pp_text, sentence

Usage:
    uv run python scripts/extract_de_pps.py \\
        --input data/segmented/hp1_de_ch01.jsonl \\
        --output data/extracted/hp1_de_ch01_pps.tsv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Contracted preposition+article forms (Standard German).
# Source: Duden grammar + Schwarz 2009. Each maps to (preposition, article).
CONTRACTED = {
    "am": ("an", "dem"),
    "ans": ("an", "das"),
    "aufs": ("auf", "das"),
    "beim": ("bei", "dem"),
    "durchs": ("durch", "das"),
    "fürs": ("für", "das"),
    "hinterm": ("hinter", "dem"),
    "hinters": ("hinter", "das"),
    "im": ("in", "dem"),
    "ins": ("in", "das"),
    "überm": ("über", "dem"),
    "übers": ("über", "das"),
    "ums": ("um", "das"),
    "unterm": ("unter", "dem"),
    "unters": ("unter", "das"),
    "vom": ("von", "dem"),
    "vorm": ("vor", "dem"),
    "vors": ("vor", "das"),
    "zum": ("zu", "dem"),
    "zur": ("zu", "der"),
}

# Prepositions that can appear in uncontracted PPs with definite articles.
PREPOSITIONS = {
    "an",
    "auf",
    "aus",
    "bei",
    "durch",
    "für",
    "gegen",
    "hinter",
    "in",
    "mit",
    "nach",
    "neben",
    "über",
    "um",
    "unter",
    "von",
    "vor",
    "zu",
    "zwischen",
    "ausser",
    "außer",
    "trotz",
    "während",
    "wegen",
    "innerhalb",
    "oberhalb",
    "unterhalb",
}

# German definite articles (nominative / accusative / dative / genitive).
DEFINITE_ARTICLES = {"der", "die", "das", "dem", "den", "des"}


def extract_contracted(text: str) -> list[tuple[str, str, str]]:
    """Find contracted PPs. Returns [(prep, article, contracted_token), ...]."""
    out = []
    # Word-boundary regex; case-insensitive but typically lowercase in text
    pattern = re.compile(
        r"\b(" + "|".join(CONTRACTED.keys()) + r")(?=\b|(?<=\w)\W)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        token = m.group(1).lower()
        prep, article = CONTRACTED[token]
        out.append((prep, article, m.group(1)))
    return out


def extract_uncontracted(text: str) -> list[tuple[str, str, str]]:
    """Find uncontracted PPs (preposition followed by definite article).

    Returns [(prep, article, matched_text), ...].
    Heuristic — may include relative-pronoun false positives.
    """
    out = []
    tokens = re.findall(r"[\wäöüÄÖÜß]+|[^\w\s]", text, re.UNICODE)
    i = 0
    while i < len(tokens) - 1:
        tok = tokens[i].lower()
        nxt = tokens[i + 1].lower()
        if tok in PREPOSITIONS and nxt in DEFINITE_ARTICLES:
            out.append((tok, nxt, f"{tokens[i]} {tokens[i + 1]}"))
            i += 2
            continue
        i += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="Segmented DE JSONL")
    ap.add_argument("--output", required=True, type=Path, help="Output TSV path")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_contracted = 0
    n_uncontracted = 0
    with (
        open(args.input, encoding="utf-8") as fin,
        open(args.output, "w", encoding="utf-8") as fout,
    ):
        fout.write("segment_id\tkind\tprep\tarticle\tpp_text\tsentence\n")
        for line in fin:
            line = line.strip()
            if not line:
                continue
            seg = json.loads(line)
            sid = seg["id"]
            text = seg["text"]
            for prep, article, pp_text in extract_contracted(text):
                fout.write(f"{sid}\tcontracted\t{prep}\t{article}\t{pp_text}\t{text}\n")
                n_contracted += 1
            for prep, article, pp_text in extract_uncontracted(text):
                fout.write(f"{sid}\tuncontracted\t{prep}\t{article}\t{pp_text}\t{text}\n")
                n_uncontracted += 1

    print(f"wrote {args.output}: {n_contracted} contracted, {n_uncontracted} uncontracted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
