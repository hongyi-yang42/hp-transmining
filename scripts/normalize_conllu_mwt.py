"""Normalize a CoNLL-U file by collapsing every multi-word-token range
(e.g. ``5-6	im``) into a single token, renumbering subsequent IDs and
head references.

Why: Stanza's tokenizer emits MWT ranges even when the MWT processor is
turned off — the contraction's surface form ("im") lives on the range
line, while POS/head info lives on the component lines. The
time-in-translation/conll-extractor expects surface forms as plain
single-line tokens. This adapter closes that gap without requiring a
re-parse.

Usage:
    uv run python scripts/normalize_conllu_mwt.py \\
        --input data/parsed/hp1_de_ch01.conllu \\
        --output data/parsed/hp1_de_ch01_nomwt.conllu
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RANGE_ID_RE = re.compile(r"^(\d+)-(\d+)$")


def parse_sentence(lines: list[str]) -> tuple[list[str], list[dict]]:
    """Split into comments + token rows. Returns (comments, token_dicts)."""
    comments = [ln for ln in lines if ln.startswith("#")]
    tokens = []
    for ln in lines:
        if ln.startswith("#"):
            continue
        cols = ln.split("\t")
        if len(cols) < 10:
            continue
        tokens.append(
            {
                "id": cols[0],
                "form": cols[1],
                "lemma": cols[2],
                "upos": cols[3],
                "xpos": cols[4],
                "feats": cols[5],
                "head": cols[6],
                "deprel": cols[7],
                "deps": cols[8],
                "misc": cols[9],
            }
        )
    return comments, tokens


def collapse_mwt(tokens: list[dict]) -> list[dict]:
    """Merge each MWT range into a single token. The merged token takes
    the SURFACE form from the range line, but POS/head/deprel/feats from
    the first component word (so it links into the dependency tree the
    same way the original preposition did)."""
    # Map: original_id (str) → token dict in input order.
    by_id = {t["id"]: t for t in tokens}
    # Walk tokens in order; for each range, emit ONE merged token, mark
    # its components as consumed.
    consumed: set[str] = set()
    out: list[dict] = []
    for t in tokens:
        if t["id"] in consumed:
            continue
        m = RANGE_ID_RE.match(t["id"])
        if m:
            first_id, last_id = m.group(1), m.group(2)
            # The component IDs are the integers in [first_id, last_id].
            comp_ids = [str(i) for i in range(int(first_id), int(last_id) + 1)]
            first_comp = by_id.get(first_id)
            if first_comp is None:
                # No component lines for this range — drop entirely.
                continue
            merged = {
                "id": first_id,  # will be renumbered later
                "form": t["form"],  # surface form from range line
                "lemma": first_comp["lemma"],
                "upos": first_comp["upos"],
                "xpos": first_comp["xpos"],
                "feats": first_comp["feats"],
                "head": first_comp["head"],
                "deprel": first_comp["deprel"],
                "deps": first_comp["deps"],
                "misc": first_comp["misc"],
            }
            out.append(merged)
            consumed.update(comp_ids)
        else:
            out.append(t)
    return out


def renumber(tokens: list[dict]) -> list[dict]:
    """Renumber IDs 1..N and rewrite head references accordingly."""
    # Build old_id → new_id map. Heads reference single IDs (not ranges).
    old_to_new: dict[str, int] = {}
    for new_idx, t in enumerate(tokens, start=1):
        old_to_new[t["id"]] = new_idx
    out = []
    for new_idx, t in enumerate(tokens, start=1):
        new = dict(t)
        new["id"] = str(new_idx)
        head = t["head"]
        if head in old_to_new:
            new["head"] = str(old_to_new[head])
        elif head == "0":
            new["head"] = "0"
        else:
            # Unknown head (shouldn't happen) — leave as-is.
            pass
        out.append(new)
    return out


def serialize(comments: list[str], tokens: list[dict]) -> str:
    lines = list(comments)
    for t in tokens:
        lines.append(
            "\t".join(
                [
                    t["id"],
                    t["form"],
                    t["lemma"],
                    t["upos"],
                    t["xpos"],
                    t["feats"],
                    t["head"],
                    t["deprel"],
                    t["deps"],
                    t["misc"],
                ]
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    text = args.input.read_text(encoding="utf-8")
    # Sentences separated by blank lines.
    blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b]
    n_in = 0
    n_out = 0
    n_ranges = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        for block in blocks:
            lines = block.split("\n")
            comments, tokens = parse_sentence(lines)
            n_in += len(tokens)
            # Count ranges before collapse
            n_ranges += sum(1 for t in tokens if RANGE_ID_RE.match(t["id"]))
            tokens = collapse_mwt(tokens)
            tokens = renumber(tokens)
            n_out += len(tokens)
            fout.write(serialize(comments, tokens) + "\n")
    print(f"normalized {args.input}: {n_in} → {n_out} tokens ({n_ranges} MWT ranges collapsed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
