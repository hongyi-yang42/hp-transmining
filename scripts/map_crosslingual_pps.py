"""Generate cross-lingual PP mappings for the 110 author-match contracted DE PPs.

Reads the Step 4 candidate JSONL (already built by build_step4_annotation_pack.py),
filters to ``de_form == 'contracted' AND author_resource_match == True`` (110 of
the 239 candidates), then proposes an EN and ZH equivalent PP for each one
using ``hp_corpus.crosslingual_map``.

Output goes under ``data/derived/step4/`` (gitignored):

  - crosslingual_mappings.jsonl     one record per DE PP, with proposed
                                    EN/ZH spans + scores + alternatives
  - crosslingual_mappings.summary.json   aggregate counts only

Stdout carries aggregate counts only — no surface forms, lemmas, or
segment IDs from the source novels.

Usage:
    uv run python scripts/map_crosslingual_pps.py \
        [--candidates data/derived/step4/ch1_3_all_candidates.jsonl] \
        [--parsed-dir data/parsed] \
        [--output-dir data/derived/step4]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from hp_corpus.crosslingual_map import (
    MappingResult,
    PPElement,
    Sentence,
    index_blocks_by_segment,
    parse_conllu,
    propose_for_sides,
)


def _load_candidates(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_parsed_chapters(parsed_dir: Path, lang: str, chapters: list[int]) -> dict[str, Sentence]:
    """Load migrated CoNLL-U for one language across the requested chapters.

    Returns a {parse_block_id → Sentence} map. ``parse_conllu`` fails
    closed on unmigrated or inconsistent provenance. Block ids embed the
    chapter, so a collision across chapter files would mean corrupt
    input — fail closed instead of silently overwriting.
    """
    out: dict[str, Sentence] = {}
    for ch in chapters:
        path = parsed_dir / f"hp1_{lang}_ch{ch:02d}.conllu"
        if not path.exists():
            print(f"WARN: missing parsed file {path}", file=sys.stderr)
            continue
        chapter_sents = parse_conllu(path)
        overlap = out.keys() & chapter_sents.keys()
        if overlap:
            raise ValueError(
                f"duplicate parse_block_ids across chapter files for {lang}: "
                f"{len(overlap)} colliding ids (first chapter {ch})"
            )
        out.update(chapter_sents)
    return out


def _serialize_pp(pp: PPElement) -> dict:
    return {
        "sent_id": pp.sent_id,
        "prep_surface": pp.prep_surface,
        "prep_lemma": pp.prep_lemma,
        "prep_token_id": pp.prep_token_id,
        "head_surface": pp.head_surface,
        "head_lemma": pp.head_lemma,
        "head_token_id": pp.head_token_id,
        "head_upos": pp.head_upos,
        "span_text": pp.span_text,
        "span_token_ids": list(pp.span_token_ids),
        "char_start": pp.char_start,
        "char_end": pp.char_end,
    }


def _serialize_scored(scored) -> dict:
    return {
        "score": round(scored.score, 3),
        "components": [{"signal": s, "weight": round(w, 3)} for s, w in scored.components],
        "pp": _serialize_pp(scored.pp),
    }


def _serialize_result(side: str, result: MappingResult) -> dict:
    return {
        f"{side}_status": result.status,
        f"{side}_best_score": round(result.best_score, 3),
        f"{side}_components": [
            {"signal": s, "weight": round(w, 3)} for s, w in result.components
        ],
        f"{side}_best": _serialize_pp(result.best) if result.best else None,
        f"{side}_alternatives": [_serialize_scored(s) for s in result.alternatives],
        f"{side}_n_candidates": result.n_candidates_considered,
        f"{side}_reason": result.reason,
    }


def _blocks_for_alignment(
    alignment_ids: list[str], by_segment: dict[str, list[Sentence]]
) -> tuple[list[Sentence], int]:
    """Resolve alignment-level segment ids to their parse blocks.

    ``cand["{lang}_sentence_ids"]`` holds source_segment_id values (the
    alignment unit), while parsed files are keyed by parse_block_id. A
    segment the parser split into several blocks contributes **all** its
    blocks, in document order. Returns ``(blocks, n_unresolved_ids)``;
    an alignment id that resolves to zero blocks is counted, not
    silently dropped.
    """
    blocks: list[Sentence] = []
    unresolved = 0
    for sid in alignment_ids:
        got = by_segment.get(sid, [])
        if not got:
            unresolved += 1
        blocks.extend(got)
    return blocks, unresolved


def build_mapping_record(
    cand: dict,
    en_by_segment: dict[str, list[Sentence]],
    zh_by_segment: dict[str, list[Sentence]],
) -> dict:
    """Build one cross-lingual mapping record from a candidate."""
    en_sents, en_unresolved = _blocks_for_alignment(
        cand.get("en_sentence_ids", []), en_by_segment
    )
    zh_sents, zh_unresolved = _blocks_for_alignment(
        cand.get("zh_sentence_ids", []), zh_by_segment
    )

    en_result, zh_result = propose_for_sides(
        de_prep_normalized=cand["de_prep_normalized"],
        de_pp_surface=cand["de_pp_surface"],
        de_head_lemma=cand["de_head_lemma"],
        en_sentences=en_sents,
        zh_sentences=zh_sents,
        en_align_confidence=float(cand.get("en_alignment_confidence", 0.0)),
        zh_align_confidence=float(cand.get("zh_alignment_confidence", 0.0)),
    )

    record: dict = {
        "datapoint_id": cand["datapoint_id"],
        "chapter": cand["chapter"],
        "de_parse_block_id": cand["de_parse_block_id"],
        "de_source_segment_id": cand["de_source_segment_id"],
        "de_token_start": cand["de_token_start"],
        "de_token_end": cand["de_token_end"],
        "de_pp_surface": cand["de_pp_surface"],
        "de_prep_normalized": cand["de_prep_normalized"],
        "de_head_lemma": cand["de_head_lemma"],
        "de_form": cand["de_form"],
        "author_resource_match": cand["author_resource_match"],
        "en_alignment_cardinality": cand["en_alignment_cardinality"],
        "en_alignment_confidence": cand["en_alignment_confidence"],
        "zh_alignment_cardinality": cand["zh_alignment_cardinality"],
        "zh_alignment_confidence": cand["zh_alignment_confidence"],
        "n_en_sentences_seen": len(en_sents),
        "n_zh_sentences_seen": len(zh_sents),
        "n_en_alignment_ids_unresolved": en_unresolved,
        "n_zh_alignment_ids_unresolved": zh_unresolved,
    }
    record.update(_serialize_result("en", en_result))
    record.update(_serialize_result("zh", zh_result))
    return record


def summarize(records: list[dict]) -> dict:
    """Aggregate counts only — no surface forms or lemmas."""
    en_status = Counter(r["en_status"] for r in records)
    zh_status = Counter(r["zh_status"] for r in records)

    # Cross-tabulate EN × ZH status so we can see how many candidates
    # have a high-quality match on both sides simultaneously.
    both: Counter = Counter()
    for r in records:
        key = (r["en_status"], r["zh_status"])
        both[key] += 1

    # Signal-frequency breakdown among the 'matched' best candidates.
    en_signals: Counter = Counter()
    zh_signals: Counter = Counter()
    for r in records:
        for sig in r["en_components"]:
            en_signals[sig["signal"]] += 1
        for sig in r["zh_components"]:
            zh_signals[sig["signal"]] += 1

    # Score distributions (rounded to nearest int for compactness).
    en_score_buckets: Counter = Counter()
    zh_score_buckets: Counter = Counter()
    for r in records:
        en_score_buckets[round(r["en_best_score"])] += 1
        zh_score_buckets[round(r["zh_best_score"])] += 1

    by_chapter: dict[int, dict[str, int]] = {}
    for r in records:
        ch = r["chapter"]
        bucket = by_chapter.setdefault(ch, {"matched": 0, "candidate": 0, "unmappable": 0})
        bucket[r["en_status"]] += 1  # EN-side status per chapter as a representative cut

    en_unresolved = sum(r["n_en_alignment_ids_unresolved"] for r in records)
    zh_unresolved = sum(r["n_zh_alignment_ids_unresolved"] for r in records)

    return {
        "record_total": len(records),
        "en_status": dict(en_status),
        "zh_status": dict(zh_status),
        "both_status": {f"{k[0]}|{k[1]}": v for k, v in sorted(both.items())},
        "en_signal_frequency": dict(en_signals.most_common()),
        "zh_signal_frequency": dict(zh_signals.most_common()),
        "en_score_distribution": {str(k): v for k, v in sorted(en_score_buckets.items())},
        "zh_score_distribution": {str(k): v for k, v in sorted(zh_score_buckets.items())},
        "per_chapter_en_status": {str(k): v for k, v in sorted(by_chapter.items())},
        "en_alignment_ids_unresolved": en_unresolved,
        "zh_alignment_ids_unresolved": zh_unresolved,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/derived/step4/ch1_3_all_candidates.jsonl"),
    )
    ap.add_argument("--parsed-dir", type=Path, default=Path("data/parsed"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/derived/step4"))
    ap.add_argument(
        "--form",
        choices=["contracted", "uncontracted", "both"],
        default="contracted",
        help="Which DE form to map (default: contracted, the paper's primary set).",
    )
    ap.add_argument(
        "--require-author-match",
        action="store_true",
        default=True,
        help=(
            "Only map candidates where author_resource_match is True "
            "(paper FILTER_CONTRACTED_123)."
        ),
    )
    ap.add_argument(
        "--no-require-author-match",
        dest="require_author_match",
        action="store_false",
        help="Include candidates the paper didn't annotate.",
    )
    args = ap.parse_args(argv)

    candidates = _load_candidates(args.candidates)

    selected = [
        c
        for c in candidates
        if (args.form == "both" or c["de_form"] == args.form)
        and (not args.require_author_match or c["author_resource_match"])
    ]

    if not selected:
        print("NO candidates matched the filter.", file=sys.stderr)
        return 2

    chapters = sorted({int(c["chapter"]) for c in selected})
    en_parsed = _load_parsed_chapters(args.parsed_dir, "en", chapters)
    zh_parsed = _load_parsed_chapters(args.parsed_dir, "zh", chapters)
    en_by_segment = index_blocks_by_segment(en_parsed)
    zh_by_segment = index_blocks_by_segment(zh_parsed)

    records: list[dict] = []
    for cand in selected:
        records.append(build_mapping_record(cand, en_by_segment, zh_by_segment))

    # Stable-sort for reproducible output.
    records.sort(
        key=lambda r: (
            r["chapter"],
            r["de_source_segment_id"],
            r["de_parse_block_id"],
            r["de_token_start"],
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output_dir / "crosslingual_mappings.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    summary = summarize(records)
    summary["parameters"] = {
        "candidates_path": str(args.candidates),
        "form": args.form,
        "require_author_match": args.require_author_match,
        "chapters": chapters,
    }
    out_summary = args.output_dir / "crosslingual_mappings.summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    # ----- stdout: aggregate counts only -----
    print(
        f"selected candidates: {len(selected)} "
        f"(form={args.form}, author_match={args.require_author_match})"
    )
    print(f"  chapters: {chapters}")
    print(f"  EN status: {summary['en_status']}")
    print(f"  ZH status: {summary['zh_status']}")
    print(f"  both EN and ZH 'matched': {summary['both_status'].get('matched|matched', 0)}")
    print(
        f"  unresolved alignment segment ids: "
        f"EN={summary['en_alignment_ids_unresolved']} "
        f"ZH={summary['zh_alignment_ids_unresolved']}"
    )
    print("outputs:")
    print(f"  {out_jsonl}")
    print(f"  {out_summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
