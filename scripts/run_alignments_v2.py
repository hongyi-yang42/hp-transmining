"""Run chapter-pair alignments, write per-pair manifests, and (optionally)
cross-pair diagnostics.

The production path is the DE-centred triangle the annotation pack consumes:
DE–EN and DE–ZH per chapter (``--chapters`` defaults to the full novel,
Ch.1–17 → 34 alignments). EN–ZH has no downstream consumer; pass
``--diagnostics`` to also produce it and run the cross-pair
confidence-uniqueness check — a cache-contamination heuristic whose result
is advisory only and never gates the exit code.

Outputs (all gitignored under ``data/derived/alignment_v2/``):
  * ``hp1_{src}_{tgt}_ch{NN}.jsonl`` — the alignment records (also copied to
    ``data/aligned/`` for use by downstream steps)
  * ``hp1_{src}_{tgt}_ch{NN}.manifest.json`` — input identity, segment
    counts, the full effective AlignmentConfig, the exact cache identities
    used per side, alignment type counts, missing count, confidence
    aggregates
  * ``cross_pair_uniqueness.json`` — with ``--diagnostics``: per-chapter
    check that the three pairwise confidence vectors are mutually distinct

Stdout: aggregate counts only — never novel text, never segment IDs from
the source novels (only the input/output file names and aggregate metrics).

Usage:
    uv run python scripts/run_alignments_v2.py [--force-recompute]
    uv run python scripts/run_alignments_v2.py --chapters 1 2 3
    uv run python scripts/run_alignments_v2.py --diagnostics
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from hp_corpus.align import (
    CACHE_SCHEMA_VERSION,
    AlignmentConfig,
    align_segments,
    load_segments,
    write_alignments_jsonl,
)
from hp_corpus.schema import Alignment

SEGMENTED_DIR = Path("data/segmented")
ALIGNED_DIR = Path("data/aligned")
MANIFEST_DIR = Path("data/derived/alignment_v2")
EMBED_DIR = Path("data/embeddings")
DEFAULT_CHAPTERS = tuple(range(1, 18))
MIN_CHAPTER = 1
MAX_CHAPTER = 17
# Production pairs: everything downstream (step4 master, cross-lingual map,
# annotation CSV) reads DE–EN and DE–ZH only. EN–ZH is diagnostics-only.
PRODUCTION_PAIRS = (("de", "en"), ("de", "zh"))
DIAGNOSTIC_PAIRS = (("en", "zh"),)
MODEL_NAME = "models/LaBSE"  # canonical since 2026-08-18 judge audit; e5 runs: set gap 0.1


def effective_pairs(diagnostics: bool) -> tuple[tuple[str, str], ...]:
    """Production pairs always; EN–ZH only under --diagnostics."""
    return PRODUCTION_PAIRS + (DIAGNOSTIC_PAIRS if diagnostics else ())


def validate_chapters(values: list[int]) -> tuple[tuple[int, ...], str | None]:
    """Normalise and range-check a --chapters list.

    Returns ``(chapters, None)`` or ``((), error_message)``. Kept pure for
    tests. Duplicate values collapse; order is normalised ascending."""
    if not values:
        return (), "no chapters requested"
    bad = [c for c in values if not MIN_CHAPTER <= c <= MAX_CHAPTER]
    if bad:
        return (), f"chapters must be within {MIN_CHAPTER}..{MAX_CHAPTER}: {sorted(set(bad))}"
    return tuple(sorted(set(values))), None


def _id_digest(segment_ids: list[str]) -> str:
    """Cheap stable digest of the ordered segment IDs, for the manifest.
    Independent of the embedding cache identity (which also covers texts)."""
    import hashlib

    h = hashlib.sha256()
    for sid in segment_ids:
        h.update(sid.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def _confidence_aggs(confidences: list[float]) -> dict[str, float]:
    if not confidences:
        return {"count": 0}
    return {
        "count": len(confidences),
        "min": min(confidences),
        "max": max(confidences),
        "mean": statistics.fmean(confidences),
        "median": statistics.median(confidences),
        "below_0p5": sum(1 for c in confidences if c < 0.5),
    }


def _type_counts(alignments: list[Alignment]) -> dict[str, int]:
    out: dict[str, int] = {}
    for a in alignments:
        out[a.type] = out.get(a.type, 0) + 1
    return out


def _verify_ids_belong_to(
    alignments: list[Alignment], src_ids: set[str], tgt_ids: set[str], src_lang: str, tgt_lang: str
) -> list[str]:
    """Return list of human-readable issues: any alignment ID that references
    a segment not in the expected src/tgt segmented input set."""
    issues = []
    for a in alignments:
        for sid in a.src:
            if sid not in src_ids:
                issues.append(f"{a.align_id}: src side ID {sid} not in {src_lang} input")
        for tid in a.tgt:
            if tid not in tgt_ids:
                issues.append(f"{a.align_id}: tgt side ID {tid} not in {tgt_lang} input")
    return issues


def _check_no_src_side_duplicates(
    alignments: list[Alignment], src_lang: str
) -> list[str]:
    """A German (src) segment must not appear in two conflicting records —
    i.e., be aligned to two different target sets. (Multi-sentence 1:n is
    fine because both targets are in one record.) Returns list of dup IDs."""
    seen: dict[str, str] = {}  # seg_id → align_id first seen in
    dups: list[str] = []
    for a in alignments:
        for sid in a.src:
            if sid in seen and seen[sid] != a.align_id:
                dups.append(f"{sid} (in {seen[sid]} and {a.align_id})")
            else:
                seen[sid] = a.align_id
    return dups


def run_one(src_lang: str, tgt_lang: str, chapter: int, force: bool) -> dict[str, Any]:
    ch = f"ch{chapter:02d}"
    src_path = SEGMENTED_DIR / f"hp1_{src_lang}_{ch}.jsonl"
    tgt_path = SEGMENTED_DIR / f"hp1_{tgt_lang}_{ch}.jsonl"
    out_name = f"hp1_{src_lang}_{tgt_lang}_{ch}.jsonl"

    src = load_segments(src_path)
    tgt = load_segments(tgt_path)

    cfg = AlignmentConfig(
        embed_cache_dir=EMBED_DIR,
        model_name=MODEL_NAME,
        force_recompute=force,
    )  # gap/multi penalties + arity come from AlignmentConfig defaults
    run = align_segments(src, tgt, cfg)
    alignments = run.records

    # Write alignment JSONL to BOTH the canonical data/aligned/ path (used by
    # Step 4 builder) and reflect the same file in the v2 manifest dir.
    out_aligned = ALIGNED_DIR / out_name
    write_alignments_jsonl(alignments, out_aligned)

    src_ids = {s.id for s in src}
    tgt_ids = {s.id for s in tgt}

    confidences = [a.confidence for a in alignments]
    type_counts = _type_counts(alignments)
    # "Missing count" = (records of type 1:0) + (records of type 0:1)
    missing_src_side = type_counts.get("1:0", 0)
    missing_tgt_side = type_counts.get("0:1", 0)

    id_issues = _verify_ids_belong_to(alignments, src_ids, tgt_ids, src_lang, tgt_lang)
    src_dups = _check_no_src_side_duplicates(alignments, src_lang)

    manifest = {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "chapter": chapter,
        "src_input": {
            "path": str(src_path),
            "segment_count": len(src),
            "id_digest_sha256": _id_digest([s.id for s in src]),
        },
        "tgt_input": {
            "path": str(tgt_path),
            "segment_count": len(tgt),
            "id_digest_sha256": _id_digest([s.id for s in tgt]),
        },
        # The full effective config, so a manifest alone reproduces the run's
        # semantics (penalties, priors, arity — not just model + gap).
        "alignment_config": asdict(cfg),
        # Cache identities come from the run itself — never re-discovered by
        # listing the cache directory (multiple models coexist there).
        "embedding_cache": {
            "schema_version": CACHE_SCHEMA_VERSION,
            "model_name": MODEL_NAME,
            "model_fingerprint": run.src_identity.model_fingerprint,
            "src": {
                "scope": run.src_identity.scope,
                "lang": run.src_identity.lang,
                "digest": run.src_identity.digest,
            },
            "tgt": {
                "scope": run.tgt_identity.scope,
                "lang": run.tgt_identity.lang,
                "digest": run.tgt_identity.digest,
            },
        },
        "output": {
            "path": str(out_aligned),
            "record_count": len(alignments),
            "type_counts": type_counts,
            "missing_count": {
                "src_side_unaligned_1_0": missing_src_side,
                "tgt_side_unaligned_0_1": missing_tgt_side,
            },
            "confidence": _confidence_aggs(confidences),
        },
        "verification": {
            "ids_in_input": len(id_issues) == 0,
            "no_src_side_duplicates": len(src_dups) == 0,
            "id_issues_count": len(id_issues),
            "src_side_duplicate_count": len(src_dups),
        },
    }
    return manifest


def _cross_pair_uniqueness(chapter: int) -> dict[str, Any]:
    """Verify that for one chapter, the three pairwise alignment confidence
    vectors are mutually distinct (not byte-identical, which would indicate
    cache reuse). Diagnostics only."""
    ch = f"ch{chapter:02d}"
    pairs = {}
    for src, tgt in effective_pairs(diagnostics=True):
        path = ALIGNED_DIR / f"hp1_{src}_{tgt}_{ch}.jsonl"
        if not path.exists():
            return {"chapter": chapter, "error": f"missing {path.name}"}
        confs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    a = Alignment.model_validate_json(line)
                    confs.append(round(a.confidence, 6))
        pairs[f"{src}_{tgt}"] = confs

    min_len = min(len(v) for v in pairs.values())
    overlap = {}
    keys = list(pairs.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            same = sum(1 for k in range(min_len) if pairs[a][k] == pairs[b][k])
            # Wholesale cache contamination would make ~every position agree.
            # LaBSE clamps some perfect pairs to confidence 1.0, so two
            # different pair files coincidentally holding 1.0 at the same
            # index is expected noise (e5 never reached the ceiling, hence
            # the historical zero-match baseline). Fail on proportional
            # agreement, not on any single coincidence.
            overlap[f"{a}_vs_{b}"] = {
                "matches": same,
                "compared": min_len,
                "match_rate": round(same / min_len, 4) if min_len else None,
                "status": "FAIL" if min_len and same / min_len > 0.10 else "OK",
            }

    return {"chapter": chapter, "overlap_in_first_min_len": overlap}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force-recompute", action="store_true")
    ap.add_argument(
        "--chapters",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHAPTERS),
        help=(f"chapters to align, {MIN_CHAPTER}..{MAX_CHAPTER} "
              f"(default: {' '.join(map(str, DEFAULT_CHAPTERS))})"),
    )
    ap.add_argument(
        "--diagnostics",
        action="store_true",
        help="also align EN-ZH (no downstream consumer) and run the "
             "cross-pair confidence-uniqueness check — advisory output "
             "that never gates the exit code",
    )
    args = ap.parse_args(argv)

    chapters, err = validate_chapters(args.chapters)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    pairs = effective_pairs(args.diagnostics)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"running {len(pairs) * len(chapters)} alignments on cache "
          f"{CACHE_SCHEMA_VERSION} (ch{chapters[0]:02d}..ch{chapters[-1]:02d})...")
    manifests = []
    for chapter in chapters:
        for src_lang, tgt_lang in pairs:
            m = run_one(src_lang, tgt_lang, chapter, args.force_recompute)
            manifests.append(m)
            ch = f"ch{chapter:02d}"
            mani_path = MANIFEST_DIR / f"hp1_{src_lang}_{tgt_lang}_{ch}.manifest.json"
            mani_path.write_text(
                json.dumps(m, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            v = m["verification"]
            print(
                f"  {src_lang}->{tgt_lang} {ch}: "
                f"records={m['output']['record_count']}, "
                f"mean_conf={m['output']['confidence'].get('mean', 0):.4f}, "
                f"missing(src/tgt)={v['id_issues_count']}/{v['src_side_duplicate_count']}, "
                f"id_issues={v['id_issues_count']}, "
                f"src_dups={v['src_side_duplicate_count']}"
            )

    if args.diagnostics:
        print("\ncross-pair confidence uniqueness (diagnostic, advisory):")
        overlaps = []
        all_unique = True
        for chapter in chapters:
            chk = _cross_pair_uniqueness(chapter)
            overlaps.append(chk)
            for pair_key, info in chk["overlap_in_first_min_len"].items():
                ok = info["status"] == "OK"
                all_unique = all_unique and ok
                print(
                    f"  ch{chapter:02d} {pair_key}: "
                    f"{info['matches']}/{info['compared']} matches "
                    f"(rate={info['match_rate']}) ({'OK' if ok else 'FAIL'})"
                )
        (MANIFEST_DIR / "cross_pair_uniqueness.json").write_text(
            json.dumps(overlaps, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"uniqueness: {'OK' if all_unique else 'FAIL'} (advisory)")

    # Aggregate verification summary — production checks only; EN-ZH and the
    # uniqueness heuristic never gate the exit code.
    n_verifications_ok = sum(
        1
        for m in manifests
        if m["verification"]["ids_in_input"]
        and m["verification"]["no_src_side_duplicates"]
    )
    print(
        f"\nverification: {n_verifications_ok}/{len(manifests)} alignments pass "
        f"all checks"
    )
    return 0 if n_verifications_ok == len(manifests) else 1


if __name__ == "__main__":
    sys.exit(main())
