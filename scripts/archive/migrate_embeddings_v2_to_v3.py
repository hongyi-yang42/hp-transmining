"""Migrate v2 embedding caches to the v3 identity WITHOUT re-encoding.

v3 cache identities additionally cover the model fingerprint
(``hp_corpus.align.model_fingerprint``), so swapped weights under an
unchanged model path can never silently reuse old vectors. The vectors
themselves are unchanged by that formula change: this script recomputes
each v2 entry's v3 identity from the segmented JSONL (the exact texts that
produced the vectors), verifies the segment-ID list and vector shape, then
copies the .npy and writes a fresh .meta.json under the v3 namespace.

Scope and safety:

  * Only entries whose ``model_name`` matches ``--model`` (default
    ``models/LaBSE``) are migrated; e5-era v2 entries stay in v2 untouched.
  * v2 is never modified or deleted — rollback is "stop using v3".
  * Entries whose segmented source is missing or no longer contains the
    recorded segment IDs (e.g. pre-source-switch ZH caches) are skipped and
    reported; the next alignment run simply re-encodes those scopes.
  * Idempotent: an entry whose v3 destination already exists is skipped.

Usage::

    uv run python scripts/migrate_embeddings_v2_to_v3.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from hp_corpus.align import (
    CACHE_SCHEMA_VERSION,
    _build_cache_identity,
    _cache_paths,
    model_fingerprint,
    verify_model_fingerprint,
)

V2_VERSION = "v2"
DEFAULT_MODEL = "models/LaBSE"


def load_segment_texts(path: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                texts[rec["id"]] = rec["text"]
    return texts


def migrate_one(
    meta_path: Path,
    *,
    model: str,
    fingerprint: str,
    embed_dir: Path,
    segmented_dir: Path,
) -> str:
    """Migrate a single v2 cache entry; return an outcome label."""
    import numpy as np

    meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("model_name") != model:
        return "skipped-other-model"
    ids = list(meta.get("segment_ids") or [])
    if not ids:
        return "skipped-no-ids"

    seg_path = segmented_dir / f"{meta.get('scope') or ''}.jsonl"
    if not seg_path.exists():
        return "skipped-missing-segments"
    texts_by_id = load_segment_texts(seg_path)
    try:
        texts = [texts_by_id[sid] for sid in ids]
    except KeyError:
        return "skipped-id-mismatch"

    identity = _build_cache_identity(ids, texts, model, fingerprint, CACHE_SCHEMA_VERSION)
    if identity.lang != meta.get("lang") or identity.scope != meta.get("scope"):
        return "skipped-scope-mismatch"

    dst_npy, dst_meta = _cache_paths(identity, embed_dir)
    if dst_npy.exists() and dst_meta.exists():
        return "already-migrated"

    src_npy = meta_path.with_name(meta_path.name[: -len(".meta.json")] + ".npy")
    if not src_npy.exists():
        return "skipped-npy-missing"
    shape = np.load(src_npy, mmap_mode="r").shape
    if shape[0] != len(ids):
        return "skipped-shape-mismatch"

    dst_npy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_npy, dst_npy)
    dst_meta.write_text(
        json.dumps(
            {
                "schema_version": identity.schema_version,
                "model_name": identity.model_name,
                "model_fingerprint": identity.model_fingerprint,
                "lang": identity.lang,
                "scope": identity.scope,
                "n_rows": int(shape[0]),
                "embedding_dim": int(shape[1]),
                "digest": identity.digest,
                "digest16": identity.digest16,
                "segment_ids": ids,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return "migrated"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--embed-dir", type=Path, default=Path("data/embeddings"))
    ap.add_argument("--segmented-dir", type=Path, default=Path("data/segmented"))
    args = ap.parse_args(argv)

    src_root = args.embed_dir / V2_VERSION
    if not src_root.exists():
        print(f"FAIL: no v2 cache directory at {src_root}", file=sys.stderr)
        return 2

    fingerprint = model_fingerprint(args.model, args.embed_dir)
    verify_model_fingerprint(args.model, fingerprint)
    print(f"model: {args.model} (fingerprint {fingerprint[:16]}…)")

    counts: dict[str, int] = {}
    metas = sorted(src_root.glob("*/*/*.meta.json"))
    for meta_path in metas:
        outcome = migrate_one(
            meta_path,
            model=args.model,
            fingerprint=fingerprint,
            embed_dir=args.embed_dir,
            segmented_dir=args.segmented_dir,
        )
        counts[outcome] = counts.get(outcome, 0) + 1
        if outcome != "migrated":
            print(f"  {outcome}: {meta_path.parent.relative_to(args.embed_dir)}")
    print(
        f"entries: {len(metas)} | "
        + " | ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
