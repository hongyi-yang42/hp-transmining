"""One-time patch: rewrite Stanza lemma .pt files from dicts_version=3
(JSON-in-gzip) to dicts_version=2 (pickle-in-gzip).

Why: Modelscope's July 2026 upload of stanfordnlp/stanza-{de,en,zh-hans}
uses dicts_version=3 for lemma files, where the ``dicts`` field is
gzip-compressed JSON. Stanza 1.14.0 (current PyPI as of 2026-08) only
knows dicts_version=1 (legacy 2-tuple) and dicts_version=2 (gzip-pickle).
Loading a v3 file fails::

    ValueError: too many values to unpack (expected 2)

The actual data format is identical between v2 and v3 — gzip-wrapped
pos_dict — only the serialization of pos_dict differs (pickle vs JSON).
This script reads each v3 file, parses the JSON, rewrites it as
gzip-pickle, and bumps dicts_version to 2.

Idempotent: files already at v2 (or v1) are skipped. Re-run safely after
re-downloading models.

Usage:
    uv run python scripts/patch_stanza_lemma_format.py [--resources-dir DIR]

Default resources dir: ~/stanza_resources (override with
$STANZA_RESOURCES_DIR).
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import pickle
import sys
from pathlib import Path


def pack_pos_dict(pos_dict: dict) -> bytes:
    """Serialize pos_dict as gzip-compressed pickle (Stanza v2 format)."""
    raw = pickle.dumps(pos_dict, protocol=4)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9) as gz:
        gz.write(raw)
    return buf.getvalue()


def patch_file(fp: Path) -> str:
    """Patch a single lemma .pt file in-place. Returns action taken."""
    import torch

    checkpoint = torch.load(fp, lambda storage, loc: storage, weights_only=True)
    v = checkpoint.get("dicts_version")
    if v != 3:
        return f"skip (dicts_version={v})"
    data = checkpoint["dicts"]
    if not isinstance(data, (bytes, bytearray)):
        return f"skip (dicts not bytes: {type(data).__name__})"
    raw = gzip.decompress(data)
    pos_dict = json.loads(raw.decode("utf-8"))
    checkpoint["dicts"] = pack_pos_dict(pos_dict)
    checkpoint["dicts_version"] = 2
    torch.save(checkpoint, fp)
    return "patched v3 → v2"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--resources-dir",
        type=Path,
        default=Path(os.environ.get("STANZA_RESOURCES_DIR", Path.home() / "stanza_resources")),
    )
    args = ap.parse_args()

    root = args.resources_dir
    if not root.exists():
        print(f"ERROR: resources dir does not exist: {root}", file=sys.stderr)
        return 1

    n_patched = n_skipped = 0
    for fp in sorted(root.glob("*/lemma/*.pt")):
        action = patch_file(fp)
        print(f"  {fp.relative_to(root)}: {action}")
        if action.startswith("patched"):
            n_patched += 1
        else:
            n_skipped += 1
    print(f"\ndone: {n_patched} patched, {n_skipped} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
