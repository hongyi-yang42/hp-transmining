"""Synthetic tests for scripts/migrate_embeddings_v2_to_v3.py.

The migration must reuse the already-encoded v2 vectors (no re-encode),
carry the model fingerprint into the new meta, leave v2 untouched, skip
entries it cannot rebuild (missing / diverged segmented source), and be
idempotent. All text is synthetic.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from hp_corpus.align import CACHE_SCHEMA_VERSION, embed_sentences

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate_embeddings_v2_to_v3.py"
_spec = importlib.util.spec_from_file_location("migrate_embeddings_v2_to_v3", _SCRIPT)
mig = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(mig)


def _ids(n: int) -> list[str]:
    return [f"hp1_de_ch01_p0001_s{i:03d}" for i in range(1, n + 1)]


def _write_segmented(path: Path, ids: list[str], texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sid, text in zip(ids, texts, strict=True):
            f.write(json.dumps({"id": sid, "text": text}, ensure_ascii=False) + "\n")


def _v2_entries(embed_dir: Path) -> list[Path]:
    return sorted((embed_dir / "v2").glob("*/*/*.meta.json"))


def test_migration_copies_vectors_without_reencoding(
    tmp_path: Path, fake_encoder
) -> None:
    embed_dir = tmp_path / "embeddings"
    seg_dir = tmp_path / "segmented"
    ids = _ids(3)
    texts = [f"synthetic de sentence {i}" for i in range(3)]

    # v2-era cache (the pre-fingerprint identity algorithm).
    embed_sentences(texts, ids, embed_dir, "fake/M", schema_version="v2")
    v2_metas = _v2_entries(embed_dir)
    assert len(v2_metas) == 1
    v2_npy = v2_metas[0].with_name(v2_metas[0].name[: -len(".meta.json")] + ".npy")
    v2_vecs = np.load(v2_npy)

    _write_segmented(seg_dir / "hp1_de_ch01.jsonl", ids, texts)
    rc = mig.main(
        ["--model", "fake/M", "--embed-dir", str(embed_dir), "--segmented-dir", str(seg_dir)]
    )
    assert rc == 0

    v3_metas = sorted((embed_dir / CACHE_SCHEMA_VERSION).glob("*/*/*.meta.json"))
    assert len(v3_metas) == 1
    meta = json.loads(v3_metas[0].read_text(encoding="utf-8"))
    assert meta["model_fingerprint"] == "id:fake/M"
    assert meta["schema_version"] == CACHE_SCHEMA_VERSION
    v3_npy = v3_metas[0].with_name(v3_metas[0].name[: -len(".meta.json")] + ".npy")
    assert np.array_equal(np.load(v3_npy), v2_vecs)

    # v2 untouched (rollback path).
    assert len(_v2_entries(embed_dir)) == 1

    # Idempotent: second run reports already-migrated, still one v3 entry.
    rc = mig.main(
        ["--model", "fake/M", "--embed-dir", str(embed_dir), "--segmented-dir", str(seg_dir)]
    )
    assert rc == 0
    assert len(sorted((embed_dir / CACHE_SCHEMA_VERSION).glob("*/*/*.meta.json"))) == 1


def test_migration_skips_other_models_and_diverged_sources(
    tmp_path: Path, fake_encoder
) -> None:
    embed_dir = tmp_path / "embeddings"
    seg_dir = tmp_path / "segmented"

    ids = _ids(2)
    texts = ["alpha synth", "beta synth"]
    # One canonical-model entry with matching segmented source...
    embed_sentences(texts, ids, embed_dir, "fake/M", schema_version="v2")
    # ...one entry from a different model (must stay in v2)...
    embed_sentences(texts, ids, embed_dir, "other/M", schema_version="v2")
    # ...and one canonical-model entry whose segmented source is absent
    # (scope hp1_de_ch02 has no JSONL in seg_dir).
    ch02_ids = ["hp1_de_ch02_p0001_s001", "hp1_de_ch02_p0001_s002"]
    embed_sentences(texts, ch02_ids, embed_dir, "fake/M", schema_version="v2")

    _write_segmented(seg_dir / "hp1_de_ch01.jsonl", ids, texts)
    rc = mig.main(
        ["--model", "fake/M", "--embed-dir", str(embed_dir), "--segmented-dir", str(seg_dir)]
    )
    assert rc == 0

    # Only the ch01 fake/M entry made it to v3.
    v3_scopes = {p.parent.name for p in (embed_dir / CACHE_SCHEMA_VERSION).glob("*/*/*.meta.json")}
    assert v3_scopes == {"hp1_de_ch01"}
    # All three v2 entries remain.
    assert len(_v2_entries(embed_dir)) == 3
