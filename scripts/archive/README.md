# Archived one-time scripts

Retired scripts kept for provenance: each documents a migration or audit
that has already been executed against this project's data. They are not
part of the production pipeline, are not covered by CI, and will not be
updated if `src/hp_corpus/` APIs change. For the exact execution context,
see git history (`git log --follow scripts/archive/<name>.py`).

| Script | What it did (already executed) | Where its outputs live |
|---|---|---|
| `migrate_embeddings_v2_to_v3.py` | Re-encoded-free migration of existing embedding caches to the fingerprint-scoped v3 cache identity. Only needed if you still hold v2 caches; fresh runs build v3 directly. | `data/embeddings/` |
| `migrate_parse_block_provenance.py` | Backfilled `parse_block_id` provenance into CoNLL-U files produced before `parse.py` emitted it natively (dc83b0a). | `data/parsed/` |
| `build_zh_manifests.py` | Wrote the per-chapter source manifests for the 2026-08-15 ZH source switch (calibre PDF, Source Option A). | `data/derived/zh_source_audit/manifests/` |
| `zh_source_audit.py` | Aggregate-only audit comparing the unified ZH ebook source against the retired image-only scan (line-loss defect). | `data/derived/zh_source_audit/` |
| `zh_id_impact.py` | Measured sentence-ID churn from the ZH source switch — established that old ZH IDs do not carry over. | `data/derived/zh_source_audit/` |
