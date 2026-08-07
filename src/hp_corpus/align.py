"""Sentence alignment via multilingual embeddings + DP.

The DP supports the standard Vecalign alignment types (1:0, 0:1, 1:1, 1:2, 2:1,
1:3, 3:1, 2:2) over precomputed cosine similarities. The vendored
``vendor/vecalign`` checkout is not required — this module re-implements the
DP inline so the project has no hard dependency on the upstream code.

``align_segments`` is **language-pair-agnostic**: pass any two languages as
``src`` and ``tgt``. Embedding caches are **content-addressed** — the cache
identity is a SHA-256 over (schema version, model name, derived language,
derived chapter scope, ordered segment IDs, ordered sentence texts), so any
change to the input corpus produces a fresh cache. Mixed-language,
mixed-chapter, malformed-ID, duplicate-ID, or empty inputs raise
``CacheValidationError`` rather than silently falling back to a shared key.

The algorithm is a **global sentence-level DP with a diagonal band**:
(i, j) is reachable only when ``|i/n - j/m| <= locality_band``. This enforces
the translation-locality assumption (sentence at proportional position k in
the source almost certainly aligns to a sentence near proportional position k
in the target), which both speeds up the DP and prevents far-apart false
positives.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .schema import Alignment, Segment

_SEGMENT_ID_LANG_RE = re.compile(r"^[a-z0-9]+_([a-z]{2,3})_ch\d{2}_p\d{4}_s\d{3}$")
# Captures the corpus scope (book + lang + chapter) from a segment ID, e.g.
# "hp1_de_ch01_p0001_s001" → "hp1_de_ch01". Used to scope embedding caches so
# different chapters of the same language do not share vectors.
_SEGMENT_ID_SCOPE_RE = re.compile(r"^([a-z0-9]+_[a-z]{2,3}_ch\d{2})_p\d{4}_s\d{3}$")

# Bump when the cache file format or identity algorithm changes. Existing v2
# caches become invisible to a bumped version (a new subdir is used).
CACHE_SCHEMA_VERSION = "v2"

MANUAL_CONFIDENCE_THRESHOLD = 0.5
# Diagonal-band width as a fraction of length. (i, j) is reachable only when
# |i/n - j/m| <= locality_band. 0.15 = sentences can drift by ±15% of length.
DEFAULT_LOCALITY_BAND = 0.15
# Penalties for non-1:1 alignments (mimic Vecalign's deletion/insertion costs).
PENALTY_1_TO_0 = 0.2
PENALTY_0_TO_1 = 0.2
# Sentence-similarity floor; below this, prefer to leave both sides unaligned.
SIMILARITY_FLOOR = 0.3


class CacheValidationError(ValueError):
    """Raised when inputs to the embedding cache fail validation. Carries a
    human-readable ``reason`` explaining which invariant was violated."""


@dataclass(frozen=True)
class CacheIdentity:
    """Content-addressed cache identity for one embedding file.

    The ``digest`` is a SHA-256 over (schema_version, model_name, lang, scope,
    ordered segment_ids, ordered sentence_texts). Any change to inputs forces
    a new cache file; stale caches are invisible to the lookup path.
    """

    schema_version: str
    model_name: str
    lang: str
    scope: str
    segment_ids: tuple[str, ...]
    sentence_texts: tuple[str, ...]
    digest: str

    @property
    def digest16(self) -> str:
        return self.digest[:16]


@dataclass
class AlignmentConfig:
    embed_cache_dir: Path
    vecalign_dir: Path | None = None  # reserved for future subprocess use; currently unused
    manual_threshold: float = MANUAL_CONFIDENCE_THRESHOLD
    # Default to multilingual-e5-base: empirically produces much higher-quality
    # EN–ZH alignment than the smaller MiniLM (mean conf 0.78 vs 0.66 on
    # Chapter 1, with 0 records below the 0.5 manual-review threshold vs 36).
    # Caveat: e5's confidence is less discriminative (all pairs score >0.69),
    # so for review workflows consider running with both models and flagging
    # pairs where they disagree.
    model_name: str = "intfloat/multilingual-e5-base"
    locality_band: float = DEFAULT_LOCALITY_BAND
    # When True, recompute embeddings even if a cache file with matching digest
    # exists. Useful for forcing a clean re-encode after model upgrades or
    # suspected cache corruption.
    force_recompute: bool = False


def load_segments(path: str | Path) -> list[Segment]:
    out: list[Segment] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Segment.model_validate_json(line))
    return out


def _lang_of_segments(segments: list[Segment]) -> str:
    """Infer the language of a segment list from its first ID. Returns the
    derived language code (e.g. ``"de"``) or ``""`` if no segment has a
    recognizable ID. The empty string lets callers decide whether to treat
    that as an error (the cache layer does)."""
    for s in segments:
        m = _SEGMENT_ID_LANG_RE.match(s.id)
        if m:
            return m.group(1)
    return ""


def _scope_of_segments(segments: list[Segment]) -> str:
    """Infer the corpus scope (book_lang_chNN) of a segment list from its
    first ID, e.g. ``hp1_de_ch01``. Returns ``""`` if no segment has a
    recognizable ID."""
    for s in segments:
        m = _SEGMENT_ID_SCOPE_RE.match(s.id)
        if m:
            return m.group(1)
    return ""


def _build_cache_identity(
    segment_ids: list[str],
    sentences: list[str],
    model_name: str,
    schema_version: str = CACHE_SCHEMA_VERSION,
) -> CacheIdentity:
    """Validate inputs and build an immutable cache identity.

    Raises ``CacheValidationError`` on any of:
      * empty input
      * length mismatch (``len(segment_ids) != len(sentences)``)
      * malformed segment ID (regex mismatch)
      * mixed language across segments
      * mixed chapter scope across segments
      * duplicate segment IDs
    """
    if not segment_ids:
        raise CacheValidationError("empty segment list")
    if len(segment_ids) != len(sentences):
        raise CacheValidationError(
            f"length mismatch: {len(segment_ids)} segment_ids vs "
            f"{len(sentences)} sentences"
        )

    # Uniform-lang check (also catches malformed IDs).
    langs: set[str] = set()
    scopes: set[str] = set()
    for sid in segment_ids:
        m_lang = _SEGMENT_ID_LANG_RE.match(sid)
        if not m_lang:
            raise CacheValidationError(
                f"malformed segment ID (does not match expected pattern): {sid!r}"
            )
        langs.add(m_lang.group(1))
        m_scope = _SEGMENT_ID_SCOPE_RE.match(sid)
        if not m_scope:
            # Unreachable given the lang regex matched, but kept for safety.
            raise CacheValidationError(f"malformed segment ID (scope parse): {sid!r}")
        scopes.add(m_scope.group(1))
    if len(langs) > 1:
        raise CacheValidationError(
            f"mixed language in segment IDs: {sorted(langs)}"
        )
    if len(scopes) > 1:
        raise CacheValidationError(
            f"mixed chapter scope in segment IDs: {sorted(scopes)}"
        )
    lang = next(iter(langs))
    scope = next(iter(scopes))

    # Duplicate-ID check (with first-offender reporting).
    seen: set[str] = set()
    dupes: list[str] = []
    for sid in segment_ids:
        if sid in seen:
            dupes.append(sid)
            if len(dupes) >= 3:
                break
        seen.add(sid)
    if dupes:
        raise CacheValidationError(
            f"duplicate segment IDs (showing up to 3): {dupes}"
        )

    h = hashlib.sha256()
    h.update(schema_version.encode("utf-8"))
    h.update(b"\x1f")
    h.update(model_name.encode("utf-8"))
    h.update(b"\x1f")
    h.update(lang.encode("utf-8"))
    h.update(b"\x1f")
    h.update(scope.encode("utf-8"))
    h.update(b"\x1f")
    for sid, text in zip(segment_ids, sentences, strict=True):
        h.update(sid.encode("utf-8"))
        h.update(b"\x1e")  # RS — separates id from text within one record
        h.update(text.encode("utf-8"))
        h.update(b"\x1f")  # US — separates records
    digest = h.hexdigest()

    return CacheIdentity(
        schema_version=schema_version,
        model_name=model_name,
        lang=lang,
        scope=scope,
        segment_ids=tuple(segment_ids),
        sentence_texts=tuple(sentences),
        digest=digest,
    )


def _cache_paths(
    identity: CacheIdentity, cache_dir: Path
) -> tuple[Path, Path]:
    """Return (npy_path, meta_path) for a content-addressed cache entry."""
    root = cache_dir / identity.schema_version / identity.lang / identity.scope
    return root / f"{identity.digest16}.npy", root / f"{identity.digest16}.meta.json"


def _read_meta(meta_path: Path) -> dict[str, Any] | None:
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _meta_matches(
    meta: dict[str, Any], identity: CacheIdentity, n_rows: int, dim: int
) -> bool:
    """Verify that a loaded meta + vector shape are consistent with identity."""
    return (
        meta.get("schema_version") == identity.schema_version
        and meta.get("model_name") == identity.model_name
        and meta.get("lang") == identity.lang
        and meta.get("scope") == identity.scope
        and meta.get("digest") == identity.digest
        and meta.get("n_rows") == n_rows
        and meta.get("embedding_dim") == dim
        and tuple(meta.get("segment_ids") or ()) == identity.segment_ids
    )


def embed_sentences(
    sentences: list[str],
    segment_ids: list[str],
    cache_dir: str | Path,
    model_name: str = "intfloat/multilingual-e5-base",
    *,
    force_recompute: bool = False,
    schema_version: str = CACHE_SCHEMA_VERSION,
) -> np.ndarray:
    """Embed ``sentences`` with a sentence-transformers model, content-addressed.

    Cache layout::

        {cache_dir}/{schema_version}/{lang}/{scope}/{digest16}.npy
        {cache_dir}/{schema_version}/{lang}/{scope}/{digest16}.meta.json

    The cache identity (and therefore ``digest``) covers ``schema_version``,
    ``model_name``, the derived ``lang`` and ``scope``, the ordered
    ``segment_ids``, and the ordered ``sentences``. A cache hit requires the
    ``.meta.json`` to round-trip the same identity and a row count + dim that
    matches the loaded ``.npy``; otherwise the cache is treated as stale and
    a fresh encode is written.

    Raises:
        CacheValidationError: if ``segment_ids`` and ``sentences`` fail
            uniformity / well-formedness checks (mixed language, mixed
            chapter, malformed IDs, duplicates, empty, or length mismatch).
    """
    identity = _build_cache_identity(segment_ids, sentences, model_name, schema_version)
    npy_path, meta_path = _cache_paths(identity, Path(cache_dir))
    npy_path.parent.mkdir(parents=True, exist_ok=True)

    if not force_recompute and npy_path.exists():
        try:
            vecs = np.load(npy_path)
            meta = _read_meta(meta_path)
            if meta is not None and _meta_matches(
                meta, identity, vecs.shape[0], int(vecs.shape[1])
            ):
                return vecs
        except (OSError, ValueError):
            pass  # corrupt cache file → fall through to recompute

    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

    model = SentenceTransformer(model_name)
    # E5 models are trained with "query: "/"passage: " prefixes. For symmetric
    # sentence-similarity both sides get "query: ". Other models ignore this
    # safely (it just becomes part of the input), so we apply universally.
    prefix = "query: " if "e5" in model_name.lower() else ""
    inputs = [prefix + s for s in sentences]
    vecs = model.encode(
        inputs,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if vecs.shape[0] != len(sentences):
        raise RuntimeError(
            f"encoder returned {vecs.shape[0]} vectors for {len(sentences)} sentences"
        )

    # Atomic-ish write: write .npy first, then .meta.json. If the meta is
    # missing or stale on next load, the .npy is ignored (recomputed).
    np.save(npy_path, vecs)
    meta_path.write_text(
        json.dumps(
            {
                "schema_version": identity.schema_version,
                "model_name": identity.model_name,
                "lang": identity.lang,
                "scope": identity.scope,
                "n_rows": int(vecs.shape[0]),
                "embedding_dim": int(vecs.shape[1]),
                "digest": identity.digest,
                "digest16": identity.digest16,
                "segment_ids": list(identity.segment_ids),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return vecs


def _global_dp_align(
    src_vecs: np.ndarray,
    tgt_vecs: np.ndarray,
    band: float,
) -> list[tuple[list[int], list[int], float]]:
    """Global DP over all sentences, restricted to a diagonal band.

    (i, j) is reachable only when ``|i/n - j/m| <= band``. This enforces the
    locality assumption (translations preserve sentence order) and prevents
    far-apart false-positive matches. Returns one record per (src_idx_list,
    tgt_idx_list) tuple — no duplicates because each sentence index is
    consumed by exactly one transition.
    """
    n = src_vecs.shape[0]
    m = tgt_vecs.shape[0]
    if n == 0 or m == 0:
        return []

    sims = src_vecs @ tgt_vecs.T  # (n, m)

    NEG = -1e9
    dp = np.full((n + 1, m + 1), NEG, dtype=np.float64)
    back: list[list[tuple[int, int, float] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0, 0] = 0.0

    # For each i, compute the reachable j-window from the diagonal constraint.
    def _j_window(i: int) -> tuple[int, int]:
        center = int(round(i * m / n))
        delta = int(max(1, band * m))
        return max(0, center - delta), min(m + 1, center + delta + 1)

    for i in range(n + 1):
        j_lo, j_hi = _j_window(i)
        for j in range(j_lo, j_hi):
            if i == 0 and j == 0:
                continue
            best = NEG
            best_move: tuple[int, int, float] | None = None

            # 1:0 — src sentence i-1 unaligned (penalized)
            if i >= 1 and dp[i - 1, j] > NEG:
                cand = dp[i - 1, j] - PENALTY_1_TO_0
                if cand > best:
                    best, best_move = cand, (1, 0, -PENALTY_1_TO_0)

            # 0:1 — tgt sentence j-1 unaligned (penalized)
            if j >= 1 and dp[i, j - 1] > NEG:
                cand = dp[i, j - 1] - PENALTY_0_TO_1
                if cand > best:
                    best, best_move = cand, (0, 1, -PENALTY_0_TO_1)

            # 1:1
            if i >= 1 and j >= 1 and dp[i - 1, j - 1] > NEG:
                s = float(sims[i - 1, j - 1])
                if s >= SIMILARITY_FLOOR:
                    cand = dp[i - 1, j - 1] + s
                    if cand > best:
                        best, best_move = cand, (1, 1, s)

            # 1:2 — one src, two tgt
            if i >= 1 and j >= 2 and dp[i - 1, j - 2] > NEG:
                s = float(max(sims[i - 1, j - 2], sims[i - 1, j - 1]))
                if s >= SIMILARITY_FLOOR:
                    cand = dp[i - 1, j - 2] + s * 0.95
                    if cand > best:
                        best, best_move = cand, (1, 2, s * 0.95)

            # 2:1 — two src, one tgt
            if i >= 2 and j >= 1 and dp[i - 2, j - 1] > NEG:
                s = float(max(sims[i - 2, j - 1], sims[i - 1, j - 1]))
                if s >= SIMILARITY_FLOOR:
                    cand = dp[i - 2, j - 1] + s * 0.95
                    if cand > best:
                        best, best_move = cand, (2, 1, s * 0.95)

            # 1:3 — one src, three tgt (translator expanded src into multiple tgt sentences)
            if i >= 1 and j >= 3 and dp[i - 1, j - 3] > NEG:
                s = float(max(sims[i - 1, j - 3], sims[i - 1, j - 2], sims[i - 1, j - 1]))
                if s >= SIMILARITY_FLOOR:
                    cand = dp[i - 1, j - 3] + s * 0.90  # larger discount for 3-way merges
                    if cand > best:
                        best, best_move = cand, (1, 3, s * 0.90)

            # 3:1 — three src, one tgt (translator condensed)
            if i >= 3 and j >= 1 and dp[i - 3, j - 1] > NEG:
                s = float(max(sims[i - 3, j - 1], sims[i - 2, j - 1], sims[i - 1, j - 1]))
                if s >= SIMILARITY_FLOOR:
                    cand = dp[i - 3, j - 1] + s * 0.90
                    if cand > best:
                        best, best_move = cand, (3, 1, s * 0.90)

            # 2:2 — two src, two tgt (rare but occurs in dialogue-heavy passages)
            if i >= 2 and j >= 2 and dp[i - 2, j - 2] > NEG:
                s = float(
                    max(
                        sims[i - 2, j - 2],
                        sims[i - 2, j - 1],
                        sims[i - 1, j - 2],
                        sims[i - 1, j - 1],
                    )
                )
                if s >= SIMILARITY_FLOOR:
                    cand = dp[i - 2, j - 2] + s * 0.92
                    if cand > best:
                        best, best_move = cand, (2, 2, s * 0.92)

            if best_move is not None:
                dp[i, j] = best
                back[i][j] = best_move

    matches: list[tuple[list[int], list[int], float]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move is None:
            break
        di, dj, score = move
        src_idx = list(range(i - di, i)) if di else []
        tgt_idx = list(range(j - dj, j)) if dj else []
        if src_idx or tgt_idx:
            matches.append((src_idx, tgt_idx, max(0.0, score)))
        i -= di
        j -= dj

    matches.reverse()
    return matches


def align_segments(
    src: list[Segment],
    tgt: list[Segment],
    config: AlignmentConfig,
) -> list[Alignment]:
    """Align two lists of segments. Language-pair-agnostic: cache identities
    are derived from the segment IDs (book_lang_chNN scope) AND the segment
    texts, so any change to either side produces a fresh content-addressed
    cache. Strict validation: mixed language, mixed chapter, malformed IDs,
    duplicate IDs, or empty inputs raise ``CacheValidationError``."""
    src_ids = [s.id for s in src]
    tgt_ids = [s.id for s in tgt]
    src_texts = [s.text for s in src]
    tgt_texts = [s.text for s in tgt]
    src_vecs = embed_sentences(
        src_texts,
        src_ids,
        config.embed_cache_dir,
        config.model_name,
        force_recompute=config.force_recompute,
    )
    tgt_vecs = embed_sentences(
        tgt_texts,
        tgt_ids,
        config.embed_cache_dir,
        config.model_name,
        force_recompute=config.force_recompute,
    )

    records: list[Alignment] = []
    align_n = 0
    for src_idx_list, tgt_idx_list, score in _global_dp_align(
        src_vecs, tgt_vecs, config.locality_band
    ):
        src_ids = [src[i].id for i in src_idx_list if 0 <= i < len(src)]
        tgt_ids = [tgt[i].id for i in tgt_idx_list if 0 <= i < len(tgt)]
        if not src_ids and not tgt_ids:
            continue
        align_n += 1
        type_str = f"{len(src_ids)}:{len(tgt_ids)}"
        method = "vecalign_labse" if score >= config.manual_threshold else "manual"
        records.append(
            Alignment(
                align_id=f"a{align_n:04d}",
                en=src_ids,
                zh=tgt_ids,
                type=type_str,  # type: ignore[arg-type]
                confidence=max(0.0, min(1.0, score)),
                method=method,  # type: ignore[arg-type]
                validated=False,
            )
        )
    return records


def write_alignments_jsonl(alignments: list[Alignment], output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for a in alignments:
            f.write(a.model_dump_json() + "\n")
    return out


def alignment_summary(alignments: list[Alignment]) -> dict[str, Any]:
    if not alignments:
        return {"count": 0}
    confs = [a.confidence for a in alignments]
    by_type: dict[str, int] = {}
    by_method: dict[str, int] = {}
    for a in alignments:
        by_type[a.type] = by_type.get(a.type, 0) + 1
        by_method[a.method] = by_method.get(a.method, 0) + 1
    return {
        "count": len(alignments),
        "mean_confidence": sum(confs) / len(confs),
        "below_0p5": sum(1 for c in confs if c < 0.5),
        "by_type": by_type,
        "by_method": by_method,
    }
