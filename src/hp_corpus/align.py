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
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .lexicon import anchor_weights, extract_anchors, lexical_bonus_matrix
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
# Penalties for unaligned sentences (mimic Vecalign's deletion/insertion
# costs). NOTE the sign of the tradeoff: a LOWER penalty makes gaps cheaper,
# so the DP forces FEWER spurious pairings; raising it pushes the DP to
# fabricate matches. 0.1 (not Vecalign's default) is calibrated against e5's
# cosine range on this corpus (true pairs ≈0.75–0.9, shifted/wrong 1:1 ≈0.70):
# at 0.2, a wrong shifted 1:1 (≈+0.72) beats the correct pairing plus one gap
# (≈0.85 − 0.2 = 0.65), so whole dialogue runs slid by ±1–3 sentences; at 0.1
# the correct path wins again (0.75 > 0.72). A positive-similarity 1:1 still
# beats gapping BOTH sentences at any penalty (e5 never scores below ~0.69),
# so this only arbitrates shift-vs-gap and multi-sentence groupings.
PENALTY_1_TO_0 = 0.1  # symmetric gap cost; configurable via AlignmentConfig
# Sentence-similarity floor; below this, prefer to leave both sides unaligned.
# (Inert in practice for e5 — no pair scores below 0.69 — kept for other
# models and as a guard.)
SIMILARITY_FLOOR = 0.3
# Per-sentence-beyond-1:1 discount for N:M moves (Vecalign-style). Mean-pair
# scoring alone would let an N:M absorb any neighbouring orphan, because e5
# scores even unrelated pairs ≥0.69; the discount leans the DP back toward
# 1:1 when the extra sentence adds little.
MULTI_SENTENCE_PENALTY = 0.02


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
    # --- Lexical prior (see hp_corpus.lexicon) -----------------------------
    # Add an IDF-weighted anchor bonus (shared proper nouns / distinctive
    # numbers) to the cosine matrix, and penalize pairing moves whose total
    # character-length ratio deviates far beyond the corpus norm. Targets the
    # residual error classes where e5 is not discriminative: crossed 1:1
    # pairings in dialogue, and short units force-paired with long ones.
    use_lexical_prior: bool = True
    # Calibrated on Ch.1-17 sandbox attribution-consistency: 0.3 is the
    # plateau point (58.8% baseline -> 69.8%; 0.4-0.5 dip slightly as the
    # capped bonus saturates into a binary "shares a rare anchor" signal).
    anchor_weight: float = 0.3
    anchor_bonus_cap: float = 0.15
    # de/zh char-ratio distribution on high-margin 1:1 pairs of this corpus:
    # median 3.45, |log deviation| ~1.1 at p5/p95. Pairs inside the band are
    # untouched; beyond it the penalty grows linearly in log space.
    length_ratio_mu: float = 3.45
    length_ratio_tau: float = 1.1
    length_penalty: float = 0.05
    # --- Similarity scoring mode --------------------------------------------
    # "cosine": raw cosine (current behaviour). "ratio": Artetxe & Schwenk
    # (2019) margin criterion — each cosine is divided by the mean of its
    # top-k neighbours on each side of the matrix, which amplifies pairs that
    # are distinctive relative to their local competition. Targets e5's cosine
    # compression (true ≈0.75-0.9, wrong ≈0.70): in a dialogue cluster where a
    # src sentence scores 0.78 on the right tgt and 0.77 on the wrong one, the
    # wrong one's neighbourhood (every other line mentioning the same cast)
    # inflates its denominator and the ratio criterion separates the two.
    similarity_mode: str = "cosine"
    ratio_k: int = 4
    # Gap / multi-sentence penalties, configurable because the ratio mode's
    # squashed score band (≈0.03 between true and wrong pairs) is narrower
    # than cosine's (≈0.05-0.2): penalties calibrated on the cosine scale
    # (0.1 / 0.02) overwhelm the ratio band and suppress gaps entirely (1:0
    # records collapse from 15 to 3 on the 6-chapter calibration set).
    gap_penalty: float = PENALTY_1_TO_0
    # 0.01 (down from the historical uniform 0.02): with the length-scaled
    # penalty short attribution neighbours join groups nearly free either
    # way, and 0.01 beat 0.02 on the DE->ZH recall audit (37 vs 34/59) at
    # identical ZH->DE attribution quality.
    multi_penalty: float = 0.01
    # N:M group arity cap. 3 was the historical move set (1:1..3:1, 2:2);
    # 5 is the judge-calibrated default: the DE->ZH recall audit (59-item
    # LLM-judged sample) showed arity-5 grouping recovers formerly gapped DE
    # sentences (gap stratum 0% -> 54% correct placement, overall sample
    # 47% -> 64%) with no change on the ZH->DE attribution layer and no
    # regression on stable pairs.
    max_group: int = 5
    # --- Two-pass anchor-constrained DP (bertalign-style) --------------------
    # Pass 1 runs a 1:1-only skeleton DP; its records with margin >=
    # anchor_margin_min become fixed walls. Pass 2 runs the full move set
    # between consecutive walls, so an error in one dialogue cluster cannot
    # propagate across a high-margin anchor into the next. Off by default
    # until the sandbox gate is met.
    two_pass: bool = False
    anchor_margin_min: float = 0.15


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
            f"length mismatch: {len(segment_ids)} segment_ids vs {len(sentences)} sentences"
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
        raise CacheValidationError(f"mixed language in segment IDs: {sorted(langs)}")
    if len(scopes) > 1:
        raise CacheValidationError(f"mixed chapter scope in segment IDs: {sorted(scopes)}")
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
        raise CacheValidationError(f"duplicate segment IDs (showing up to 3): {dupes}")

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


def _cache_paths(identity: CacheIdentity, cache_dir: Path) -> tuple[Path, Path]:
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


def _meta_matches(meta: dict[str, Any], identity: CacheIdentity, n_rows: int, dim: int) -> bool:
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


def _ratio_transform(sims: np.ndarray, k: int) -> np.ndarray:
    """Artetxe & Schwenk (2019) ratio margin criterion, squashed to [0, 1).

    For each cell (i, j): ``r = sims[i, j] / ((fwd[i] + bwd[j]) / 2)`` where
    ``fwd[i]`` is the mean of src sentence i's top-k cosines over all targets
    and ``bwd[j]`` the mean of tgt sentence j's top-k cosines over all
    sources. Both neighbourhoods live in the cross-language similarity matrix
    itself — exactly the candidates the DP is choosing among. A pair scores
    high only when it beats its own local competition, which discounts
    "everyone-mentions-Harry" dialogue clusters where raw cosines compress to
    a 0.05 band.

    The ratio r is centred near 1.0 (r > 1 means better than the neighbourhood
    average), so it is squashed with ``r / (1 + r)`` back onto the [0, 1)
    scale the DP's penalties (0.1 gaps, 0.02 multi-sentence) were calibrated
    on: true pairs (r ≈ 1.1-1.4) map to ≈0.52-0.58, neighbourhood-average
    pairs to ≈0.50, also-rans below.
    """
    n, m = sims.shape
    kk = max(1, min(k, n, m))
    fwd = -np.partition(-sims, kk - 1, axis=1)[:, :kk].mean(axis=1)
    bwd = -np.partition(-sims, kk - 1, axis=0)[:kk, :].mean(axis=0)
    denom = (fwd[:, None] + bwd[None, :]) / 2.0
    r = sims / np.maximum(denom, 1e-9)
    return r / (1.0 + r)


def _global_dp_align(
    src_vecs: np.ndarray,
    tgt_vecs: np.ndarray,
    band: float,
    *,
    bonus: np.ndarray | None = None,
    src_lens: list[int] | None = None,
    tgt_lens: list[int] | None = None,
    length_mu: float = 3.45,
    length_tau: float = 1.1,
    length_penalty: float = 0.1,
    similarity_mode: str = "cosine",
    ratio_k: int = 4,
    one_to_one_only: bool = False,
    gap_penalty: float = PENALTY_1_TO_0,
    multi_penalty: float = MULTI_SENTENCE_PENALTY,
    max_group: int = 3,
) -> list[tuple[list[int], list[int], float, float | None]]:
    """Global DP over all sentences, restricted to a diagonal band.

    (i, j) is reachable only when ``|i/n - j/m| <= band``. This enforces the
    locality assumption (translations preserve sentence order) and prevents
    far-apart false-positive matches. Returns one record per (src_idx_list,
    tgt_idx_list) tuple — no duplicates because each sentence index is
    consumed by exactly one transition.

    Pairing moves (all (di, dj) with ``1 < di + dj <= max_group + 1``,
    ``max_group=3`` giving the historical set 1:1, 1:2, 2:1, 1:3, 3:1, 2:2)
    score the **mean pairwise cosine over the cartesian product** of the
    consumed sentence groups, minus a length-scaled per-extra-sentence
    penalty. Mean — not max-single-pair — scoring credits every sentence the
    move consumes: under max scoring, a 1:1 on the best-matching half always
    dominates the N:M that also covers the remaining halves, so
    partial-context alignments win structurally.

    Each returned record carries a ``margin``: the score gap between the
    chosen move's path total and the best competing move at the same DP cell.
    Absolute cosine confidence is not discriminative for e5 (all pairs ≥0.69);
    the margin — how much the winning alignment beat its nearest alternative —
    is the ranking/review signal.
    """
    n = src_vecs.shape[0]
    m = tgt_vecs.shape[0]
    if n == 0 or m == 0:
        return []

    sims = src_vecs @ tgt_vecs.T  # (n, m)
    if similarity_mode == "ratio":
        sims = _ratio_transform(sims, ratio_k)
    elif similarity_mode != "cosine":
        raise ValueError(f"unknown similarity_mode: {similarity_mode!r}")
    if bonus is not None:
        sims = sims + bonus

    NEG = -1e9
    dp = np.full((n + 1, m + 1), NEG, dtype=np.float64)
    back: list[list[tuple[int, int, float] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0, 0] = 0.0

    # For each i, compute the reachable j-window from the diagonal constraint.
    def _j_window(i: int) -> tuple[int, int]:
        center = int(round(i * m / n))
        delta = int(max(1, band * m))
        return max(0, center - delta), min(m + 1, center + delta + 1)

    def _length_prior(pi: int, i: int, pj: int, j: int) -> float:
        """Penalty for a pairing move whose consumed character totals sit far
        outside the corpus length-ratio band. Applied at the MOVE level (total
        chars of the consumed group on each side), not per pair — a long DE
        quote legitimately pairs with a short ZH attribution member inside an
        N:M block; only the group totals should look balanced."""
        if length_penalty <= 0 or src_lens is None or tgt_lens is None:
            return 0.0
        ls = sum(src_lens[pi:i])
        lt = sum(tgt_lens[pj:j])
        if lt == 0:
            return 0.0
        deviation = abs(math.log((ls / lt) / length_mu))
        return length_penalty * max(0.0, deviation - length_tau)

    def _scaled_extra(pi: int, i: int, pj: int, j: int) -> float:
        """Extra-sentence penalty, scaled by sentence length. Short sentences
        (attribution fragments like *sagte er*) join a group nearly free —
        the judge audit found 47/182 attribution-layer pairs missing exactly
        such a neighbour — while full-length sentences keep the whole
        per-sentence discount that suppresses over-merging. Without length
        arrays this degrades to the historical per-count penalty."""
        e = 0.0
        for k in range(pi + 1, i):
            e += 1.0 if src_lens is None else min(1.0, src_lens[k] / 60.0)
        for k in range(pj + 1, j):
            e += 1.0 if tgt_lens is None else min(1.0, tgt_lens[k] / 60.0)
        return e

    def _cands(i: int, j: int) -> list[tuple[int, int, float]]:
        """Feasible (di, dj, move_score) moves into cell (i, j)."""
        out: list[tuple[int, int, float]] = []
        # 1:0 / 0:1 — leave one sentence unaligned (penalized)
        if i >= 1 and dp[i - 1, j] > NEG:
            out.append((1, 0, -gap_penalty))
        if j >= 1 and dp[i, j - 1] > NEG:
            out.append((0, 1, -gap_penalty))
        # Pairing moves — mean pairwise similarity over the consumed block.
        # one_to_one_only restricts to (1,1): used as pass 1 of the two-pass
        # scheme, where the goal is a robust 1:1 skeleton, not coverage.
        # max_group=3 reproduces the historical move set (1:1..3:1, 2:2).
        if one_to_one_only:
            moves: tuple[tuple[int, int], ...] = ((1, 1),)
        else:
            moves = tuple(
                (di, dj)
                for di in range(1, max_group + 1)
                for dj in range(1, max_group + 1)
                if 1 < di + dj <= max_group + 1
            )
        for di, dj in moves:
            pi, pj = i - di, j - dj
            if pi < 0 or pj < 0 or dp[pi, pj] <= NEG:
                continue
            s = float(sims[pi:i, pj:j].mean())
            if s < SIMILARITY_FLOOR:
                continue
            out.append(
                (
                    di,
                    dj,
                    s - multi_penalty * _scaled_extra(pi, i, pj, j) - _length_prior(pi, i, pj, j),
                )
            )
        return out

    for i in range(n + 1):
        j_lo, j_hi = _j_window(i)
        for j in range(j_lo, j_hi):
            if i == 0 and j == 0:
                continue
            best = NEG
            best_move: tuple[int, int, float] | None = None
            for di, dj, score in _cands(i, j):
                cand = dp[i - di, j - dj] + score
                if cand > best:
                    best, best_move = cand, (di, dj, score)

            if best_move is not None:
                dp[i, j] = best
                back[i][j] = best_move

    matches: list[tuple[list[int], list[int], float, float | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move is None:
            break
        di, dj, score = move
        alt_totals = [dp[i - a, j - b] + s for a, b, s in _cands(i, j) if (a, b) != (di, dj)]
        margin = float(dp[i, j] - max(alt_totals)) if alt_totals else None
        src_idx = list(range(i - di, i)) if di else []
        tgt_idx = list(range(j - dj, j)) if dj else []
        if src_idx or tgt_idx:
            matches.append((src_idx, tgt_idx, max(0.0, score), margin))
        i -= di
        j -= dj

    matches.reverse()
    return matches


def _two_pass_align(
    src_vecs: np.ndarray,
    tgt_vecs: np.ndarray,
    band: float,
    *,
    anchor_margin_min: float = 0.15,
    **dp_kwargs: Any,
) -> list[tuple[list[int], list[int], float, float | None]]:
    """Anchor-constrained two-pass DP (algorithm ported from bertalign's
    approach, GPL-free re-implementation against our own DP).

    Pass 1 runs a **1:1-only** DP (gaps allowed, no N:M moves) and keeps the
    records whose margin clears ``anchor_margin_min`` as fixed walls — a
    robust 1:1 skeleton. Restricting pass 1 to 1:1 is what makes the walls
    load-bearing: with the full move set, pass 1's own optimal path already
    passes through the anchors, and re-running the same scoring between them
    is a near no-op by the DP's interval optimality. The skeleton instead
    commits the unambiguous sentences, and pass 2 — the **full** move set on
    each interval between consecutive walls — fills in N:M groupings and gaps
    without being able to drag an error across an anchor.

    Interval scoring is interval-local: the bonus matrix and length arrays
    are sliced, and in ratio mode the neighbourhood is recomputed on the
    sub-block. Margins on interval records are relative to the interval's own
    competing moves; anchor records keep their pass-1 margin. With fewer than
    two qualifying anchors there is no skeleton to speak of, and the normal
    single-pass (full moves) result is returned instead.
    """
    n = src_vecs.shape[0]
    m = tgt_vecs.shape[0]
    pass1 = _global_dp_align(src_vecs, tgt_vecs, band, one_to_one_only=True, **dp_kwargs)
    anchors = [
        (src[0], tgt[0], score, margin)
        for src, tgt, score, margin in pass1
        if len(src) == 1 and len(tgt) == 1 and margin is not None and margin >= anchor_margin_min
    ]
    if len(anchors) < 2:
        return _global_dp_align(src_vecs, tgt_vecs, band, **dp_kwargs)

    bonus = dp_kwargs.get("bonus")
    src_lens = dp_kwargs.get("src_lens")
    tgt_lens = dp_kwargs.get("tgt_lens")

    def _interval(pi: int, pj: int, i: int, j: int):
        """Align the open interval (walls excluded); [] when empty. A
        one-sided interval (sentences on only one side) cannot enter the DP
        (it rejects empty matrices), so it is emitted as unaligned records
        directly — same shape a gap move would have produced."""
        if pi >= i and pj >= j:
            return []
        if pi >= i:  # only target sentences between the walls
            return [([], [j2], 0.0, None) for j2 in range(pj, j)]
        if pj >= j:  # only source sentences between the walls
            return [([i2], [], 0.0, None) for i2 in range(pi, i)]
        kwargs = dict(dp_kwargs)
        if bonus is not None:
            kwargs["bonus"] = bonus[pi:i, pj:j]
        if src_lens is not None:
            kwargs["src_lens"] = src_lens[pi:i]
        if tgt_lens is not None:
            kwargs["tgt_lens"] = tgt_lens[pj:j]
        return [
            ([a + pi for a in sa], [b + pj for b in tb], score, margin)
            for sa, tb, score, margin in _global_dp_align(
                src_vecs[pi:i], tgt_vecs[pj:j], band, **kwargs
            )
        ]

    out: list[tuple[list[int], list[int], float, float | None]] = []
    prev_i, prev_j = 0, 0
    for si, ti, score, margin in [*anchors, (n, m, None, None)]:
        out.extend(_interval(prev_i, prev_j, si, ti))
        if score is not None:  # the anchor itself (walls are records too)
            out.append(([si], [ti], score, margin))
            prev_i, prev_j = si + 1, ti + 1  # anchor indices are consumed
        else:
            prev_i, prev_j = si, ti
    return out


def _lexical_prior_kwargs(
    src_texts: list[str], tgt_texts: list[str], config: AlignmentConfig
) -> dict[str, Any]:
    """Keyword arguments for :func:`_global_dp_align` implementing the
    lexical prior, or an empty dict when disabled. Kept as a helper so the
    DP function stays embeddable/testable in isolation."""
    if not config.use_lexical_prior:
        return {}
    src_anchor_sets = [extract_anchors(t) for t in src_texts]
    tgt_anchor_sets = [extract_anchors(t) for t in tgt_texts]
    weights = anchor_weights(src_anchor_sets + tgt_anchor_sets)
    bonus = lexical_bonus_matrix(
        src_anchor_sets,
        tgt_anchor_sets,
        weights,
        weight=config.anchor_weight,
        cap=config.anchor_bonus_cap,
    )
    return {
        "bonus": bonus,
        "src_lens": [len(t) for t in src_texts],
        "tgt_lens": [len(t) for t in tgt_texts],
        "length_mu": config.length_ratio_mu,
        "length_tau": config.length_ratio_tau,
        "length_penalty": config.length_penalty,
    }


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
    dp_kwargs: dict[str, Any] = {
        "similarity_mode": config.similarity_mode,
        "ratio_k": config.ratio_k,
        "gap_penalty": config.gap_penalty,
        "multi_penalty": config.multi_penalty,
        "max_group": config.max_group,
        **_lexical_prior_kwargs(src_texts, tgt_texts, config),
    }
    if config.two_pass:
        dp_fn = _two_pass_align
        dp_kwargs["anchor_margin_min"] = config.anchor_margin_min
    else:
        dp_fn = _global_dp_align
    for src_idx_list, tgt_idx_list, score, margin in dp_fn(
        src_vecs,
        tgt_vecs,
        config.locality_band,
        **dp_kwargs,
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
                margin=margin,
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
    margins = [a.margin for a in alignments if a.margin is not None]
    by_type: dict[str, int] = {}
    by_method: dict[str, int] = {}
    for a in alignments:
        by_type[a.type] = by_type.get(a.type, 0) + 1
        by_method[a.method] = by_method.get(a.method, 0) + 1
    return {
        "count": len(alignments),
        "mean_confidence": sum(confs) / len(confs),
        "below_0p5": sum(1 for c in confs if c < 0.5),
        "mean_margin": (sum(margins) / len(margins)) if margins else None,
        "margin_below_0p02": sum(1 for mg in margins if mg < 0.02),
        "by_type": by_type,
        "by_method": by_method,
    }
