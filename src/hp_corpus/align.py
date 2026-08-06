"""EN–ZH sentence alignment via LaBSE embeddings + DP.

The DP supports the standard Vecalign alignment types (1:0, 0:1, 1:1, 1:2, 2:1)
over precomputed LaBSE cosine similarities. The vendored ``vendor/vecalign``
checkout is not required — this module re-implements the DP inline so the
project has no hard dependency on the upstream code. If you want to use the
upstream Vecalign for cross-checking, clone it per the README and call it
separately; results from this module are produced by the inline DP.

The algorithm is a **global sentence-level DP with a diagonal band**:
(i, j) is reachable only when ``|i/n - j/m| <= locality_band``. This enforces
the translation-locality assumption (sentence at proportional position k in
EN almost certainly aligns to a sentence near proportional position k in ZH),
which both speeds up the DP and prevents far-apart false positives.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .schema import Alignment, Segment

MANUAL_CONFIDENCE_THRESHOLD = 0.5
# Diagonal-band width as a fraction of length. (i, j) is reachable only when
# |i/n - j/m| <= locality_band. 0.15 = sentences can drift by ±15% of length.
DEFAULT_LOCALITY_BAND = 0.15
# Penalties for non-1:1 alignments (mimic Vecalign's deletion/insertion costs).
PENALTY_1_TO_0 = 0.2
PENALTY_0_TO_1 = 0.2
# Sentence-similarity floor; below this, prefer to leave both sides unaligned.
SIMILARITY_FLOOR = 0.3


@dataclass
class AlignmentConfig:
    embed_cache_dir: Path
    vecalign_dir: Path | None = None  # reserved for future subprocess use; currently unused
    manual_threshold: float = MANUAL_CONFIDENCE_THRESHOLD
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    locality_band: float = DEFAULT_LOCALITY_BAND


def load_segments(path: str | Path) -> list[Segment]:
    out: list[Segment] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Segment.model_validate_json(line))
    return out


def _cache_key(lang: str, model_name: str) -> str:
    """Embedding cache key scoped by language + model so swapping models does
    not silently reuse stale vectors."""
    slug = model_name.replace("/", "_").replace("-", "_").replace(".", "_").lower()
    return f"{lang}_{slug}"


def embed_sentences(
    sentences: list[str],
    cache_key: str,
    cache_dir: str | Path,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> np.ndarray:
    """Embed sentences with the configured sentence-transformers model; cache to disk.

    Default is the multilingual MiniLM (~470 MB, 384-dim) which is dramatically
    smaller than LaBSE (~1.8 GB) and adequate for first-pass alignment. Override
    via AlignmentConfig.model_name to use LaBSE or any other sentence-transformer.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{cache_key}.npy"
    if cache.exists():
        return np.load(cache)

    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

    model = SentenceTransformer(model_name)
    vecs = model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    np.save(cache, vecs)
    return vecs


def _global_dp_align(
    en_vecs: np.ndarray,
    zh_vecs: np.ndarray,
    band: float,
) -> list[tuple[list[int], list[int], float]]:
    """Global DP over all sentences, restricted to a diagonal band.

    (i, j) is reachable only when ``|i/n - j/m| <= band``. This enforces the
    locality assumption (translations preserve sentence order) and prevents
    far-apart false-positive matches. Returns one record per (en_idx_list,
    zh_idx_list) tuple — no duplicates because each sentence index is consumed
    by exactly one transition.
    """
    n = en_vecs.shape[0]
    m = zh_vecs.shape[0]
    if n == 0 or m == 0:
        return []

    sims = en_vecs @ zh_vecs.T  # (n, m)

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

            # 1:0 — en sentence i-1 unaligned (penalized)
            if i >= 1 and dp[i - 1, j] > NEG:
                cand = dp[i - 1, j] - PENALTY_1_TO_0
                if cand > best:
                    best, best_move = cand, (1, 0, -PENALTY_1_TO_0)

            # 0:1 — zh sentence j-1 unaligned (penalized)
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

            # 1:2 — one en, two zh
            if i >= 1 and j >= 2 and dp[i - 1, j - 2] > NEG:
                s = float(max(sims[i - 1, j - 2], sims[i - 1, j - 1]))
                if s >= SIMILARITY_FLOOR:
                    cand = dp[i - 1, j - 2] + s * 0.95
                    if cand > best:
                        best, best_move = cand, (1, 2, s * 0.95)

            # 2:1 — two en, one zh
            if i >= 2 and j >= 1 and dp[i - 2, j - 1] > NEG:
                s = float(max(sims[i - 2, j - 1], sims[i - 1, j - 1]))
                if s >= SIMILARITY_FLOOR:
                    cand = dp[i - 2, j - 1] + s * 0.95
                    if cand > best:
                        best, best_move = cand, (2, 1, s * 0.95)

            # 1:3 — one en, three zh (translator expanded EN into multiple ZH sentences)
            if i >= 1 and j >= 3 and dp[i - 1, j - 3] > NEG:
                s = float(max(sims[i - 1, j - 3], sims[i - 1, j - 2], sims[i - 1, j - 1]))
                if s >= SIMILARITY_FLOOR:
                    cand = dp[i - 1, j - 3] + s * 0.90  # larger discount for 3-way merges
                    if cand > best:
                        best, best_move = cand, (1, 3, s * 0.90)

            # 3:1 — three en, one zh (translator condensed)
            if i >= 3 and j >= 1 and dp[i - 3, j - 1] > NEG:
                s = float(max(sims[i - 3, j - 1], sims[i - 2, j - 1], sims[i - 1, j - 1]))
                if s >= SIMILARITY_FLOOR:
                    cand = dp[i - 3, j - 1] + s * 0.90
                    if cand > best:
                        best, best_move = cand, (3, 1, s * 0.90)

            # 2:2 — two en, two zh (rare but occurs in dialogue-heavy passages)
            if i >= 2 and j >= 2 and dp[i - 2, j - 2] > NEG:
                s = float(max(
                    sims[i - 2, j - 2], sims[i - 2, j - 1],
                    sims[i - 1, j - 2], sims[i - 1, j - 1],
                ))
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
        en_idx = list(range(i - di, i)) if di else []
        zh_idx = list(range(j - dj, j)) if dj else []
        if en_idx or zh_idx:
            matches.append((en_idx, zh_idx, max(0.0, score)))
        i -= di
        j -= dj

    matches.reverse()
    return matches


def align_segments(
    en: list[Segment],
    zh: list[Segment],
    config: AlignmentConfig,
) -> list[Alignment]:
    en_vecs = embed_sentences(
        [s.text for s in en],
        _cache_key("en", config.model_name),
        config.embed_cache_dir,
        config.model_name,
    )
    zh_vecs = embed_sentences(
        [s.text for s in zh],
        _cache_key("zh", config.model_name),
        config.embed_cache_dir,
        config.model_name,
    )

    records: list[Alignment] = []
    align_n = 0
    for en_idx_list, zh_idx_list, score in _global_dp_align(en_vecs, zh_vecs, config.locality_band):
        en_ids = [en[i].id for i in en_idx_list if 0 <= i < len(en)]
        zh_ids = [zh[i].id for i in zh_idx_list if 0 <= i < len(zh)]
        if not en_ids and not zh_ids:
            continue
        align_n += 1
        type_str = f"{len(en_ids)}:{len(zh_ids)}"
        method = "vecalign_labse" if score >= config.manual_threshold else "manual"
        records.append(
            Alignment(
                align_id=f"a{align_n:04d}",
                en=en_ids,
                zh=zh_ids,
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
