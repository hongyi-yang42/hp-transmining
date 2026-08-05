"""EN–ZH sentence alignment via LaBSE embeddings + DP.

The DP supports the standard Vecalign alignment types (1:0, 0:1, 1:1, 1:2, 2:1)
over precomputed LaBSE cosine similarities. The vendored ``vendor/vecalign``
checkout is not required — this module re-implements the DP inline so the
project has no hard dependency on the upstream code. If you want to use the
upstream Vecalign for cross-checking, clone it per the README and call it
separately; results from this module are produced by the inline DP.

To keep the search tractable for long chapters, we first group sentences by
paragraph and only consider cross-language paragraph pairs whose mean-vector
cosine similarity is in the top-K per source paragraph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .schema import Alignment, Segment

MANUAL_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_TOP_K_PARAGRAPHS = 3
# Penalties for non-1:1 alignments (mimic Vecalign's deletion/insertion costs).
PENALTY_1_TO_0 = 0.2
PENALTY_0_TO_1 = 0.2
# Sentence-similarity floor; below this, prefer to leave both sides unaligned.
SIMILARITY_FLOOR = 0.3


@dataclass
class AlignmentConfig:
    embed_cache_dir: Path
    vecalign_dir: Path | None = None  # reserved for future subprocess use; currently unused
    top_k_paragraphs: int = DEFAULT_TOP_K_PARAGRAPHS
    manual_threshold: float = MANUAL_CONFIDENCE_THRESHOLD
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


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


def _paragraph_group(segments: list[Segment]) -> list[tuple[int, list[Segment]]]:
    """Group segments by paragraph, preserving paragraph order."""
    by_para: dict[int, list[Segment]] = {}
    para_order: list[int] = []
    for s in segments:
        if s.paragraph not in by_para:
            by_para[s.paragraph] = []
            para_order.append(s.paragraph)
        by_para[s.paragraph].append(s)
    return [(p, by_para[p]) for p in para_order]


def _best_paragraph_pairs(
    en_paras: list[tuple[int, list[Segment]]],
    zh_paras: list[tuple[int, list[Segment]]],
    en_vecs: np.ndarray,
    zh_vecs: np.ndarray,
    en_offset: list[int],
    zh_offset: list[int],
    top_k: int,
) -> list[tuple[int, int]]:
    """Return (en_para_idx, zh_para_idx) candidate pairs."""
    pairs: list[tuple[int, int]] = []
    for i, (_, en_segs) in enumerate(en_paras):
        i0 = en_offset[i]
        i1 = i0 + len(en_segs)
        if i1 == i0:
            continue
        en_mean = en_vecs[i0:i1].mean(axis=0)
        sims = []
        for j, (_, zh_segs) in enumerate(zh_paras):
            j0 = zh_offset[j]
            j1 = j0 + len(zh_segs)
            if j1 == j0:
                continue
            zh_mean = zh_vecs[j0:j1].mean(axis=0)
            sims.append((j, float(np.dot(en_mean, zh_mean))))
        sims.sort(key=lambda t: t[1], reverse=True)
        for j, _ in sims[:top_k]:
            pairs.append((i, j))
    return pairs


def _dp_align(en_vecs: np.ndarray, zh_vecs: np.ndarray) -> list[tuple[list[int], list[int], float]]:
    """DP over the (i, j) lattice. Transitions: 1:0, 0:1, 1:1, 1:2, 2:1.

    Returns list of (en_idx_list, zh_idx_list, score). Indices are local to
    the passed-in vec arrays.
    """
    n = en_vecs.shape[0]
    m = zh_vecs.shape[0]
    if n == 0 or m == 0:
        return []

    sims = en_vecs @ zh_vecs.T  # (n, m)

    # dp[i, j] = best total score aligning en[:i] with zh[:j]
    # back[i, j] = (di, dj, score_delta, type) used to reach this cell
    NEG = -1e9
    dp = np.full((n + 1, m + 1), NEG, dtype=np.float64)
    back: list[list[tuple[int, int, float] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0, 0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
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
                    cand = dp[i - 1, j - 2] + s * 0.95  # tiny discount for merges
                    if cand > best:
                        best, best_move = cand, (1, 2, s * 0.95)

            # 2:1 — two en, one zh
            if i >= 2 and j >= 1 and dp[i - 2, j - 1] > NEG:
                s = float(max(sims[i - 2, j - 1], sims[i - 1, j - 1]))
                if s >= SIMILARITY_FLOOR:
                    cand = dp[i - 2, j - 1] + s * 0.95
                    if cand > best:
                        best, best_move = cand, (2, 1, s * 0.95)

            if best_move is not None:
                dp[i, j] = best
                back[i][j] = best_move

    # Walk back from (n, m)
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
    en_paras = _paragraph_group(en)
    zh_paras = _paragraph_group(zh)

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

    en_offset: list[int] = []
    acc = 0
    for _, segs in en_paras:
        en_offset.append(acc)
        acc += len(segs)
    zh_offset: list[int] = []
    acc = 0
    for _, segs in zh_paras:
        zh_offset.append(acc)
        acc += len(segs)

    candidate_pairs = _best_paragraph_pairs(
        en_paras,
        zh_paras,
        en_vecs,
        zh_vecs,
        en_offset,
        zh_offset,
        top_k=config.top_k_paragraphs,
    )

    seen_pairs: set[tuple[int, int]] = set()
    records: list[Alignment] = []
    align_n = 0
    for en_para_i, zh_para_i in candidate_pairs:
        if (en_para_i, zh_para_i) in seen_pairs:
            continue
        seen_pairs.add((en_para_i, zh_para_i))
        _, en_segs = en_paras[en_para_i]
        _, zh_segs = zh_paras[zh_para_i]
        en_start = en_offset[en_para_i]
        zh_start = zh_offset[zh_para_i]
        en_sub = en_vecs[en_start : en_start + len(en_segs)]
        zh_sub = zh_vecs[zh_start : zh_start + len(zh_segs)]

        for en_idx_list, zh_idx_list, score in _dp_align(en_sub, zh_sub):
            en_ids = [en_segs[i].id for i in en_idx_list if 0 <= i < len(en_segs)]
            zh_ids = [zh_segs[i].id for i in zh_idx_list if 0 <= i < len(zh_segs)]
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
