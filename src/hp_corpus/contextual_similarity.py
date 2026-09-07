"""One local contextual cross-lingual similarity signal (pilot).

The minimal SimAlign equivalent, on the corpus-canonical local encoder
LaBSE (``models/LaBSE``, already used by the production alignment and
judge-audited for this corpus — see ``docs/ALIGNMENT_MODEL_DECISION.md``):
word-level contextual vectors from the frozen encoder (subword
mean-pooling, exactly SimAlign's representation strategy), cosine
similarity between the German PP tokens and each target candidate
span, deterministic on CPU fp32. No fine-tuning, no model zoo, no
generative API, no GLM Coding Plan usage.

This is NOT a word aligner — it ranks a small parse-derived candidate
set inside an already sentence-aligned context. It stays a separate,
inspectable signal from eflomal (see
``hp_corpus.constituent_candidates.route_side``).
"""

from __future__ import annotations

CONTEXTUAL_MODEL_NAME = "LaBSE"
CONTEXTUAL_MODEL_PATH = "models/LaBSE"
CONTEXTUAL_STRATEGY = "frozen-encoder-word-vectors-subword-mean-pool-cosine"
CONTEXTUAL_DEVICE = "cpu"


def config_summary() -> dict[str, str]:
    """Inference configuration recorded beside every artifact."""
    info: dict[str, str] = {
        "model": CONTEXTUAL_MODEL_NAME,
        "model_path": CONTEXTUAL_MODEL_PATH,
        "strategy": CONTEXTUAL_STRATEGY,
        "device": CONTEXTUAL_DEVICE,
        "dtype": "float32",
        "fine_tuned": "no",
    }
    try:
        import sentence_transformers

        info["sentence_transformers"] = sentence_transformers.__version__
    except ImportError:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
    except ImportError:
        pass
    return info


class ContextualSimilarity:
    """Word-level contextual vectors from the frozen LaBSE encoder."""

    def __init__(self, model_path: str = CONTEXTUAL_MODEL_PATH):
        from pathlib import Path

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"contextual encoder not found at {model_path} "
                "(expected the local LaBSE checkout)"
            )
        import sentence_transformers

        self._model = sentence_transformers.SentenceTransformer(
            model_path, device=CONTEXTUAL_DEVICE
        )

    def word_vectors(self, tokens: list[str]) -> list[list[float]]:
        """Contextual vector per token: mean over the token's subword
        vectors from the frozen encoder (in-sentence context)."""
        if not tokens:
            return []
        tokenizer = self._model.tokenizer
        encoder = self._model[0].auto_model
        enc = tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
        )
        hidden = encoder(**enc).last_hidden_state[0]
        word_ids = enc.word_ids()
        out: list[list[float]] = []
        for i in range(len(tokens)):
            positions = [j for j, w in enumerate(word_ids) if w == i]
            if not positions:
                # token produced no subword (over-truncation): zero vec
                out.append([0.0] * hidden.shape[1])
                continue
            vec = hidden[positions[0]]
            for j in positions[1:]:
                vec = vec + hidden[j]
            vec = vec / len(positions)
            out.append(vec.tolist())
        return out


def cosine(a: list[float], b: list[float]) -> float:
    """Deterministic plain-Python cosine (CPU fp32 vectors are small;
    avoids any BLAS reduction-order nondeterminism)."""
    num = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return num / (na * nb)


def mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            out[i] += x
    n = float(len(vectors))
    return [x / n for x in out]


def rank_candidates(
    de_word_vectors: list[list[float]],
    pp_slice: range,
    tgt_word_vectors: list[list[float]],
    candidates: list,
) -> dict[int, float]:
    """Cosine similarity between the German PP token mean vector and
    each candidate's span-token mean vector. ``candidates`` are
    ``constituent_candidates.Candidate`` objects (lo/hi index into the
    target layout). Pure function of the injected vectors — the routing
    policy and its tests never need the encoder."""
    query = mean_vector([de_word_vectors[i] for i in pp_slice])
    out: dict[int, float] = {}
    for idx, cand in enumerate(candidates):
        span_vecs = [tgt_word_vectors[i] for i in range(cand.lo, cand.hi + 1)]
        out[idx] = round(cosine(query, mean_vector(span_vecs)), 6)
    return out


__all__ = [
    "CONTEXTUAL_MODEL_NAME",
    "CONTEXTUAL_MODEL_PATH",
    "CONTEXTUAL_STRATEGY",
    "CONTEXTUAL_DEVICE",
    "config_summary",
    "ContextualSimilarity",
    "cosine",
    "mean_vector",
    "rank_candidates",
]
