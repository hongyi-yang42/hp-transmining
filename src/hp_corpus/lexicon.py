"""Bilingual lexical anchors for the alignment prior.

Sentence-level embedding similarity (multilingual-e5) is not discriminative
on short dialogue units — quotes, attributions, scene narrations — which is
where the residual DE–ZH alignment errors concentrate (crossed 1:1 pairings,
±1–2 local shifts). Proper nouns and distinctive numbers translate
deterministically, so their co-occurrence is a strong, cheap pairing signal
that embeddings lack.

The dictionary maps the German (Klaus Fritz) rendering to the Mandarin
(Su Nong) rendering of the same entity. Entries are canonical anchor IDs
(the German key); ``extract_anchors`` matches either side's surface form,
so language pairs that share a spelling with either side (e.g. EN "Voldemort"
matches the German key verbatim) pick up the anchors incidentally.

Weights are IDF-style: an anchor's contribution is scaled by how rare it is
across the sentences being aligned, so ubiquitous names (Harry/哈利, df ≈ 24%)
nudge while rare ones (Firenze/费伦泽, df < 1%) anchor decisively.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

# Corpus-validated against Ch.1–17 segmented text (both sides must occur).
# Zero-hit and generic entries were pruned.
DE_ZH_LEXICON: dict[str, str] = {
    # people
    "Harry": "哈利",
    "Ron": "罗恩",
    "Hermine": "赫敏",
    "Hagrid": "海格",
    "Dumbledore": "邓布利多",
    "McGonagall": "麦格",
    "Snape": "斯内普",
    "Malfoy": "马尔福",
    "Neville": "纳威",
    "Fred": "弗雷德",
    "George": "乔治",
    "Percy": "珀西",
    "Ginny": "金妮",
    "Quirrell": "奇洛",
    "Filch": "费尔奇",
    "Flitwick": "弗立维",
    "Norbert": "诺伯",
    "Griphook": "拉环",
    "Ollivander": "奥利凡德",
    "Hedwig": "海德薇",
    "Fang": "牙牙",
    "Firenze": "费伦泽",
    "Bane": "贝恩",
    "Ronan": "罗南",
    "Flamel": "勒梅",
    "Nicolas": "尼可",
    "Voldemort": "伏地魔",
    "Dudley": "达力",
    "Petunia": "佩妮",
    "Vernon": "弗农",
    "Piers": "皮尔",
    "Pomfrey": "庞弗雷",
    "Hooch": "霍琦",
    "Wood": "伍德",
    "Charlie": "查理",
    "Bill": "比尔",
    "Marge": "玛姬",
    "Norris": "洛丽丝",
    "Trevor": "莱福",
    "Crabbe": "克拉布",
    "Goyle": "高尔",
    "Dean": "迪安",
    "Seamus": "西莫",
    "Oliver": "奥利弗",
    "Alicia": "艾丽娅",
    "Angelina": "安吉利娜",
    "Katie": "凯蒂",
    "Marcus": "马库斯",
    "Flint": "弗林特",
    "Jordan": "乔丹",
    # places / institutions
    "Hogwarts": "霍格沃茨",
    "Gryffindor": "格兰芬多",
    "Slytherin": "斯莱特林",
    "Hufflepuff": "赫奇帕奇",
    "Ravenclaw": "拉文克劳",
    "Winkelgasse": "对角巷",
    "Gringotts": "古灵阁",
    "Ligusterweg": "女贞路",
    "King": "国王十字",
    # objects / concepts / currency
    "Nimbus": "光轮",
    "Muggel": "麻瓜",
    "Du-weißt-schon-wer": "神秘人",
    "Nerhegeb": "厄里斯",
    "Galleone": "加隆",
    "Sickel": "西可",
}

# Distinctive numbers: multi-digit sequences, decimals, and fractions like
# 3/4. Lone digits (9, 4) are too common to anchor anything.
_DIGIT_ANCHOR_RE = re.compile(r"\d+(?:[.,/]\d+)*")


def extract_anchors(text: str, lexicon: dict[str, str] = DE_ZH_LEXICON) -> set[str]:
    """Anchor IDs present in one sentence (either language's surface form)."""
    found = set()
    for de_key, zh_val in lexicon.items():
        if de_key in text or zh_val in text:
            found.add(de_key)
    for num in _DIGIT_ANCHOR_RE.findall(text):
        if len(num) >= 2 or "/" in num:
            found.add(num)
    return found


def anchor_weights(
    anchor_sets_per_sentence: list[set[str]],
) -> dict[str, float]:
    """IDF-normalized weight per anchor, in [0, 1].

    ``w = log(N / df) / log(N)`` over the sentences being aligned (both
    sides pooled). df = 1 → w = 1 (maximally rare); an anchor in every
    sentence → w → 0. N is capped at 2 so tiny inputs still give sane
    weights (log(N/df)/log(N) is degenerate for N == 1).
    """
    n = max(2, len(anchor_sets_per_sentence))
    df: Counter[str] = Counter()
    for anchors in anchor_sets_per_sentence:
        df.update(anchors)
    log_n = math.log(n)
    return {
        anchor: min(1.0, math.log(n / count) / log_n) if count < n else 0.0
        for anchor, count in df.items()
    }


def lexical_bonus_matrix(
    src_anchor_sets: list[set[str]],
    tgt_anchor_sets: list[set[str]],
    weights: dict[str, float],
    *,
    weight: float,
    cap: float,
) -> np.ndarray:
    """(n, m) additive bonus: capped weighted sum of anchors shared by the
    two sentences. Callers add this to the cosine-similarity matrix before
    the DP; blocks consume it via the mean, so N:M groups benefit too."""
    n, m = len(src_anchor_sets), len(tgt_anchor_sets)
    bonus = np.zeros((n, m), dtype=np.float64)
    for i, src_anchors in enumerate(src_anchor_sets):
        if not src_anchors:
            continue
        for j, tgt_anchors in enumerate(tgt_anchor_sets):
            shared = src_anchors & tgt_anchors
            if shared:
                bonus[i, j] = min(cap, weight * sum(weights[a] for a in shared))
    return bonus
