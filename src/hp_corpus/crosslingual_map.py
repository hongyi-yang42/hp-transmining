"""Cross-lingual PP mapping for Step 4 (Bremmers reproduction).

For each German PP occurrence (already linked to its aligned EN and ZH
sentence IDs by ``hp_corpus.step4``), this module proposes the most
likely **equivalent PP span** in EN and ZH using POS-aware candidate
extraction from CoNLL-U plus a transparent scoring rule.

Methodology boundaries (see docs/METHODS.md):

  * Output is a **machine proposal**, never a definitive annotation.
    The pilot TSV's human-annotation columns stay blank; this module's
    results live in a separate JSONL so a human annotator can compare.
  * Scoring is rule-based and deterministic: prep-semantics match,
    proper-name overlap, position-in-sentence, and a small bonus from
    sentence-alignment confidence. No neural translation, no embeddings.
  * When no candidate clears the threshold, the mapping is recorded as
    ``unmappable`` with a reason — never silently dropped.

The CoNLL-U walk relies only on the parts of UD that are stable across
Stanza outputs: token id, form, lemma, upos, head, deprel. MWT range
lines (e.g. ``5-6``) are skipped; only single-token rows participate.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

# --------------------------------------------------------------------- types


@dataclass(frozen=True)
class Token:
    """Minimal CoNLL-U token view used by the matcher."""

    token_id: int  # 1-indexed, post-MWT-collapse
    form: str
    lemma: str
    upos: str
    head: int  # 0 means root
    deprel: str


@dataclass(frozen=True)
class Sentence:
    """A CoNLL-U sentence keyed by its UD sent_id."""

    sent_id: str
    text: str
    tokens: tuple[Token, ...]


@dataclass(frozen=True)
class PPElement:
    """A PP candidate extracted from a sentence.

    The span covers every token from the preposition through the head
    noun's last contiguous dependent — modifiers (det, amod, nummod,
    compound, appos, flat, acl, advmod) that sit between prep and head
    or immediately after the head are included.
    """

    sent_id: str
    prep_surface: str
    prep_lemma: str
    prep_token_id: int
    head_surface: str
    head_lemma: str
    head_token_id: int
    head_upos: str
    span_text: str
    span_token_ids: tuple[int, ...]
    char_start: int  # offset into Sentence.text
    char_end: int  # exclusive


@dataclass(frozen=True)
class ScoredCandidate:
    """An EN/ZH PP candidate paired with its match score and rationale."""

    pp: PPElement
    score: float
    components: tuple[tuple[str, float], ...]  # (signal_name, contribution)


@dataclass
class MappingResult:
    """The proposal for one side (EN or ZH) of one DE PP occurrence."""

    status: str  # 'matched' | 'candidate' | 'unmappable'
    best: PPElement | None = None
    best_score: float = 0.0
    components: tuple[tuple[str, float], ...] = ()
    alternatives: list[ScoredCandidate] = field(default_factory=list)
    n_candidates_considered: int = 0
    reason: str = ""  # populated when status == 'unmappable'


# --------------------------------------------------------------------- constants

# Preposition semantics — sets of EN surface ADP forms that can realize
# each canonical German preposition. Used for prep_semantic_match.
# Built from the project's contraction normalization table plus common
# English realization patterns observed in HP1 Ch.1–3.
DE_EN_PREP_MAP: dict[str, frozenset[str]] = {
    "in": frozenset({"in", "into", "inside", "within", "at", "on"}),
    "an": frozenset({"at", "on", "upon"}),
    "bei": frozenset({"at", "by", "near", "with"}),
    "zu": frozenset({"to", "towards", "for"}),
    "von": frozenset({"from", "of", "off"}),
    "auf": frozenset({"on", "upon", "onto"}),
    "durch": frozenset({"through", "via", "by"}),
    "für": frozenset({"for"}),
    "hinter": frozenset({"behind", "after"}),
    "über": frozenset({"over", "above", "beyond"}),
    "um": frozenset({"around", "at", "about"}),
    "unter": frozenset({"under", "below", "beneath", "among"}),
    "vor": frozenset({"before", "in", "ago", "ahead"}),
    "gegen": frozenset({"against", "for", "around"}),
}

# ZH preposition/coverb forms that can realize each German preposition.
# Chinese has a smaller closed set; one coverb often covers several
# German prepositions, so matching is coarser than EN.
DE_ZH_PREP_MAP: dict[str, frozenset[str]] = {
    "in": frozenset({"在", "于", "里", "中"}),
    "an": frozenset({"在", "于"}),
    "bei": frozenset({"在", "于"}),
    "zu": frozenset({"到", "向", "朝"}),
    "von": frozenset({"从", "由", "的"}),
    "auf": frozenset({"在", "上"}),
    "durch": frozenset({"通过", "穿", "经"}),
    "für": frozenset({"为", "给"}),
    "hinter": frozenset({"后", "在"}),
    "über": frozenset({"超", "过", "在"}),
    "um": frozenset({"围绕", "为"}),
    "unter": frozenset({"下", "在"}),
    "vor": frozenset({"前", "在"}),
    "gegen": frozenset({"对", "朝", "向"}),
}

# Modifiers that extend a PP head noun's span (UD deprels).
# Conservative: only true nominal modifiers are included. Clausal
# modifiers (acl, acl:relcl, advmod, nmod, case) are deliberately
# excluded — they can pull in arbitrary-length sub-phrases that distort
# the span (e.g. "the man who lived in the cupboard" would balloon a
# simple "in the man" span into a full clause).
_HEAD_MODIFIER_DEPRELS: frozenset[str] = frozenset(
    {
        "det",
        "amod",
        "nummod",
        "compound",
        "compound:prt",
        "appos",
        "flat",
        "flat:name",
        "nmod:poss",  # possessive determiner (his/her/their) — small, safe
    }
)

# UD UPOS tags that count as "nominal" for matching against a German
# proper-name head — used to surface proper-name overlap as a signal.
_PROPER_UPOS: frozenset[str] = frozenset({"PROPN"})

# Match-status thresholds — see MappingResult.status.
MATCH_THRESHOLD: float = 5.0  # >= → status='matched'
CANDIDATE_THRESHOLD: float = 2.0  # >= → status='candidate'

# Per-language alignment-confidence bonus thresholds. LaBSE scores are
# systematically lower for cross-script pairs (DE↔ZH) than same-script
# (DE↔EN), so a single 0.9 cutoff would never fire for ZH. These cutoffs
# sit near the upper quartile of observed confidences in Ch.1–3:
#   DE↔EN: median ≈ 0.91, top quartile > 0.93 → 0.92
#   DE↔ZH: median ≈ 0.81, top quartile > 0.84 → 0.84
ALIGN_CONFIDENCE_BONUS: dict[str, float] = {
    "en": 0.92,
    "zh": 0.84,
}

# Signal weights. These are the only knobs in the scorer; keep them
# explicit and named so future tuning is one place.
W_PREP_SEM: float = 3.0  # ADP semantically maps to DE canonical prep
W_PROPER_NAME: float = 5.0  # PROPN token-form overlap (case-insensitive)
W_LEMMA_COVER: float = 2.0  # non-proper lemma overlap (e.g. "Gedanke"/"thought")
W_POSITION: float = 2.0  # same ordinal position among PPs in the sentence
W_ALIGN_BONUS: float = 1.0  # sentence-level alignment confidence > 0.9

# ID pattern — second underscore-separated field is the language tag.
_SENT_ID_LANG_RE = re.compile(r"^[a-z0-9]+_([a-z]{2,3})_ch\d{2}_p\d{4}_s\d{3}$")

# Range-line detector (e.g. "5-6" for German MWT).
_RANGE_ID_RE = re.compile(r"^\d+-\d+$")


# --------------------------------------------------------------------- parsing


def _is_range_id(token_id: str | None) -> bool:
    return bool(token_id) and bool(_RANGE_ID_RE.match(token_id))


def parse_conllu(path) -> dict[str, Sentence]:
    """Parse a CoNLL-U file into a {sent_id → Sentence} map.

    Implements just enough of the format for our needs: comment lines
    starting with ``#`` capture ``sent_id`` and ``text``; token lines
    follow the 10-column TSV. MWT range lines are skipped. With pyconll
    available, callers may prefer it; this parser keeps the module
    dependency-free for unit tests.
    """
    out: dict[str, Sentence] = {}
    current_sent_id: str | None = None
    current_text: str = ""
    current_tokens: list[Token] = []

    def _flush() -> None:
        nonlocal current_sent_id, current_text, current_tokens
        if current_sent_id is not None:
            out[current_sent_id] = Sentence(
                sent_id=current_sent_id,
                text=current_text,
                tokens=tuple(current_tokens),
            )
        current_sent_id = None
        current_text = ""
        current_tokens = []

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                _flush()
                continue
            if line.startswith("#"):
                m_sent = re.match(r"^# sent_id\s*=\s*(\S+)", line)
                if m_sent:
                    current_sent_id = m_sent.group(1)
                m_text = re.match(r"^# text\s*=\s*(.*)$", line)
                if m_text:
                    current_text = m_text.group(1)
                continue
            cols = line.split("\t")
            if len(cols) < 8:
                continue
            tid_str = cols[0]
            if _is_range_id(tid_str):
                continue
            try:
                tid = int(tid_str)
                head = int(cols[6]) if cols[6] else 0
            except ValueError:
                continue
            current_tokens.append(
                Token(
                    token_id=tid,
                    form=cols[1],
                    lemma=cols[2],
                    upos=cols[3],
                    head=head,
                    deprel=cols[7],
                )
            )
        _flush()
    return out


# --------------------------------------------------------------------- extraction


def _build_token_positions(sentence: Sentence) -> dict[int, tuple[int, int]]:
    """Walk ``sentence.text`` and assign ``(start, end_exclusive)`` char
    positions to every token, in token-id ascending order.

    Each token's form is searched starting at the previous token's end,
    so duplicate forms (e.g. two ``of``s in one sentence) resolve to
    distinct positions in surface order. Falls back to plain substring
    search when word-boundary search fails (Chinese has no word
    boundaries; rare Stanza oddities emit forms that don't match the
    text verbatim).
    """
    text = sentence.text
    positions: dict[int, tuple[int, int]] = {}
    cursor = 0
    for tok in sentence.tokens:  # tokens are stored in id-ascending order
        if not tok.form:
            continue
        m = re.search(r"\b" + re.escape(tok.form) + r"\b", text[cursor:])
        if m is not None:
            start = cursor + m.start()
        else:
            idx = text.find(tok.form, cursor)
            if idx < 0:
                # Skip — leaving this token without a position. Downstream
                # callers treat missing positions as "no span".
                continue
            start = idx
        end = start + len(tok.form)
        positions[tok.token_id] = (start, end)
        cursor = end
    return positions


def _char_span_for_tokens(sentence: Sentence, token_ids: Iterable[int]) -> tuple[int, int]:
    """``(start, end_exclusive)`` char range in ``sentence.text`` for the
    given token ids. Uses a sentence-wide token-to-position map so
    duplicate forms resolve to distinct positions in surface order.
    Returns ``(-1, -1)`` if no requested token has a known position.
    """
    sorted_ids = sorted({t for t in token_ids if t > 0})
    if not sorted_ids:
        return (-1, -1)
    positions = _build_token_positions(sentence)
    spans = [positions[tid] for tid in sorted_ids if tid in positions]
    if not spans:
        return (-1, -1)
    return (spans[0][0], spans[-1][1])


def extract_pps(sentence: Sentence) -> list[PPElement]:
    """Extract every PP candidate from a parsed sentence.

    A PP candidate is an ``ADP`` token whose deprel is ``case`` (the UD
    marker for prepositional case-marking); its head is the noun it
    attaches to. The span extends from the ADP through the head noun
    plus any contiguous modifiers of the head (det / amod / nummod /
    compound / appos / flat / acl / advmod / nmod / case).

    Returns PPs in sentence order. Empty list if the sentence has no
    ADP+case construction.
    """
    if not sentence.tokens:
        return []
    by_id = {t.token_id: t for t in sentence.tokens}
    pps: list[PPElement] = []
    for tok in sentence.tokens:
        if tok.upos != "ADP" or tok.deprel != "case":
            continue
        head_id = tok.head
        head = by_id.get(head_id)
        if head is None:
            continue
        # Collect the ADP, the head, and any modifier of the head that
        # sits within the [prep_token_id, head_token_id] window OR right
        # after the head.
        window_ids: set[int] = {tok.token_id, head.token_id}
        for other in sentence.tokens:
            if other.token_id in window_ids:
                continue
            if other.head == head.token_id and other.deprel in _HEAD_MODIFIER_DEPRELS:
                window_ids.add(other.token_id)
        # Build the surface text from the contiguous run that contains
        # the ADP and head, extended by any included modifiers.
        span_ids = sorted(window_ids)
        # Drop trailing tokens that are pure punctuation so the span
        # text reads cleanly (e.g. "of number four, Privet Drive" rather
        # than "of number four, Privet Drive,").
        while span_ids and by_id[span_ids[-1]].upos == "PUNCT":
            span_ids.pop()
        if not span_ids:
            continue
        # Re-add the head if we stripped too much punctuation.
        if head.token_id not in span_ids and tok.token_id not in span_ids:
            continue
        span_text = _surface_from_ids(sentence, span_ids)
        char_start, char_end = _char_span_for_tokens(sentence, span_ids)
        pps.append(
            PPElement(
                sent_id=sentence.sent_id,
                prep_surface=tok.form,
                prep_lemma=tok.lemma,
                prep_token_id=tok.token_id,
                head_surface=head.form,
                head_lemma=head.lemma,
                head_token_id=head.token_id,
                head_upos=head.upos,
                span_text=span_text,
                span_token_ids=tuple(span_ids),
                char_start=char_start,
                char_end=char_end,
            )
        )
    return pps


def _surface_from_ids(sentence: Sentence, ids: list[int]) -> str:
    """Reconstruct the surface form by walking tokens in id order, joining
    with single spaces. Skips punctuation and empty forms.

    This deliberately produces a normalized, machine-readable form rather
    than the verbatim substring of Sentence.text — callers who need the
    verbatim substring use ``char_start`` / ``char_end`` from
    ``_char_span_for_tokens``.
    """
    by_id = {t.token_id: t for t in sentence.tokens}
    parts: list[str] = []
    prev_id: int | None = None
    for tid in ids:
        tok = by_id.get(tid)
        if tok is None or not tok.form:
            continue
        if tok.upos == "PUNCT":
            # Include only inner punctuation that sits between content
            # tokens (e.g. "Privet, Drive"). Trailing punct already
            # stripped upstream; we still emit a single-token form.
            parts.append(tok.form)
            prev_id = tid
            continue
        if prev_id is not None:
            parts.append(" ")
        parts.append(tok.form)
        prev_id = tid
    return "".join(parts).strip()


# --------------------------------------------------------------------- scoring


def _normalize_for_compare(s: str) -> str:
    return s.strip().lower()


def _proper_name_overlap(de_pp_text: str, cand_pp: PPElement) -> set[str]:
    """Return the set of case-insensitive proper-name tokens shared
    between the DE PP surface and the candidate's head + modifiers.

    We treat any token whose surface starts with an uppercase letter in
    the DE side, OR whose UPOS is PROPN on the candidate side, as a
    proper-name candidate.
    """
    de_tokens = {
        _normalize_for_compare(t)
        for t in re.split(r"\s+", de_pp_text)
        if t and t[0].isupper()
    }
    cand_proper = set()
    if cand_pp.head_upos == "PROPN":
        cand_proper.add(_normalize_for_compare(cand_pp.head_surface))
    # Also surface proper nouns embedded in the candidate span text.
    cand_proper |= {
        _normalize_for_compare(t)
        for t in re.split(r"\s+", cand_pp.span_text)
        if t and t[0].isupper()
    }
    return de_tokens & cand_proper


def _lemma_overlap(de_head_lemma: str, cand_pp: PPElement) -> bool:
    """True if the candidate's head lemma contains the (lowercased) DE
    head lemma or vice versa. Useful for cognates like "Haus"/"house".
    """
    a = _normalize_for_compare(de_head_lemma)
    b = _normalize_for_compare(cand_pp.head_lemma)
    if not a or not b:
        return False
    return a in b or b in a


def score_candidate(
    *,
    de_prep_normalized: str,
    de_pp_surface: str,
    de_head_lemma: str,
    cand_pp: PPElement,
    cand_prep_surface: str,
    prep_map: dict[str, frozenset[str]],
    position_de: int,
    n_de_pps: int,
    position_cand: int,
    n_cand_pps: int,
    align_confidence: float,
    side: str,  # 'en' or 'zh' — picks the right confidence threshold
) -> ScoredCandidate:
    """Score one EN/ZH PP candidate against the DE PP.

    Position terms use 1-indexed PP ordinal in its sentence. When either
    side has only one PP, position_match is automatic.
    """
    components: list[tuple[str, float]] = []

    # --- prep semantic match ---
    expected_forms = prep_map.get(de_prep_normalized, frozenset())
    if cand_prep_surface.lower() in expected_forms:
        components.append(("prep_semantic_match", W_PREP_SEM))

    # --- proper-name overlap ---
    proper_overlap = _proper_name_overlap(de_pp_surface, cand_pp)
    if proper_overlap:
        components.append(("proper_name_overlap", W_PROPER_NAME))

    # --- non-proper lemma overlap (cognates) ---
    if _lemma_overlap(de_head_lemma, cand_pp):
        components.append(("lemma_overlap", W_LEMMA_COVER))

    # --- position match ---
    if n_de_pps == 1 or n_cand_pps == 1:
        # Single-PP side → automatic position match.
        components.append(("position_single_side", W_POSITION))
    elif position_de == position_cand:
        components.append(("position_match", W_POSITION))

    # --- alignment confidence bonus (per-side threshold) ---
    threshold = ALIGN_CONFIDENCE_BONUS.get(side, 0.9)
    if align_confidence > threshold:
        components.append(("align_confidence_bonus", W_ALIGN_BONUS))

    score = sum(w for _, w in components)
    return ScoredCandidate(pp=cand_pp, score=score, components=tuple(components))


def select_best(
    de_prep_normalized: str,
    de_pp_surface: str,
    de_head_lemma: str,
    candidates: list[PPElement],
    prep_map: dict[str, frozenset[str]],
    align_confidence: float,
    side: str,  # 'en' or 'zh'
) -> MappingResult:
    """Pick the highest-scoring EN/ZH candidate; classify the result.

    Threshold semantics:
      * score >= MATCH_THRESHOLD (5.0)  → status='matched'
      * score >= CANDIDATE_THRESHOLD (2.0) → status='candidate'
      * else, or zero candidates        → status='unmappable'
    """
    if not candidates:
        return MappingResult(
            status="unmappable",
            n_candidates_considered=0,
            reason="no_pp_in_target_sentence",
        )

    scored: list[ScoredCandidate] = []
    for i, cand in enumerate(candidates):
        # The candidate's prep surface is its own ADP form (e.g. "of",
        # "在"). We pass it explicitly because prep_map keys are DE-side
        # canonical prepositions, not EN/ZH surface forms.
        s = score_candidate(
            de_prep_normalized=de_prep_normalized,
            de_pp_surface=de_pp_surface,
            de_head_lemma=de_head_lemma,
            cand_pp=cand,
            cand_prep_surface=cand.prep_surface,
            prep_map=prep_map,
            position_de=1,
            n_de_pps=1,
            position_cand=i + 1,
            n_cand_pps=len(candidates),
            align_confidence=align_confidence,
            side=side,
        )
        scored.append(s)
    scored.sort(key=lambda s: s.score, reverse=True)

    best = scored[0]
    alternatives = scored[1:4]  # keep top 3 alternatives

    if best.score >= MATCH_THRESHOLD:
        status = "matched"
    elif best.score >= CANDIDATE_THRESHOLD:
        status = "candidate"
    else:
        return MappingResult(
            status="unmappable",
            best=best.pp,
            best_score=best.score,
            components=best.components,
            alternatives=alternatives,
            n_candidates_considered=len(candidates),
            reason="best_score_below_threshold",
        )
    return MappingResult(
        status=status,
        best=best.pp,
        best_score=best.score,
        components=best.components,
        alternatives=alternatives,
        n_candidates_considered=len(candidates),
    )


# --------------------------------------------------------------------- aggregation


def propose_for_sides(
    *,
    de_prep_normalized: str,
    de_pp_surface: str,
    de_head_lemma: str,
    en_sentences: list[Sentence],
    zh_sentences: list[Sentence],
    en_align_confidence: float,
    zh_align_confidence: float,
) -> tuple[MappingResult, MappingResult]:
    """Run the matcher for the EN side and the ZH side independently.

    Returns ``(en_result, zh_result)`` so callers can serialize each
    side's proposal in parallel fields.
    """
    en_pps: list[PPElement] = []
    for s in en_sentences:
        en_pps.extend(extract_pps(s))
    zh_pps: list[PPElement] = []
    for s in zh_sentences:
        zh_pps.extend(extract_pps(s))

    en_result = select_best(
        de_prep_normalized=de_prep_normalized,
        de_pp_surface=de_pp_surface,
        de_head_lemma=de_head_lemma,
        candidates=en_pps,
        prep_map=DE_EN_PREP_MAP,
        align_confidence=en_align_confidence,
        side="en",
    )
    zh_result = select_best(
        de_prep_normalized=de_prep_normalized,
        de_pp_surface=de_pp_surface,
        de_head_lemma=de_head_lemma,
        candidates=zh_pps,
        prep_map=DE_ZH_PREP_MAP,
        align_confidence=zh_align_confidence,
        side="zh",
    )
    return en_result, zh_result


__all__ = [
    "ALIGN_CONFIDENCE_BONUS",
    "DE_EN_PREP_MAP",
    "DE_ZH_PREP_MAP",
    "CANDIDATE_THRESHOLD",
    "MATCH_THRESHOLD",
    "MappingResult",
    "PPElement",
    "ScoredCandidate",
    "Sentence",
    "Token",
    "extract_pps",
    "parse_conllu",
    "propose_for_sides",
    "score_candidate",
    "select_best",
]
