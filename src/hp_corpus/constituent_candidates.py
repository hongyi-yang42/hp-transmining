"""Constituent-constrained deterministic counterpart recovery (pilot v2).

Reframes the deterministic baseline from ``word links → token hull`` to

    known German PP
    → eflomal lexical anchor evidence
    → Stanza target constituent candidates
    → cross-lingual candidate scoring
    → one defensible counterpart or an explicit unresolved state
    → deterministic paper-form classification

Conceptual roles: eflomal supplies corpus-specific cross-lingual
lexical evidence; the Stanza parses supply target-language constituent
structure; a separate contextual-similarity signal
(``hp_corpus.contextual_similarity``) supplies an independent semantic
ranking; deterministic rules assign the observable form. Nothing here
is generative, and no second word aligner is introduced.

Constituent boundary convention (explicit, tested): the analysis
counterpart is the **nominal expression** — head noun plus its local
nominal dependents — with the external adposition kept OUT. The outer
adposition belongs to the PP surface, not to the reference expression;
the paper-form classifier operates on the nominal expression.

Candidate generation is parse-driven and independent of eflomal: every
NOUN/PROPN/PRON token in the anchor target blocks yields one candidate
(head + selected dependents, contiguous hull, adposition-free edges).
Because candidates come from the parse, the correct constituent is
still in play when eflomal fails to link the right noun.

Routing ends in a finite state space per language side:
``nominal_counterpart``, ``non_nominal_counterpart`` (pronoun /
proper-name realization), ``ambiguous_multiple``, ``unresolved``,
``omission_candidate`` (locus reliable, nothing plausible — explicitly
NOT a final ``omitted``; that stays a later LLM/human adjudication),
``not_aligned`` (unreliable/absent retrieval locus). "No word link" is
never converted into omission, and unreliable retrieval is never
converted into omission.

Operational pilot rules (frozen constants, not tuned against GLM):
CONTEXTUAL_MARGIN = 0.05 cosine; CONTEXTUAL_FLOOR = 0.50 cosine;
MAX_EXPRESSION_TOKENS = 8.
"""

from __future__ import annotations

from dataclasses import dataclass

from hp_corpus.deterministic_tuples import (
    SideLayout,
    Token,
    _exact_substring,
    intersection_links,
)
from hp_corpus.machine_preannotation import MACHINE_TUPLE_COLUMNS, project_source_cells

CONSTITUENT_METHOD_ID = "eflomal-stanza-contextual-v1"

# Frozen operational pilot rules (reported, not tuned on GLM output).
CONTEXTUAL_MARGIN = 0.05
CONTEXTUAL_FLOOR = 0.50
MAX_EXPRESSION_TOKENS = 8

ROUTE_STATES = (
    "nominal_counterpart",
    "non_nominal_counterpart",
    "ambiguous_multiple",
    "unresolved",
    "omission_candidate",
    "not_aligned",
)
EVIDENCE_KINDS = ("both", "eflomal_only", "contextual_only", "none")

# Dependency relations that belong to the nominal expression, per
# language. The external adposition (``case``) is deliberately absent.
_EN_NOMINAL_DEPRELS = frozenset({"det", "amod", "compound", "nmod:poss"})
_ZH_NOMINAL_DEPRELS = frozenset({"det", "nummod", "clf", "nmod", "compound", "amod"})
# ZH classifiers attach to the numeral (clf -> NUM), one level below
# the head noun, so numeral material pulls its classifier in too.
_EDGE_SHRINK_UPOS = frozenset({"ADP", "PUNCT", "SYM", "PART"})


@dataclass
class Candidate:
    """One parse-derived referential candidate in a target block."""

    block_slot: int  # layout block index
    head_global: int  # global layout token index of the head
    lo: int  # global layout token index, span start (inclusive)
    hi: int  # global layout token index, span end (inclusive)
    category: str  # noun | pronoun | proper_name
    span_text: str  # exact substring of the block text

    @property
    def token_indices(self) -> set[int]:
        return set(range(self.lo, self.hi + 1))


def _category_for(upos: str) -> str | None:
    if upos == "PRON":
        return "pronoun"
    if upos == "PROPN":
        return "proper_name"
    if upos == "NOUN":
        return "noun"
    return None


def nominal_expression_indices(
    local_tokens: list[Token], head_local: int, lang: str
) -> tuple[int, int]:
    """Local [lo, hi] of the nominal expression headed at ``head_local``.

    Head + direct nominal dependents (per-language deprel set), plus —
    for ZH — the classifier dependents of included numerals. The result
    is the contiguous hull, shrunk at the edges while the edge token is
    an adposition/punctuation/particle, and capped: a hull wider than
    MAX_EXPRESSION_TOKENS falls back to the head token alone (a head
    with a sprawling dependent structure is not one defensible
    expression).
    """
    nominal = _ZH_NOMINAL_DEPRELS if lang == "zh" else _EN_NOMINAL_DEPRELS
    # CoNLL-U head indices are 1-based within the sentence; local
    # positions here are 0-based.
    head_one_based = head_local + 1
    include = {head_local}
    for i, tok in enumerate(local_tokens):
        if tok.deprel in nominal and _head_of(tok) == head_one_based:
            include.add(i)
            if lang == "zh" and tok.deprel == "nummod":
                for j, tok2 in enumerate(local_tokens):
                    if tok2.deprel == "clf" and _head_of(tok2) == i + 1:
                        include.add(j)
    lo, hi = min(include), max(include)
    while lo < hi and local_tokens[lo].upos in _EDGE_SHRINK_UPOS:
        lo += 1
    while hi > lo and local_tokens[hi].upos in _EDGE_SHRINK_UPOS:
        hi -= 1
    if hi - lo + 1 > MAX_EXPRESSION_TOKENS:
        return head_local, head_local
    return lo, hi


def _head_of(tok: Token) -> int:
    try:
        return int(tok.head)
    except ValueError:
        return -1


def generate_candidates(tgt_layout: SideLayout, lang: str) -> list[Candidate]:
    """All NOUN/PROPN/PRON-headed candidates in the layout's blocks.

    Small by construction: one candidate per referential head, not a
    subtree enumeration. Every span is verified to be an exact
    substring of its block's raw text.
    """
    block_bounds: list[tuple[int, int]] = []
    start = 0
    for slot in range(len(tgt_layout.block_ids)):
        end = start
        while end < len(tgt_layout.token_block) and tgt_layout.token_block[end] == slot:
            end += 1
        block_bounds.append((start, end))
        start = end

    candidates: list[Candidate] = []
    for slot, (b_start, b_end) in enumerate(block_bounds):
        local = tgt_layout.tokens[b_start:b_end]
        for head_local, tok in enumerate(local):
            category = _category_for(tok.upos)
            if category is None:
                continue
            lo_l, hi_l = nominal_expression_indices(local, head_local, lang)
            lo, hi = b_start + lo_l, b_start + hi_l
            forms = [t.form for t in tgt_layout.tokens[lo : hi + 1]]
            span_text = _exact_substring(tgt_layout.block_texts[slot], forms)
            if span_text is None or not span_text.strip():
                continue
            candidates.append(
                Candidate(
                    block_slot=slot,
                    head_global=b_start + head_local,
                    lo=lo,
                    hi=hi,
                    category=category,
                    span_text=span_text,
                )
            )
    return candidates


def eflomal_supported_indices(
    candidates: list[Candidate],
    pp_slice: range,
    fwd: set[tuple[int, int]],
    rev: set[tuple[int, int]],
) -> set[int]:
    """Candidate indices with at least one consensus link from the DE
    PP token span into the candidate's span."""
    intersect = intersection_links(fwd, rev)
    pp_links = {t for (s, t) in intersect if s in pp_slice}
    return {
        i
        for i, cand in enumerate(candidates)
        if cand.token_indices & pp_links
    }


@dataclass
class RouteResult:
    state: str
    chosen: Candidate | None = None
    evidence: str = "none"  # both | eflomal_only | contextual_only | none
    contextual_margin: float | None = None
    reason: str = ""


def route_side(
    candidates: list[Candidate],
    supported: set[int],
    contextual_sim: dict[int, float],
    *,
    locus_reliable: bool,
) -> RouteResult:
    """Agreement-based routing of one language side.

    ``contextual_sim`` maps candidate index → cosine similarity with
    the German PP (an independent signal; injected so the policy is a
    pure, testable function). High confidence requires eflomal and the
    contextual ranking to agree; single-signal choices stay marked as
    such (lower confidence); genuine disagreement or a too-close top-2
    is ``ambiguous_multiple``; positive absence of any plausible
    realization on a reliable locus is ``omission_candidate`` — never a
    final ``omitted``.
    """
    if not locus_reliable:
        return RouteResult("not_aligned", reason="unreliable_locus")

    if not candidates:
        return RouteResult("omission_candidate", reason="no_referential_candidate")

    ranked = sorted(contextual_sim.items(), key=lambda kv: -kv[1])
    top1_idx, top1_sim = ranked[0] if ranked else (None, None)
    top2_sim = ranked[1][1] if len(ranked) > 1 else None
    margin = (
        round(top1_sim - top2_sim, 4)
        if top1_sim is not None and top2_sim is not None
        else None
    )
    clear = margin is None or margin >= CONTEXTUAL_MARGIN
    plausible = top1_sim is not None and top1_sim >= CONTEXTUAL_FLOOR

    def choose(idx: int, evidence: str) -> RouteResult:
        cand = candidates[idx]
        state = (
            "non_nominal_counterpart"
            if cand.category in ("pronoun", "proper_name")
            else "nominal_counterpart"
        )
        return RouteResult(state, chosen=cand, evidence=evidence, contextual_margin=margin)

    if supported:
        if top1_idx is not None and top1_idx in supported:
            # both signals point at the same candidate — high
            # confidence regardless of margin (eflomal breaks the tie)
            return choose(top1_idx, "both")
        if len(supported) == 1:
            # eflomal only; contextual prefers another candidate —
            # keep, but marked single-signal / lower confidence
            return choose(next(iter(supported)), "eflomal_only")
        # multiple eflomal-supported candidates and contextual prefers
        # a different one: do not force a choice
        return RouteResult("ambiguous_multiple", contextual_margin=margin)

    # no eflomal support at all
    if clear and plausible:
        return choose(top1_idx, "contextual_only")
    if any(sim >= CONTEXTUAL_FLOOR for sim in contextual_sim.values()):
        # plausible material exists but no signal separates it
        return RouteResult("ambiguous_multiple", contextual_margin=margin)
    # locus reliable, nothing plausible anywhere — an omission
    # CANDIDATE, explicitly not a final omitted label
    return RouteResult("omission_candidate", contextual_margin=margin)


def form_for_candidate(cand: Candidate, layout: SideLayout, lang: str) -> tuple[str, str]:
    """Deterministic paper form of the chosen candidate's nominal
    expression (the frozen codebook rules, applied to the adposition-
    free span)."""
    tokens = layout.tokens[cand.lo : cand.hi + 1]
    if lang == "en":
        from hp_corpus.deterministic_tuples import english_form

        return english_form(tokens)
    from hp_corpus.deterministic_tuples import chinese_form

    return chinese_form(tokens)


# --- artifact row -----------------------------------------------------------------

CONSTITUENT_EXTRA_COLUMNS: tuple[str, ...] = (
    "machine_en_evidence",
    "machine_en_contextual_margin",
    "machine_en_candidate_count",
    "machine_zh_evidence",
    "machine_zh_contextual_margin",
    "machine_zh_candidate_count",
)
CONSTITUENT_COLUMNS: tuple[str, ...] = (
    MACHINE_TUPLE_COLUMNS + CONSTITUENT_EXTRA_COLUMNS
)


def build_constituent_row(
    source_row: dict[str, str],
    routed: dict[str, RouteResult],
    forms: dict[str, tuple[str, str]],
    candidate_counts: dict[str, int],
    *,
    pack_stratum: str,
    method_params_id: str,
) -> dict[str, str]:
    """One artifact row on the shared machine-tuple schema plus the
    constituent-evidence columns."""
    row = project_source_cells(source_row)
    row["pack_stratum"] = pack_stratum
    row["machine_annotation_model"] = CONSTITUENT_METHOD_ID
    row["machine_annotation_prompt_version"] = method_params_id
    proposal, reason = _de_proposal(source_row)
    row["machine_de_valid_proposal"] = proposal
    row["machine_de_exclusion_reason"] = reason

    for side in ("en", "zh"):
        result = routed.get(side)
        if result is None:
            row[f"machine_{side}_counterpart"] = ""
            row[f"machine_{side}_paper_form"] = ""
            row[f"machine_{side}_detail_form"] = ""
            row[f"machine_{side}_status"] = "not_aligned"
            row[f"machine_{side}_evidence"] = "none"
            row[f"machine_{side}_contextual_margin"] = ""
            row[f"machine_{side}_candidate_count"] = "0"
            continue
        chosen = result.chosen
        paper, detail = forms.get(side, ("", "")) if chosen else ("", "")
        if chosen and chosen.category == "pronoun" and not detail:
            detail = "pronoun"
        elif chosen and chosen.category == "proper_name" and not detail:
            detail = "proper_name"
        row[f"machine_{side}_counterpart"] = chosen.span_text if chosen else ""
        row[f"machine_{side}_paper_form"] = paper
        row[f"machine_{side}_detail_form"] = detail
        row[f"machine_{side}_status"] = result.state
        row[f"machine_{side}_evidence"] = result.evidence
        row[f"machine_{side}_contextual_margin"] = (
            f"{result.contextual_margin:.4f}"
            if result.contextual_margin is not None
            else ""
        )
        row[f"machine_{side}_candidate_count"] = str(candidate_counts.get(side, 0))
    return {c: row.get(c, "") for c in CONSTITUENT_COLUMNS}


def _de_proposal(source_row: dict[str, str]) -> tuple[str, str]:
    from hp_corpus.deterministic_tuples import de_valid_proposal

    return de_valid_proposal(
        source_row.get("de_form", ""),
        source_row.get("de_det_xpos", ""),
        source_row.get("de_det_deprel", ""),
    )


__all__ = [
    "CONSTITUENT_METHOD_ID",
    "CONTEXTUAL_MARGIN",
    "CONTEXTUAL_FLOOR",
    "MAX_EXPRESSION_TOKENS",
    "ROUTE_STATES",
    "EVIDENCE_KINDS",
    "Candidate",
    "generate_candidates",
    "nominal_expression_indices",
    "eflomal_supported_indices",
    "RouteResult",
    "route_side",
    "form_for_candidate",
    "CONSTITUENT_EXTRA_COLUMNS",
    "CONSTITUENT_COLUMNS",
    "build_constituent_row",
]
