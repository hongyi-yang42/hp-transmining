"""Hybrid arbitration over saved machine tuple evidence (pilot).

Experimental side path — NOT the production annotation workflow. This
module consumes *already-saved* pilot artifacts and produces one hybrid
machine proposal per language side; it never calls a model, never
retrains an aligner, and never rewrites the artifacts it reads:

  * the saved GLM proposal CSV (semantic counterpart proposer);
  * the saved constituent-v2 CSV + per-row candidate dump (Stanza
    structure + eflomal lexical flags + LaBSE contextual ranks);
  * the production master row (retrieval context and provenance).

Division of responsibility (explicit):

  GLM      proposes the counterpart span — its strength is deciding
           WHICH overt expression, paraphrase, pronoun, or verbal
           realization corresponds to the German PP in context.
  Stanza   is the surface-form authority — once a span exists, the
           paper form / detail form / structural realization type are
           recomputed deterministically from the span's parse tokens;
           the GLM form labels are never trusted as authoritative.
  eflomal/LaBSE   act as independent checkers — agreement with,
           contradiction of, or absence of support for the GLM span,
           never the final answer themselves.

Two GLM failure modes get explicit guards:

  * omission guard — a GLM ``omitted`` claim is never final. It
    becomes ``omission_candidate`` at best, and is routed to
    ``local_only`` (local overt candidate found) or
    ``non_nominal_or_restructured`` (positive verbal-link evidence)
    when independent evidence contradicts it.
  * drift/scope guards — the GLM span is checked against the strict
    DP-aligned locus and the local candidate structure. A span that
    lives outside the aligned locus but inside the bounded retrieval
    context is an *alignment-scope* conflict (strict-alignment scope
    vs retrieval scope — nothing shows the referent itself is wrong);
    a span that diverges from comparable evidence *within* a usable
    locus is a genuine ``semantic_disagreement``.

Routing state space (per side, transparent policy — proposal tiers,
never claimed accuracy):

  machine_agreement          cross-method-supported machine proposal:
                             GLM span and the local deterministic
                             choice identify the same referent. NOT
                             human-validated and NOT an implied
                             correctness claim — converging machine
                             signals can share the same semantic-role
                             error (development observation), and no
                             rule here may bypass later validation.
  llm_only                   GLM span is structurally valid but local
                             evidence is unavailable or too weak (no
                             strict anchor, ambiguous/unresolved local
                             routing, or no local support at all —
                             e.g. a same-locus unsupported proposal,
                             which the current machine stack cannot
                             resolve automatically).
  local_only                 local evidence found an overt candidate
                             while GLM claims omission or proposes
                             nothing; neither side auto-wins.
  alignment_scope_conflict   the GLM proposal is a valid substring of
                             the existing bounded retrieval context
                             but the strict DP anchor does not contain
                             it — the conflict is between alignment
                             scope and retrieval scope, with no
                             independent evidence that the proposed
                             referent is semantically wrong.
  semantic_disagreement      reserved for comparable evidence within
                             a usable translation locus supporting
                             genuinely different referents or
                             realizations (divergent local choice, or
                             consensus verbal links vs a nominal GLM
                             span).
  non_nominal_or_restructured  positive evidence (verb-led GLM span
                             mapped to parse tokens, or consensus
                             verbal links) that the PP was realized
                             verbally/clausally/idiomatically — never
                             converted to omission, never forced into
                             a nominal paper form.
  omission_candidate         reliable locus, neither route identifies
                             an overt realization — never final
                             ``omitted``.
  retrieval_not_aligned      the target locus itself is unreliable.
  unresolved                 everything else.

Evidence strings form a small controlled vocabulary recorded per side
(``hybrid_{side}_evidence``); the routing policy itself lives in
:func:`arbitrate_side` as one pure, testable function.

Ontology boundary: the paper-comparable core and the production/human
annotation codebooks are frozen and untouched by this module. The
hybrid detail values ``non_nominal`` and ``quantifier`` are machine
diagnostics of this artifact only — they never enter the production
detail vocabulary, and both map to the paper core ``other``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from hp_corpus.deterministic_tuples import (
    SideLayout,
    Token,
    chinese_form,
    english_form,
)
from hp_corpus.machine_preannotation import MACHINE_TUPLE_COLUMNS

HYBRID_METHOD_ID = "glm+constituent-arbitration-v1"

HYBRID_ROUTES = (
    "machine_agreement",
    "llm_only",
    "local_only",
    "alignment_scope_conflict",
    "semantic_disagreement",
    "non_nominal_or_restructured",
    "omission_candidate",
    "retrieval_not_aligned",
    "unresolved",
)
REALIZATION_TYPES = ("nominal", "pronoun", "proper_name", "verbal", "")

# Tokens skipped when locating the span's leftmost content token for
# structural realization typing (the counterpart of a PP legitimately
# starts with adpositions, infinitive markers, determiners …).
_REALIZATION_SKIP_UPOS = frozenset({"ADP", "PART", "PUNCT", "SYM", "DET", "CCONJ", "SCONJ"})
_VERBAL_UPOS = frozenset({"VERB", "AUX"})


# --- inputs -------------------------------------------------------------------------


@dataclass(frozen=True)
class GlmSide:
    """One side of a saved GLM proposal row."""

    status: str  # aligned | not_aligned | uncertain | invalid
    span: str
    paper_form: str
    detail_form: str

    @property
    def omission_claim(self) -> bool:
        """GLM claimed a genuine translator omission (aligned + blank
        span + omitted detail). Anything else with a blank span (e.g.
        ``not_aligned``) is a GLM-side retrieval failure, not an
        omission claim."""
        return (
            self.status == "aligned" and not self.span.strip() and (self.detail_form == "omitted")
        )

    @property
    def proposes_span(self) -> bool:
        return bool(self.span.strip())


@dataclass(frozen=True)
class LocalSide:
    """One side of the saved constituent-v2 routing result."""

    state: str  # ROUTE_STATES value from constituent_candidates
    evidence: str  # both | disagreement | contextual_supported | …
    chosen_span: str
    form: tuple[str, str] = ("", "")  # (paper, detail) of the chosen span
    category: str = ""  # noun | pronoun | proper_name ("" if unchosen)

    @property
    def has_overt_choice(self) -> bool:
        return bool(self.chosen_span.strip())


@dataclass
class SpanMapping:
    """A GLM span located inside one block of a target layout."""

    block_slot: int
    start_char: int
    end_char: int
    tokens: list[Token] = field(default_factory=list)


@dataclass(frozen=True)
class SideFacts:
    """Pre-computed independent evidence about one side (runner-supplied,
    from saved artifacts only — no model calls)."""

    locus: str  # reliable | no_anchor_context_available | retrieval_not_aligned
    span_in_context: bool  # exact substring of the master retrieval context
    mapping: SpanMapping | None  # span → parse tokens inside the locus blocks
    in_locus_blocks: bool  # span is an exact substring of a locus block text
    eflomal_covers_span: bool  # any eflomal-supported candidate covers the span
    contextual_top1_covers: bool  # the contextual rank-1 candidate covers it


# --- span → parse tokens --------------------------------------------------------------


def map_span_to_tokens(layout: SideLayout, span: str) -> SpanMapping | None:
    """Locate ``span`` as an exact substring of one layout block and
    collect the parse tokens whose character intervals intersect it.

    Blocks are tried in order until one yields a non-empty token cover:
    the parse files contain blocks whose ``# text`` field repeats text
    tokenized in a neighbouring block, so a textual hit whose tokens
    live elsewhere must not abort the search. Fail-closed overall: if
    no block covers the span with tokens, returns ``None``. Tokens
    partially cut by the span (a model copying a sub-word fragment)
    still count — classification runs on the full parse tokens, never
    on fragments.
    """
    span = (span or "").strip()
    if not span:
        return None
    for slot, block_text in enumerate(layout.block_texts):
        idx = block_text.find(span)
        if idx < 0:
            continue
        block_tokens = [
            (i, t) for i, t in enumerate(layout.tokens) if layout.token_block[i] == slot
        ]
        cursor = 0
        covered: list[Token] = []
        for _, tok in block_tokens:
            m = re.search(rf"\s*{re.escape(tok.form)}", block_text[cursor:])
            if m is None:
                break
            tok_start = cursor + m.end() - len(tok.form)
            tok_end = cursor + m.end()
            cursor = tok_end
            if tok_start < idx + len(span) and tok_end > idx:
                covered.append(tok)
        if not covered:
            continue
        return SpanMapping(
            block_slot=slot, start_char=idx, end_char=idx + len(span), tokens=covered
        )
    return None


# --- deterministic re-computation of form ----------------------------------------------


def realization_type(tokens: list[Token]) -> str:
    """Structural realization type of a span from its parse tokens.

    The leftmost content token decides: verb/auxiliary-led → verbal;
    a pronoun-only span → pronoun; proper-name-led → proper_name;
    anything else → nominal. Fragmentary/unmarked spans → ``""``.
    """
    core = [t for t in tokens if t.upos not in _REALIZATION_SKIP_UPOS]
    if not core:
        return ""
    first = core[0]
    if first.upos in _VERBAL_UPOS:
        return "verbal"
    if first.upos == "PRON":
        if any(t.upos in ("NOUN", "PROPN") for t in core):
            return "nominal"
        if any(t.upos in _VERBAL_UPOS for t in core):
            return "verbal"  # pronoun-led clause, still non-nominal
        return "pronoun"
    if first.upos == "PROPN":
        return "proper_name"
    return "nominal"


def _zh_has_numeral_classifier(tokens: list[Token]) -> bool:
    """Overt adjacent numeral+classifier pair anywhere in the span."""
    for a, b in zip(tokens, tokens[1:], strict=False):
        if a.upos in ("NUM", "CD") and (b.upos in ("CL", "M") or b.deprel == "clf"):
            return True
    return False


# Quantificational EN determiners outside the frozen classifier's
# central-determiner set: a span led by one of these is a quantified DP,
# not a bare singular and not an ordinary definite.
_EN_QUANTIFIER_DETERMINERS = frozenset(
    {
        "every",
        "each",
        "all",
        "some",
        "any",
        "no",
        "most",
        "many",
        "few",
        "several",
        "both",
        "either",
        "neither",
    }
)


def recompute_form(tokens: list[Token], lang: str) -> tuple[str, str]:
    """Deterministic (paper, detail) of a proposed span, per the frozen
    codebook rules — with the pilot's structural guards:

    * a verb-led span is a non-nominal realization: ``other`` +
      ``non_nominal`` detail, never a nominal paper form;
    * a ZH span containing overt numeral marking can never survive as
      ``bare`` (adjacent NUM+classifier pair, or any overt NUM token —
      the ZH parser does not reliably tag measure words as
      classifiers, but an overt numeral alone already defeats a bare
      reading);
    * an EN span led by a quantificational determiner is ``other`` +
      ``quantifier``, never a bare singular.

    ``quantifier`` joins ``non_nominal`` as a hybrid-layer detail
    extension; the frozen codebooks in the GLM and deterministic
    modules stay untouched.
    """
    if not tokens:
        return "", ""
    if realization_type(tokens) == "verbal":
        return "other", "non_nominal"
    paper, detail = english_form(tokens) if lang == "en" else chinese_form(tokens)
    if lang == "zh" and paper == "bare":
        if _zh_has_numeral_classifier(tokens) or any(t.upos in ("NUM", "CD") for t in tokens):
            return "other", "numeral_classifier"
    if lang == "en" and paper == "bare_singular":
        stripped = [t for t in tokens if t.upos not in ("ADP", "PUNCT", "SYM")]
        if stripped and stripped[0].form.lower() in _EN_QUANTIFIER_DETERMINERS:
            return "other", "quantifier"
    return paper, detail


# --- routing ---------------------------------------------------------------------------


@dataclass
class HybridDecision:
    route: str
    counterpart: str = ""
    paper_form: str = ""
    detail_form: str = ""
    realization_type: str = ""
    evidence: str = ""
    requires_review: bool = True
    form_correction: tuple[str, str] | None = None  # (glm paper, glm detail)


def _relation(a: str, b: str) -> str:
    a, b = (a or "").strip(), (b or "").strip()
    if not a or not b:
        return "unavailable"
    if a == b:
        return "exact"
    if a in b or b in a:
        return "containment"
    return "divergent"


def _recomputed(facts: SideFacts, lang: str) -> tuple[str, str, str]:
    if facts.mapping is None:
        return "", "", ""
    paper, detail = recompute_form(facts.mapping.tokens, lang)
    return paper, detail, realization_type(facts.mapping.tokens)


def arbitrate_side(glm: GlmSide, local: LocalSide, facts: SideFacts, lang: str) -> HybridDecision:
    """One pure routing function per language side. See module docstring
    for the policy; every branch records why it fired."""
    if facts.locus == "retrieval_not_aligned":
        return HybridDecision("retrieval_not_aligned", evidence="unreliable_locus")

    if glm.proposes_span and not facts.span_in_context:
        # Fail-closed: the saved GLM artifact passed substring
        # validation at generation time, so this should never fire —
        # but a span that is not an exact substring of the supplied
        # context is unusable, never guessed.
        return HybridDecision("unresolved", evidence="glm_span_not_in_context")

    # --- omission guard: a GLM omitted claim is never final ------------------
    if glm.omission_claim:
        if local.state == "non_nominal_realization":
            return HybridDecision(
                "non_nominal_or_restructured",
                evidence="omission_blocked_local_verbal_links",
            )
        if local.has_overt_choice:
            return HybridDecision(
                "local_only",
                counterpart=local.chosen_span,
                paper_form=local.form[0],
                detail_form=local.form[1],
                realization_type=local.category or "",
                evidence="omission_blocked_local_overt",
            )
        # Case B: reliable locus, no overt local realization — at most a
        # candidate for omission, always requiring review.
        ev = "omission_candidate_no_local_overt"
        if facts.locus == "no_anchor_context_available":
            ev = "omission_candidate_no_anchor"
        elif local.state == "no_overt_candidate":
            ev = "omission_candidate_local_absence"
        elif local.state in ("ambiguous_multiple", "unresolved"):
            ev = "omission_candidate_local_ambiguous"
        return HybridDecision("omission_candidate", evidence=ev)

    # --- GLM proposes a span ---------------------------------------------------
    if glm.proposes_span:
        if not facts.in_locus_blocks and facts.locus == "reliable":
            # The proposal is a valid substring of the bounded retrieval
            # context (checked above), but the strict DP anchor does not
            # contain it: the conflict is between alignment scope and
            # retrieval scope. Nothing here shows the proposed referent
            # is semantically wrong — this is NOT a semantic
            # disagreement claim.
            return HybridDecision(
                "alignment_scope_conflict", evidence="glm_span_outside_aligned_locus"
            )
        if facts.mapping is None:
            # inside the locus blocks but not mappable to parse tokens
            # (e.g. the span crosses a block boundary) — the span
            # stands, no recomputed form; support signals still count
            if facts.eflomal_covers_span:
                ev = "llm_eflomal_supported"
            elif facts.contextual_top1_covers:
                ev = "llm_contextual_top1"
            elif facts.in_locus_blocks:
                ev = "llm_span_unmappable"
            else:
                ev = "llm_no_local_support"
            return HybridDecision("llm_only", counterpart=glm.span, evidence=ev)
        paper, detail, realization = _recomputed(facts, lang)
        correction = (
            (glm.paper_form, glm.detail_form)
            if (paper, detail) != (glm.paper_form, glm.detail_form)
            else None
        )
        if realization == "verbal":
            return HybridDecision(
                "non_nominal_or_restructured",
                counterpart=glm.span,
                paper_form=paper,
                detail_form=detail,
                realization_type=realization,
                evidence=(
                    "glm_verbal_span_local_overlap"
                    if _relation(glm.span, local.chosen_span) in ("exact", "containment")
                    else "glm_verbal_span"
                ),
                form_correction=correction,
            )
        rel = _relation(glm.span, local.chosen_span)
        if local.has_overt_choice:
            if rel in ("exact", "containment"):
                return HybridDecision(
                    "machine_agreement",
                    counterpart=glm.span,
                    paper_form=paper,
                    detail_form=detail,
                    realization_type=realization,
                    evidence=f"agreement_{rel}",
                    requires_review=local.evidence == "disagreement",
                    form_correction=correction,
                )
            strong = local.evidence in ("both", "contextual_supported")
            return HybridDecision(
                "semantic_disagreement",
                evidence=("divergent_local_strong" if strong else "divergent_local_weak"),
            )
        if local.state == "non_nominal_realization":
            # local links say the PP was realized verbally; the GLM span
            # is nominal — a genuine structural conflict, review it.
            return HybridDecision(
                "semantic_disagreement",
                evidence="local_verbal_links_vs_glm_nominal",
            )
        # no local decision (ambiguous / unresolved / no overt
        # candidate / no anchor): the GLM span stands, lower tier
        if facts.eflomal_covers_span:
            ev = "llm_eflomal_supported"
        elif facts.contextual_top1_covers:
            ev = "llm_contextual_top1"
        else:
            ev = "llm_no_local_support"
        return HybridDecision(
            "llm_only",
            counterpart=glm.span,
            paper_form=paper,
            detail_form=detail,
            realization_type=realization,
            evidence=ev,
            # a span that could not be mapped to parse tokens (e.g. it
            # crosses block boundaries in the concatenated context) has
            # no recomputed form — nothing to correct yet
            form_correction=correction if paper else None,
        )

    # --- GLM proposes nothing (not_aligned / uncertain / invalid) ---------------
    if local.state == "non_nominal_realization":
        return HybridDecision("non_nominal_or_restructured", evidence="local_verbal_links")
    if local.has_overt_choice:
        return HybridDecision(
            "local_only",
            counterpart=local.chosen_span,
            paper_form=local.form[0],
            detail_form=local.form[1],
            realization_type=local.category or "",
            evidence=(
                "local_only_no_anchor"
                if facts.locus == "no_anchor_context_available"
                else "local_only_no_glm_proposal"
            ),
        )
    return HybridDecision("unresolved", evidence="no_proposal_no_local_choice")


# --- artifact row ----------------------------------------------------------------------

_LOCAL_ECHO_COLUMNS = tuple(
    f"local_{side}_{col}" for side in ("en", "zh") for col in ("state", "counterpart", "evidence")
)
HYBRID_PROPOSAL_COLUMNS = tuple(
    f"hybrid_{side}_{col}"
    for side in ("en", "zh")
    for col in (
        "counterpart",
        "realization_type",
        "paper_form",
        "detail_form",
        "route",
        "evidence",
        "requires_review",
    )
)
HYBRID_COLUMNS: tuple[str, ...] = (
    MACHINE_TUPLE_COLUMNS
    + _LOCAL_ECHO_COLUMNS
    + HYBRID_PROPOSAL_COLUMNS
    + ("hybrid_annotation_method",)
)


def build_hybrid_row(
    glm_row: dict[str, str],
    decisions: dict[str, HybridDecision],
    local: dict[str, LocalSide],
) -> dict[str, str]:
    """Assemble one hybrid artifact row: the saved GLM row passes through
    verbatim (its ``machine_*`` cells are the GLM proposal, unchanged),
    the local routing is echoed under ``local_*`` names, and the hybrid
    proposal lives in its own ``hybrid_*`` columns."""
    row = dict(glm_row)
    for side in ("en", "zh"):
        loc = local.get(side)
        row[f"local_{side}_state"] = loc.state if loc else ""
        row[f"local_{side}_counterpart"] = loc.chosen_span if loc else ""
        row[f"local_{side}_evidence"] = loc.evidence if loc else ""
        dec = decisions.get(side)
        if dec is None:
            continue
        row[f"hybrid_{side}_counterpart"] = dec.counterpart
        row[f"hybrid_{side}_realization_type"] = dec.realization_type
        row[f"hybrid_{side}_paper_form"] = dec.paper_form
        row[f"hybrid_{side}_detail_form"] = dec.detail_form
        row[f"hybrid_{side}_route"] = dec.route
        row[f"hybrid_{side}_evidence"] = dec.evidence
        row[f"hybrid_{side}_requires_review"] = "yes" if dec.requires_review else "no"
    row["hybrid_annotation_method"] = HYBRID_METHOD_ID
    return {c: row.get(c, "") for c in HYBRID_COLUMNS}


__all__ = [
    "HYBRID_METHOD_ID",
    "HYBRID_ROUTES",
    "REALIZATION_TYPES",
    "HYBRID_COLUMNS",
    "GlmSide",
    "LocalSide",
    "SpanMapping",
    "SideFacts",
    "HybridDecision",
    "map_span_to_tokens",
    "realization_type",
    "recompute_form",
    "arbitrate_side",
    "build_hybrid_row",
]
