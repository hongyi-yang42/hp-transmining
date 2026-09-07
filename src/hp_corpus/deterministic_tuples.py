"""Deterministic-first local baseline for trilingual form tuples (pilot).

Experimental side path beside the GLM pre-annotation pilot — NOT the
production annotation workflow. Where the GLM pilot asks a model to
propose counterparts and forms, this module derives them locally and
deterministically from artifacts the production pipeline already
produced:

  * the DE/EN/ZH CoNLL-U parses (stable sentence/block ids, tokens,
    UPOS, features) — the DE side via the ``_nomwt`` variant whose token
    indices the extraction's ``de_token_start``/``de_token_end`` use;
  * the DE-EN / DE-ZH sentence-group alignment records from the
    production LaBSE + DP pipeline (the training bitext and each row's
    anchor group);
  * one local word aligner: **eflomal** (IBM1+HMM+fertility, both
    directions), with links symmetrized by intersection (precision
    first — the *smallest defensible* span, never a guess).

Method identity: ``eflomal-intersect-v1``. eflomal's Bayesian sampling
has no user seed, so the *trained links* are the frozen artifact: they
are written once with their sha256 recorded, and every downstream step
is a pure function of (links, parses, master row). Regenerating links
re-trains and may differ slightly — treat the saved links like the
embedding cache.

Counterpart recovery is conservative by construction: a non-empty span
is always an exact substring of the target block's own text AND of the
master's retrieval context; linked tokens scattered over more than one
target block, oversized hulls, or failed substring verification yield
``unresolved`` rather than a guess. ``not_aligned`` is reserved for rows
whose side has no aligned anchor at all. The deterministic path never
asserts a translator omission — that judgment needs a reader.

Paper-form codebooks are the same frozen vocabularies as the GLM pilot
(``hp_corpus.machine_preannotation``); EN/ZH form assignment here is a
deterministic function of the recovered span's tokens (determiner,
pronoun, proper name, number), after absorbing an immediately preceding
central determiner into the span (German PPs headed by a definite
article translate to EN PPs whose determiner belongs to the
counterpart; bare word-alignment links routinely miss function words).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from hp_corpus.machine_preannotation import (
    MACHINE_TUPLE_COLUMNS,
    project_source_cells,
)

DETERMINISTIC_METHOD_ID = "eflomal-intersect-v1"
LINKS_SCHEMA_VERSION = "eflomal-links-v1"

# eflomal defaults used for training (recorded in every artifact row).
EFLOMAL_PARAMETERS = {"model": 3, "n_samplers": 1, "symmetrization": "intersection"}

# Hull cap: a defensible counterpart of one German PP is not more than
# this many target tokens wide. Wider hulls mean scattered links.
MAX_SPAN_TOKENS = 12

SPAN_STATUSES = ("aligned", "unresolved", "not_aligned")


# --- CoNLL-U reading -----------------------------------------------------------


@dataclass(frozen=True)
class Token:
    """The projection of a CoNLL-U token this module needs."""

    form: str
    upos: str
    feats: str


@dataclass
class ConlluSentence:
    sent_id: str
    text: str
    tokens: list[Token]


def read_conllu(path) -> dict[str, ConlluSentence]:
    """Read a CoNLL-U file into ``{sent_id: sentence}``.

    Multi-word range lines (``5-6 im``) and empty-node lines
    (``5.1``) are skipped — token indices refer to the surviving token
    sequence, matching the extractor's ``_nomwt`` convention on the DE
    side and plain token streams on the EN/ZH side.
    """
    out: dict[str, ConlluSentence] = {}
    sent_id: str | None = None
    text: str | None = None
    tokens: list[Token] = []

    def flush() -> None:
        if sent_id is not None:
            out[sent_id] = ConlluSentence(
                sent_id=sent_id, text=text or "", tokens=tokens
            )

    for raw in open(path, encoding="utf-8"):
        line = raw.rstrip("\n")
        if line.startswith("# sent_id = "):
            flush()
            sent_id = line.split(" = ", 1)[1].strip()
            text = None
            tokens = []
        elif line.startswith("# text = "):
            text = line.split(" = ", 1)[1]
        elif line and line[0].isdigit():
            cols = line.split("\t")
            tid = cols[0]
            if "-" in tid or "." in tid:
                continue
            tokens.append(
                Token(form=cols[1], upos=cols[3], feats=cols[5] if len(cols) > 5 else "")
            )
    flush()
    return out


def _block_ordinal(sent_id: str) -> int:
    """``…#b012`` → 12; ids without a block suffix sort first."""
    m = re.search(r"#b(\d+)$", sent_id)
    return int(m.group(1)) if m else -1


def blocks_for_segment(
    sentences: dict[str, ConlluSentence], segment_id: str
) -> list[str]:
    """Ordered parse-block sent_ids of one segment (by ``#b`` ordinal)."""
    prefix = segment_id + "#b"
    return sorted(
        (sid for sid in sentences if sid == segment_id or sid.startswith(prefix)),
        key=_block_ordinal,
    )


# --- alignment-record side layout ------------------------------------------------


@dataclass
class SideLayout:
    """One alignment-record side as a token concatenation: per-token
    block membership, per-block raw text, tokens."""

    block_ids: list[str] = field(default_factory=list)
    block_texts: list[str] = field(default_factory=list)
    token_block: list[int] = field(default_factory=list)  # token idx -> block slot
    tokens: list[Token] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tokens)


def build_side_layout(
    segment_ids: list[str], sentences: dict[str, ConlluSentence]
) -> SideLayout:
    layout = SideLayout()
    for seg in segment_ids:
        for bid in blocks_for_segment(sentences, seg):
            slot = len(layout.block_ids)
            layout.block_ids.append(bid)
            layout.block_texts.append(sentences[bid].text)
            for tok in sentences[bid].tokens:
                layout.token_block.append(slot)
                layout.tokens.append(tok)
    return layout


def load_alignment_records(path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_bitext(
    records: list[dict],
) -> list[dict]:
    """One bitext line per alignment record with both sides non-empty.

    Each line carries the record's segment ids (not tokens) so the
    frozen links artifact stays corpus-text-free; tokens are re-derived
    from the parses at annotate time.
    """
    bitext = []
    for rec in records:
        src, tgt = rec.get("src") or [], rec.get("tgt") or []
        if not src or not tgt:
            continue
        bitext.append(
            {
                "de_segments": list(src),
                "tgt_segments": list(tgt),
                "type": rec.get("type", ""),
            }
        )
    return bitext


# --- eflomal links I/O -------------------------------------------------------------


def eflomal_input_line(side: SideLayout) -> str:
    """The space-joined token line eflomal expects for one record side.

    eflomal tokenizes on whitespace, but CoNLL-U token forms may
    legitimately contain internal spaces (e.g. the ZH text layer's
    ``… 」`` ellipsis artifacts). Any internal whitespace run is
    replaced by ``□`` (U+25A1, absent from these corpora) so the line
    has exactly one whitespace field per layout token — link indices
    then address the layout tokens directly. The placeholder lives
    only in the aligner's view; span recovery always uses the original
    forms and block text.
    """
    fields = [re.sub(r"\s+", "□", tok.form) for tok in side.tokens]
    return " ".join(fields)


def parse_links_file(path) -> list[set[tuple[int, int]]]:
    """Parse an eflomal links file into per-sentence link sets.

    eflomal writes exactly one line per input sentence — a sentence
    whose every token aligned to NULL is an empty line. Pairs are
    ``src-trg`` with 0-based token indices.
    """
    sentences: list[set[tuple[int, int]]] = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            links: set[tuple[int, int]] = set()
            for pair in raw.split():
                s, t = pair.split("-")
                links.add((int(s), int(t)))
            sentences.append(links)
    return sentences


def intersection_links(
    fwd: set[tuple[int, int]], rev: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    """Symmetrize by intersection (precision first).

    Both eflomal link files are source-first indexed — the reverse
    model's output is normalized to ``src-trg`` pairs like the forward
    one (this matches eflomal's own ``calculate_priors`` consumer, which
    indexes ``src_sent[i]``/``trg_sent[j]`` from both files). The
    intersection is therefore the plain set intersection.
    """
    return fwd & rev


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# --- span recovery ----------------------------------------------------------------


_EN_CENTRAL_DETERMINERS = frozenset(
    {"the", "a", "an", "this", "that", "these", "those"}
)
_EN_POSSESSIVE_DETERMINERS = frozenset(
    {"my", "your", "his", "her", "its", "our", "their", "whose"}
)
_ZH_DEMONSTRATIVE_HEADS = ("这", "那")


@dataclass
class SpanRecovery:
    status: str  # aligned | unresolved | not_aligned
    span_text: str = ""
    reason: str = ""  # machine-readable cause when not aligned
    tgt_block_id: str = ""
    absorbed_determiner: bool = False
    span_tokens: list[Token] = field(default_factory=list)  # post-absorption


def _absorb_leading_determiner(
    layout: SideLayout, start: int, end: int, lang: str
) -> tuple[int, bool]:
    """Extend the span one token left when that token is a central
    determiner. Word-alignment links routinely miss function words; the
    counterpart NP's determiner directly precedes the recovered noun
    span, so absorbing it is defensible. One step only — determiners
    do not stack."""
    prev = start - 1
    if prev < 0 or layout.token_block[prev] != layout.token_block[start]:
        return start, False
    form = layout.tokens[prev].form.lower()
    if lang == "en" and form in (_EN_CENTRAL_DETERMINERS | _EN_POSSESSIVE_DETERMINERS):
        return prev, True
    if lang == "zh" and any(form.startswith(h) for h in _ZH_DEMONSTRATIVE_HEADS):
        return prev, True
    return start, False


def _exact_substring(text: str, token_forms: list[str]) -> str | None:
    """Locate the token sequence in ``text`` as an exact substring.

    Tokens are adjacent in the original text up to whitespace, so the
    pattern allows arbitrary whitespace between tokens (EN punctuation
    splits, ZH zero-width joins) and returns the matched source text
    verbatim — never a reconstructed string.
    """
    if not token_forms:
        return None
    pattern = r"\s*".join(re.escape(f) for f in token_forms)
    m = re.search(pattern, text)
    return m.group(0) if m else None


def recover_target_span(
    pp_de_slice: range,
    de_layout: SideLayout,
    tgt_layout: SideLayout,
    fwd: set[tuple[int, int]],
    rev: set[tuple[int, int]],
    *,
    lang: str,
) -> SpanRecovery:
    """Recover the smallest defensible target span for one German PP.

    ``pp_de_slice`` is the PP's 0-based token positions within
    ``de_layout.tokens`` (the row's parse block as laid out in the
    anchor record). Links are that record's eflomal links over the two
    concatenations. The recovered span is verified to be an exact
    substring of the containing target block's raw text; everything
    else about defensibility (context window) is the caller's gate.
    """
    if not pp_de_slice or pp_de_slice.stop - 1 >= len(de_layout.tokens):
        return SpanRecovery("unresolved", reason="de_token_range_invalid")

    intersect = intersection_links(fwd, rev)
    linked = sorted(t for (s, t) in intersect if s in pp_de_slice)
    if not linked:
        return SpanRecovery("unresolved", reason="no_intersection_links")

    slots = {tgt_layout.token_block[t] for t in linked}
    if len(slots) > 1:
        return SpanRecovery("unresolved", reason="links_across_blocks")
    slot = slots.pop()

    start, end = min(linked), max(linked)
    if end - start + 1 > MAX_SPAN_TOKENS:
        return SpanRecovery("unresolved", reason="span_hull_too_wide")

    start, absorbed = _absorb_leading_determiner(tgt_layout, start, end, lang)
    forms = [tok.form for tok in tgt_layout.tokens[start : end + 1]]
    span_text = _exact_substring(tgt_layout.block_texts[slot], forms)
    if span_text is None or not span_text.strip():
        return SpanRecovery("unresolved", reason="substring_not_found")

    return SpanRecovery(
        status="aligned",
        span_text=span_text,
        tgt_block_id=tgt_layout.block_ids[slot],
        absorbed_determiner=absorbed,
        span_tokens=tgt_layout.tokens[start : end + 1],
    )


# --- deterministic form rules -------------------------------------------------------


def _strip_edges(tokens: list[Token]) -> list[Token]:
    """Drop leading adpositions and trailing punctuation — the span is
    the counterpart of a PP, so the classifier must see past the
    preposition and any sentence punctuation caught by the hull."""
    out = list(tokens)
    while out and out[0].upos == "ADP":
        out.pop(0)
    while out and out[-1].upos in ("PUNCT", "SYM"):
        out.pop()
    return out


def _has_number_sing(tokens: list[Token]) -> bool:
    nouns = [t for t in tokens if t.upos == "NOUN"]
    if not nouns:
        return False
    feats = nouns[-1].feats or ""
    if "Number=Plur" in feats:
        return False
    return True  # Number=Sing or unmarked → singular reading


def _proper_name_dominant(tokens: list[Token]) -> bool:
    if not tokens:
        return False
    propn = sum(1 for t in tokens if t.upos == "PROPN")
    return propn >= max(1, (len(tokens) + 1) // 2)


def english_form(tokens: list[Token]) -> tuple[str, str]:
    """Deterministic EN paper-form rule. Returns ``(paper_form, detail)``."""
    toks = _strip_edges(tokens)
    if not toks:
        return "other", ""
    first = toks[0].form.lower()
    if first == "the":
        return "definite", ""
    if first in ("this", "that", "these", "those"):
        return "demonstrative", ""
    if first in ("a", "an"):
        return "other", "indefinite"
    if first in _EN_POSSESSIVE_DETERMINERS:
        return "other", "possessive"
    if toks[0].upos == "PRON" and not any(t.upos == "NOUN" for t in toks):
        return "other", "pronoun"
    if _proper_name_dominant(toks):
        return "other", "proper_name"
    if _has_number_sing(toks):
        return "bare_singular", ""
    return "other", ""


_ZH_CLASSIFIER_TAGS = frozenset({"CL", "M"})


def chinese_form(tokens: list[Token]) -> tuple[str, str]:
    """Deterministic ZH paper-form rule. Returns ``(paper_form, detail)``."""
    toks = list(tokens)
    while toks and toks[0].upos == "ADP":
        toks.pop(0)
    while toks and toks[-1].upos in ("PUNCT", "SYM"):
        toks.pop()
    if not toks:
        return "other", ""
    if any(t.form.startswith(h) for t in toks[:2] for h in _ZH_DEMONSTRATIVE_HEADS):
        return "demonstrative", ""
    if (
        len(toks) >= 2
        and toks[0].upos in ("NUM", "CD")
        and toks[1].upos in _ZH_CLASSIFIER_TAGS
    ):
        return "other", "numeral_classifier"
    if any(t.upos == "PRON" for t in toks) and not any(
        t.upos in ("NOUN", "PROPN") for t in toks
    ):
        return "other", "pronoun"
    if _proper_name_dominant(toks):
        return "other", "proper_name"
    if any(t.form.endswith("的") for t in toks[:-1]):
        return "other", "possessive"
    return "bare", ""


def de_valid_proposal(de_form: str, det_xpos: str, det_deprel: str) -> tuple[str, str]:
    """Deterministic DE validity proposal, mirroring the production
    structural gate (uncontracted rows need ``det_xpos=ART`` and
    ``det_deprel=det``); contracted rows pass by construction."""
    if de_form != "uncontracted":
        return "include", ""
    if det_xpos in ("", "-") or det_deprel in ("", "-"):
        return "uncertain", ""
    if det_xpos != "ART" or det_deprel != "det":
        return "exclude", "not_target_pp"
    return "include", ""


# --- artifact row -------------------------------------------------------------------


def build_deterministic_row(
    source_row: dict[str, str],
    en: SpanRecovery | None,
    zh: SpanRecovery | None,
    en_form: tuple[str, str] | None,
    zh_form: tuple[str, str] | None,
    *,
    pack_stratum: str,
    method_params_id: str,
) -> dict[str, str]:
    """Assemble one machine artifact row on the shared MACHINE_TUPLE
    schema, with the deterministic method id in the model columns."""
    row = project_source_cells(source_row)
    row["pack_stratum"] = pack_stratum
    row["machine_annotation_model"] = DETERMINISTIC_METHOD_ID
    row["machine_annotation_prompt_version"] = method_params_id

    proposal, reason = de_valid_proposal(
        source_row.get("de_form", ""),
        source_row.get("de_det_xpos", ""),
        source_row.get("de_det_deprel", ""),
    )
    row["machine_de_valid_proposal"] = proposal
    row["machine_de_exclusion_reason"] = reason

    for side, recovery, form in (("en", en, en_form), ("zh", zh, zh_form)):
        if recovery is None:
            row[f"machine_{side}_counterpart"] = ""
            row[f"machine_{side}_paper_form"] = ""
            row[f"machine_{side}_detail_form"] = ""
            row[f"machine_{side}_status"] = "not_aligned"
            continue
        aligned = recovery.status == "aligned"
        row[f"machine_{side}_counterpart"] = recovery.span_text if aligned else ""
        paper, detail = form if (form and aligned) else ("", "")
        row[f"machine_{side}_paper_form"] = paper
        row[f"machine_{side}_detail_form"] = detail
        row[f"machine_{side}_status"] = recovery.status
    return {c: row.get(c, "") for c in MACHINE_TUPLE_COLUMNS}


__all__ = [
    "DETERMINISTIC_METHOD_ID",
    "LINKS_SCHEMA_VERSION",
    "EFLOMAL_PARAMETERS",
    "MAX_SPAN_TOKENS",
    "SPAN_STATUSES",
    "Token",
    "ConlluSentence",
    "read_conllu",
    "blocks_for_segment",
    "SideLayout",
    "build_side_layout",
    "load_alignment_records",
    "build_bitext",
    "eflomal_input_line",
    "parse_links_file",
    "intersection_links",
    "sha256_file",
    "SpanRecovery",
    "recover_target_span",
    "english_form",
    "chinese_form",
    "de_valid_proposal",
    "build_deterministic_row",
]
