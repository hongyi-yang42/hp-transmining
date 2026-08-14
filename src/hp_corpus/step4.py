"""Step 4 — Cross-lingual annotation pack builder for Bremmers reproduction.

The datapoint in Step 4 is a **German PP occurrence** (a specific token
range inside one sentence), not a (preposition, noun) type and not a
triple of full sentences. This module anchors on each German occurrence
from the Ch.1–3 extraction TSVs, joins it to the DE↔EN and DE↔ZH
sentence alignments produced earlier in the pipeline, and emits a
human-annotation pack:

  - all_candidates.jsonl  — every German occurrence with full DE/EN/ZH context
  - pilot_20.tsv          — 10 contracted + 10 uncontracted, deterministically chosen
  - pilot_20.summary.json — aggregate counts only (safe for stdout/reports)

Methodology boundaries (see docs/METHODS.md):

  * The output is a **Ch.1–3 paper-eligible annotation pool**, not the
    paper's final 96 trilingual contexts. ``FILTER_CONTRACTED_123`` /
    ``FILTER_PP`` are extraction QA only; they do not gate membership.
  * EN/ZH counterpart spans and forms are filled by a human annotator;
    this module never auto-translates or auto-classifies definiteness.
  * No weak/strong/uniqueness/familiarity labels are produced here.
  * ``crosslingual_map`` proposals never auto-populate annotation fields.

Alignment compatibility: the existing serializer names both sides
``en`` / ``zh`` regardless of the actual languages. We therefore
identify the German side by inspecting segment IDs (lang is the second
underscore-separated field) and additionally accept ``source``/``target``
or ``src``/``tgt`` keys for forward compatibility.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------- constants

DATASET_SCOPE = "ch1_3_annotation_target"
PAPER_FINAL_SAMPLE = False

# Pinned vendor revision. The tracked file ``vendor/conll-extractor.commit``
# is the source of truth; CI uses it to detect accidental upgrades of the
# vendored checkout. The fallback default below is used only when the
# manifest is missing (e.g., running tests against a partial checkout).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_VENDOR_COMMIT_PATH = _REPO_ROOT / "vendor" / "conll-extractor.commit"
_FALLBACK_COMMIT = "4a8a22030ac554d88d54566f68a9382197e43606"


def _load_expected_commit() -> str:
    try:
        return _VENDOR_COMMIT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return _FALLBACK_COMMIT


EXPECTED_CONLL_EXTRACTOR_COMMIT = _load_expected_commit()

SEGMENT_ID_LANG_RE = re.compile(r"^[a-z0-9]+_([a-z]{2,3})_ch\d{2}_p\d{4}_s\d{3}$")

# Contracted preposition → canonical surface form. Single source of truth
# for the project; if the paper introduces additional contractions, add
# them here rather than at call sites.
CONTRACTED_PREP_NORMALIZATION: dict[str, str] = {
    "am": "an",
    "ans": "an",
    "aufs": "auf",
    "aufm": "auf",
    "ausm": "aus",
    "beim": "bei",
    "durchs": "durch",
    "fürs": "für",
    "fürn": "für",
    "fürm": "für",
    "gegens": "gegen",
    "hinterm": "hinter",
    "hinters": "hinter",
    "im": "in",
    "ins": "in",
    "überm": "über",
    "übers": "über",
    "übern": "über",
    "ums": "um",
    "unterm": "unter",
    "unters": "unter",
    "untern": "unter",
    "vom": "von",
    "vorm": "vor",
    "vors": "vor",
    "zum": "zu",
    "zur": "zu",
}

# Uncontracted preposition inventory — the canonical surface forms that
# the paper's extractor treats as the "uncontracted" list (vendored
# ``conll_extractor.prepositions.data.PREPOSITIONS``). Duplicated here so
# this module stays self-contained (no vendor import).
UNCONTRACTED_PREPOSITIONS: frozenset[str] = frozenset(
    {"an", "auf", "aus", "bei", "durch", "für", "gegen", "hinter",
     "in", "über", "unter", "von", "zu"}
)

# Paper §2.2.1 selection filter: keep a PP only if its canonical preposition
# appears in BOTH inventories — i.e. the preposition has at least one
# contracted form AND is itself an uncontracted preposition. Excludes
# ``um`` and ``vor`` contractions only (neither is in the uncontracted
# PREPOSITIONS list). Computed once at module load.
PAPER_SHARED_PREPOSITIONS: frozenset[str] = frozenset(
    set(CONTRACTED_PREP_NORMALIZATION.values()) & UNCONTRACTED_PREPOSITIONS
)

# Paper's Table A (German form distribution, full novel). Reference numbers
# for comparison against our Ch.1-3 subset; lives here so any caller can
# cite the paper without re-deriving or duplicating the constants.
PAPER_TABLE_A: dict[str, Any] = {
    "contracted": 40,
    "uncontracted": 56,
    "total": 96,
    "contracted_pct": 41.5,
    "uncontracted_pct": 58.5,
}

# Alignment JSONL may use any of these field-name pairs. Both sides get
# inspected by segment ID, but we still pick up the side lists using
# these known names.
ALIGNMENT_SIDE_KEYS = ("en", "zh", "source", "target", "src", "tgt")

PILOT_DEFAULT_N_CONTRACTED = 10
PILOT_DEFAULT_N_UNCONTRACTED = 10

# --- TSV column order (source columns are immutable; annotation columns editable)

SOURCE_COLUMNS: tuple[str, ...] = (
    "datapoint_id",
    "dataset_scope",
    "paper_final_sample",
    "chapter",
    "de_sentence_id",
    "de_token_start",
    "de_token_end",
    "de_pp_surface",
    "de_sentence_text",
    "de_prep_surface",
    "de_prep_normalized",
    "de_head_lemma",
    "de_form",
    "author_resource_match",
    "minimal_pair_group",
    "en_sentence_ids",
    "en_aligned_text",
    "en_alignment_cardinality",
    "en_alignment_status",
    "en_alignment_confidence",
    "zh_sentence_ids",
    "zh_aligned_text",
    "zh_alignment_cardinality",
    "zh_alignment_status",
    "zh_alignment_confidence",
    "pilot_selected",
    "pilot_selection_reason",
    "source_row_sha256",
)

EDITABLE_COLUMNS: tuple[str, ...] = (
    # German-candidate confirmation. Annotator decides whether each row is
    # a legitimate German PP for the annotation pool. ``author_resource_match``
    # in SOURCE_COLUMNS stays extraction-QA-only and is NOT used for exclusion.
    "de_candidate_decision",
    "de_exclusion_reason",
    "de_candidate_notes",
    # English annotation: alignment QC first, then span/form/relation.
    "en_alignment_qc",
    "en_alignment_notes",
    "en_alignment_relation",
    "en_span_text",
    "en_char_ranges",
    "en_form",
    "en_confidence",
    "en_notes",
    # Mandarin annotation: same block structure as EN.
    "zh_alignment_qc",
    "zh_alignment_notes",
    "zh_alignment_relation",
    "zh_span_text",
    "zh_char_ranges",
    "zh_form",
    "zh_confidence",
    "zh_notes",
    # Cross-lingual / process metadata
    "annotator",
    "annotation_status",
    "adjudication_status",
    "general_notes",
)

ALL_TSV_COLUMNS: tuple[str, ...] = SOURCE_COLUMNS + EDITABLE_COLUMNS

# Controlled vocabularies — the validator enforces these.
ALIGNMENT_RELATIONS = frozenset({"direct", "paraphrase", "pronominal", "omitted", "uncertain"})
EN_FORMS = frozenset(
    {
        "definite",
        "bare_singular",
        "demonstrative",
        "indefinite",
        "possessive",
        "pronoun",
        "proper_name",
        "other",
        "omitted",
        "uncertain",
    }
)
ZH_FORMS = frozenset(
    {
        "bare",
        "demonstrative",
        "numeral_classifier",
        "possessive",
        "pronoun",
        "proper_name",
        "other",
        "omitted",
        "uncertain",
    }
)
CONFIDENCE_LEVELS = frozenset({"high", "medium", "low"})
ANNOTATION_STATUSES = frozenset({"", "unstarted", "in_progress", "complete"})
ADJUDICATION_STATUSES = frozenset({"", "pending", "adjudicated", "disputed"})

# German-candidate decision vocab. ``de_candidate_decision`` is blank until
# the annotator confirms. ``de_exclusion_reason`` is filled only when the
# decision is ``exclude``.
DE_CANDIDATE_DECISIONS = frozenset({"include", "exclude", "uncertain"})
DE_EXCLUSION_REASONS = frozenset(
    {"not_target_pp", "not_definite", "extraction_error", "duplicate", "other"}
)

# Alignment-QC vocab. ``assumed_ok`` is the builder-written default; the
# annotator promotes to ``confirmed`` once a counterpart has been located
# (or its absence verified). ``incorrect`` and ``uncertain`` route the row
# to the realignment queue and block annotation completion.
ALIGNMENT_QC_VALUES = frozenset({"assumed_ok", "confirmed", "incorrect", "uncertain"})

# Builder-written default for the alignment-QC columns. Kept as a constant
# so the writer, validator, and tests agree on what "initial state" means
# now that one editable column is no longer blank by default.
DEFAULT_ALIGNMENT_QC = "assumed_ok"

# Columns the builder pre-fills with a default value (rather than blank).
# Used by the validator's initial-state detection: a row is "unannotated"
# iff every editable cell equals its builder default.
BUILDER_DEFAULT_EDITABLE: dict[str, str] = {
    "en_alignment_qc": DEFAULT_ALIGNMENT_QC,
    "zh_alignment_qc": DEFAULT_ALIGNMENT_QC,
}


# --------------------------------------------------------------------- helpers


def lang_from_segment_id(segment_id: str) -> str | None:
    """hp1_de_ch01_p0001_s001 → 'de'. Returns None if the ID does not match
    the canonical pattern."""
    m = SEGMENT_ID_LANG_RE.match(segment_id)
    return m.group(1) if m else None


class Step4Error(Exception):
    """Base class for Step 4 builder errors that must surface as nonzero exit."""


class MissingInputsError(Step4Error):
    """One or more required input files for the requested chapters are
    missing, empty, or header-only. The builder refuses to emit a
    partial-corpus TSV — every requested chapter must have a complete
    input set so the annotation target is always cross-lingual.
    """

    def __init__(self, missing: list[str], *, kind: str = "missing"):
        self.missing = missing
        self.kind = kind
        super().__init__(
            f"{kind} inputs for requested chapters ({len(missing)}): " + ", ".join(missing)
        )


class UnresolvedSegmentIdError(Step4Error):
    """A DE sentence_id from the extraction TSV does not resolve to a segment
    in the segmented JSONL, or a target segment ID listed in an alignment
    record does not resolve. Indicates an upstream pipeline integrity
    failure; the builder refuses to emit a row with empty sentence text.
    """


def normalize_contracted_prep(prep: str) -> str:
    """Map a contracted preposition surface form to its canonical preposition.
    Non-contracted forms and unknown contractions pass through unchanged."""
    return CONTRACTED_PREP_NORMALIZATION.get(prep, prep)


def minimal_pair_group_key(prep_normalized: str, head_lemma: str) -> str:
    """Stable key for grouping contracted + uncontracted occurrences that
    share the same (canonical preposition, head noun lemma)."""
    return f"{prep_normalized}|{head_lemma}"


def compute_source_row_sha256(candidate: dict[str, Any]) -> str:
    """Stable SHA-256 over the immutable source columns of a candidate.

    The hash covers the column-name + value of every SOURCE_COLUMNS entry
    in fixed order, joined by '\\x1f' (ASCII unit separator) so multi-
    column collisions are impossible. The hash lets the validator detect
    whether a human editor accidentally modified a source column.
    """
    parts: list[str] = []
    for col in SOURCE_COLUMNS:
        if col == "source_row_sha256":
            continue
        val = candidate.get(col)
        if isinstance(val, list):
            val_repr = json.dumps(val, ensure_ascii=False, sort_keys=True)
        elif isinstance(val, bool):
            val_repr = "true" if val else "false"
        elif val is None:
            val_repr = ""
        else:
            val_repr = str(val)
        parts.append(f"{col}={val_repr}")
    blob = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------------- loaders


@dataclass
class _SegmentRecord:
    id: str
    text: str


def _load_segments(
    segmented_dir: Path, lang: str, chapters: Iterable[int]
) -> dict[str, _SegmentRecord]:
    """Load segmented JSONL for one language across the requested chapters.

    Returns segment_id → _SegmentRecord.
    """
    out: dict[str, _SegmentRecord] = {}
    for ch in chapters:
        path = segmented_dir / f"hp1_{lang}_ch{ch:02d}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                out[rec["id"]] = _SegmentRecord(id=rec["id"], text=rec["text"])
    return out


@dataclass
class _AlignmentSide:
    """One side of an alignment record, identified by segment IDs rather
    than by JSON key — the same JSONL serializes DE↔EN and DE↔ZH using
    the same ``en``/``zh`` field names, so the key is unreliable."""

    lang: str
    sentence_ids: list[str]


@dataclass
class _AlignmentRecord:
    align_id: str
    sides: tuple[_AlignmentSide, _AlignmentSide]
    type_str: str
    confidence: float

    def side_for_lang(self, lang: str) -> _AlignmentSide | None:
        for s in self.sides:
            if s.lang == lang:
                return s
        return None


def _identify_alignment_sides(
    record: dict[str, Any], *, expected_langs: set[str]
) -> list[_AlignmentSide]:
    """Walk all known side-key names and bucket their IDs by language.

    The DP-produced JSONL always names both sides (``en``/``zh`` in the
    current serializer), but a side may be empty for 1:0 / 0:1 records.
    We:

      * identify non-empty sides by segment-ID prefix
      * infer the language of an empty side from ``expected_langs`` minus
        the language of the non-empty side
      * accept either 1 or 2 resulting sides (1 ⇒ a 1:0 / 0:1 record)

    Raises ValueError on genuinely corrupt records (mixed-language side,
    side that doesn't match either expected language, ambiguous empty
    side when both expected languages coincide, duplicate IDs across
    fields with conflicting content).
    """
    sides: list[_AlignmentSide] = []
    seen_ids: set[str] = set()
    for key in ALIGNMENT_SIDE_KEYS:
        if key not in record:
            continue
        ids = record[key]
        if not isinstance(ids, list):
            raise ValueError(f"alignment {record.get('align_id')}: field '{key}' is not a list")
        if not ids:
            # Empty side — record its existence so we can label it later,
            # but we can't infer its language until we know the other side.
            sides.append(_AlignmentSide(lang="", sentence_ids=[]))
            continue
        langs = {lang_from_segment_id(i) for i in ids}
        if None in langs:
            raise ValueError(
                f"alignment {record.get('align_id')}: side '{key}' contains an "
                f"unparseable segment id"
            )
        if len(langs) > 1:
            raise ValueError(
                f"alignment {record.get('align_id')}: side '{key}' mixes languages {langs}"
            )
        lang = next(iter(langs))
        if expected_langs and lang not in expected_langs:
            raise ValueError(
                f"alignment {record.get('align_id')}: side '{key}' language {lang!r} "
                f"not in expected {sorted(expected_langs)}"
            )
        # Deduplicate by id-set so we don't double-count when the same side
        # is repeated under both legacy and new key names (e.g. en + source).
        id_tuple = tuple(ids)
        if any(id_tuple == tuple(s.sentence_ids) for s in sides if s.sentence_ids):
            continue
        for i in id_tuple:
            if i in seen_ids:
                raise ValueError(
                    f"alignment {record.get('align_id')}: segment id {i!r} appears in "
                    f"multiple side fields with conflicting content"
                )
            seen_ids.add(i)
        sides.append(_AlignmentSide(lang=lang, sentence_ids=list(id_tuple)))

    # Resolve any empty side's language by elimination from expected_langs.
    other_langs = {s.lang for s in sides if s.lang}
    for s in sides:
        if s.lang:
            continue
        candidates = expected_langs - other_langs
        if len(candidates) == 1:
            s.lang = next(iter(candidates))
        elif len(candidates) == 0 and len(expected_langs) == 1:
            # Degenerate case (self-alignment file): leave empty.
            pass
        # Else: ambiguous — leave blank, downstream may ignore.

    if not sides:
        raise ValueError(f"alignment {record.get('align_id')}: no side fields found")
    if len(sides) > 2:
        raise ValueError(
            f"alignment {record.get('align_id')}: expected at most 2 sides, got {len(sides)}"
        )
    return sides


def _load_alignments(
    aligned_dir: Path, src_lang: str, tgt_lang: str, chapters: Iterable[int]
) -> dict[str, _AlignmentRecord]:
    """Load DE↔{tgt} alignment JSONL files. Returns a map from each src_lang
    segment id to its alignment record.

    1:0 records (where the src side is empty) are skipped — they anchor no
    DE occurrence. 0:1 records (where the tgt side is empty) are kept so
    that callers can mark 'missing target' on the corresponding src_id.
    Conflicting duplicate src IDs across records raise ValueError.
    """
    expected = {src_lang, tgt_lang}
    out: dict[str, _AlignmentRecord] = {}
    for ch in chapters:
        path = aligned_dir / f"hp1_{src_lang}_{tgt_lang}_ch{ch:02d}.jsonl"
        if not path.exists():
            # Try reversed file name (DE↔EN file is named de_en regardless of
            # which side the serializer put first).
            path = aligned_dir / f"hp1_{tgt_lang}_{src_lang}_ch{ch:02d}.jsonl"
            if not path.exists():
                continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                try:
                    sides = _identify_alignment_sides(rec, expected_langs=expected)
                except ValueError as e:
                    raise ValueError(f"{path.name}: {e}") from e
                # Pad to a 2-tuple so callers can index predictably.
                if len(sides) == 1:
                    sides.append(_AlignmentSide(lang="", sentence_ids=[]))
                record = _AlignmentRecord(
                    align_id=rec["align_id"],
                    sides=(sides[0], sides[1]),
                    type_str=rec.get("type", ""),
                    confidence=float(rec.get("confidence", 0.0)),
                )
                src_side = record.side_for_lang(src_lang)
                if src_side is None or not src_side.sentence_ids:
                    # 0:1 record (no src sentence) — anchors no datapoint.
                    continue
                for sid in src_side.sentence_ids:
                    if sid in out:
                        raise ValueError(
                            f"{path.name}: source segment {sid!r} appears in two "
                            f"alignment records ({out[sid].align_id} and {record.align_id})"
                        )
                    out[sid] = record
    return out


def _read_extraction_tsv(path: Path) -> list[dict[str, str]]:
    """Read an extraction TSV (with the new coordinate fields) into row dicts."""
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(r) for r in reader]


# --------------------------------------------------------------------- builder


def _aligned_text(side: _AlignmentSide | None, segments: dict[str, _SegmentRecord]) -> str:
    if side is None:
        return ""
    parts: list[str] = []
    for sid in side.sentence_ids:
        seg = segments.get(sid)
        if seg is None:
            parts.append("")
        else:
            parts.append(seg.text)
    return " ".join(p for p in parts if p)


def _cardinality(side_src: _AlignmentSide | None, side_tgt: _AlignmentSide | None) -> str:
    n_src = len(side_src.sentence_ids) if side_src else 0
    n_tgt = len(side_tgt.sentence_ids) if side_tgt else 0
    return f"{n_src}:{n_tgt}"


def _status_for(de_id: str, alignments: dict[str, _AlignmentRecord], tgt_lang: str) -> str:
    """Status string for one tgt_lang side. ``aligned`` if the DE sentence has
    a record whose tgt side contains tgt_lang; ``missing`` otherwise.
    """
    rec = alignments.get(de_id)
    if rec is None:
        return "missing"
    if rec.side_for_lang(tgt_lang) is None:
        return "missing"
    return "aligned"


def _has_tsv_body(path: Path) -> bool:
    """True if the TSV has at least one non-header row with any non-empty cell."""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # header
            if line.strip():
                return True
    return False


def _assert_inputs_present(
    chapters: list[int],
    *,
    extraction_dir: Path,
    segmented_dir: Path,
    aligned_dir: Path,
    zero_hits_ok: frozenset[tuple[int, str]] = frozenset(),
) -> None:
    """Verify that every requested chapter has the full input set.

    Each chapter must contribute:

      * ``hp1_de_ch{NN}_contracted.tsv`` — non-empty body
      * ``hp1_de_ch{NN}_uncontracted.tsv`` — non-empty body
      * ``hp1_{de,en,zh}_ch{NN}.jsonl`` — exists, non-empty
      * ``hp1_de_{en,zh}_ch{NN}.jsonl`` — exists, non-empty

    Header-only TSVs are rejected by default: for Ch.1–3 every form is
    known to be populated, so a header-only file signals an upstream
    failure rather than a legitimately-empty chapter. For the full-novel
    scope, a (chapter, form) pair recorded as ``zero_hits_ok`` in the
    extraction manifest may legitimately have a header-only TSV — pass
    those pairs via ``zero_hits_ok`` (see
    ``scripts/build_full_novel_annotation.py``). Zero-byte files always
    fail regardless: an empty file is corruption, not a zero-hit form.
    """
    missing: list[str] = []
    bad: list[str] = []

    for ch in chapters:
        for kind in ("contracted", "uncontracted"):
            p = extraction_dir / f"hp1_de_ch{ch:02d}_{kind}.tsv"
            if not p.exists():
                missing.append(str(p))
            elif p.stat().st_size == 0:
                bad.append(f"{p} (empty)")
            elif not _has_tsv_body(p) and (ch, kind) not in zero_hits_ok:
                bad.append(f"{p} (header-only)")
        for lang in ("de", "en", "zh"):
            p = segmented_dir / f"hp1_{lang}_ch{ch:02d}.jsonl"
            if not p.exists():
                missing.append(str(p))
            elif p.stat().st_size == 0:
                bad.append(f"{p} (empty)")
        for tgt in ("en", "zh"):
            p = aligned_dir / f"hp1_de_{tgt}_ch{ch:02d}.jsonl"
            if not p.exists():
                # Try reversed file name (DE↔EN file is named de_en regardless
                # of which side the serializer put first).
                alt = aligned_dir / f"hp1_{tgt}_de_ch{ch:02d}.jsonl"
                if not alt.exists():
                    missing.append(str(p))
            elif p.stat().st_size == 0:
                bad.append(f"{p} (empty)")

    if missing:
        raise MissingInputsError(missing, kind="missing")
    if bad:
        raise MissingInputsError(bad, kind="malformed")


def build_candidates(
    *,
    extraction_dir: Path,
    segmented_dir: Path,
    aligned_dir: Path,
    chapters: Iterable[int],
    zero_hits_ok: frozenset[tuple[int, str]] = frozenset(),
) -> list[dict[str, Any]]:
    """Build the full candidate list for all chapters.

    Each candidate is a German PP occurrence with full DE/EN/ZH context
    plus alignment metadata. Pilot selection happens later.

    ``zero_hits_ok`` names (chapter, form) pairs whose extraction TSV is
    legitimately header-only because the full-novel extraction manifest
    recorded ``status="zero_hits_ok"`` for them; all other header-only
    TSVs are rejected as upstream failures.

    Raises :class:`MissingInputsError` if any requested chapter is missing
    a required input file, and :class:`UnresolvedSegmentIdError` if a
    DE sentence_id from the extraction TSV cannot be resolved against the
    segmented JSONL. Both errors signal upstream integrity failures; the
    builder never emits a partial-corpus or empty-sentence TSV.
    """
    chapters = list(chapters)
    _assert_inputs_present(
        chapters,
        extraction_dir=extraction_dir,
        segmented_dir=segmented_dir,
        aligned_dir=aligned_dir,
        zero_hits_ok=zero_hits_ok,
    )

    de_segments = _load_segments(segmented_dir, "de", chapters)
    en_segments = _load_segments(segmented_dir, "en", chapters)
    zh_segments = _load_segments(segmented_dir, "zh", chapters)

    de_en_alignments = _load_alignments(aligned_dir, "de", "en", chapters)
    de_zh_alignments = _load_alignments(aligned_dir, "de", "zh", chapters)

    candidates: list[dict[str, Any]] = []
    for ch in chapters:
        for kind, de_form in (("contracted", "contracted"), ("uncontracted", "uncontracted")):
            tsv_path = extraction_dir / f"hp1_de_ch{ch:02d}_{kind}.tsv"
            if not tsv_path.exists():
                continue
            for row in _read_extraction_tsv(tsv_path):
                sent_id = row["sentence_id"]
                de_seg = de_segments.get(sent_id)
                if de_seg is None:
                    # Data-integrity failure — extraction TSV references a
                    # sentence the segmented JSONL doesn't have. Fail loudly
                    # rather than emit a row with empty sentence text.
                    raise UnresolvedSegmentIdError(
                        f"DE sentence id {sent_id!r} (chapter {ch}, {kind}) "
                        f"is not present in {segmented_dir}/hp1_de_ch{ch:02d}.jsonl"
                    )
                de_sentence_text = de_seg.text

                prep_surface = row["prep"]
                prep_normalized = normalize_contracted_prep(prep_surface)
                head_lemma = row["noun"]
                token_start = int(row["pp_token_start"])
                token_end = int(row["pp_token_end"])

                en_rec = de_en_alignments.get(sent_id)
                zh_rec = de_zh_alignments.get(sent_id)

                en_side = en_rec.side_for_lang("en") if en_rec else None
                zh_side = zh_rec.side_for_lang("zh") if zh_rec else None

                # Cardinality is from DE's perspective: n_de : n_tgt.
                de_side_en = en_rec.side_for_lang("de") if en_rec else None
                de_side_zh = zh_rec.side_for_lang("de") if zh_rec else None

                candidate = {
                    "datapoint_id": _make_datapoint_id(ch, sent_id, token_start, token_end),
                    "dataset_scope": DATASET_SCOPE,
                    "paper_final_sample": PAPER_FINAL_SAMPLE,
                    "chapter": ch,
                    # German occurrence
                    "de_sentence_id": sent_id,
                    "de_token_start": token_start,
                    "de_token_end": token_end,
                    "de_pp_surface": row["pp_surface"],
                    "de_sentence_text": de_sentence_text,
                    "de_prep_surface": prep_surface,
                    "de_prep_normalized": prep_normalized,
                    "de_head_lemma": head_lemma,
                    "de_form": de_form,
                    "author_resource_match": row.get("in_filter") == "Y",
                    "minimal_pair_group": minimal_pair_group_key(prep_normalized, head_lemma),
                    # EN
                    "en_sentence_ids": list(en_side.sentence_ids) if en_side else [],
                    "en_aligned_text": _aligned_text(en_side, en_segments),
                    "en_alignment_cardinality": _cardinality(de_side_en, en_side),
                    "en_alignment_status": _status_for(sent_id, de_en_alignments, "en"),
                    "en_alignment_confidence": en_rec.confidence if en_rec else 0.0,
                    # ZH
                    "zh_sentence_ids": list(zh_side.sentence_ids) if zh_side else [],
                    "zh_aligned_text": _aligned_text(zh_side, zh_segments),
                    "zh_alignment_cardinality": _cardinality(de_side_zh, zh_side),
                    "zh_alignment_status": _status_for(sent_id, de_zh_alignments, "zh"),
                    "zh_alignment_confidence": zh_rec.confidence if zh_rec else 0.0,
                    # Pilot (filled later by select_pilot)
                    "pilot_selected": False,
                    "pilot_selection_reason": "",
                    # Hash slot — populated at write time so it always reflects
                    # the candidate's CURRENT source-column state (which changes
                    # when select_pilot flips pilot_selected to True).
                    "source_row_sha256": "",
                }
                candidates.append(candidate)

    # Stable-sort the whole list so downstream consumers (JSONL writer,
    # pilot selector) iterate deterministically without re-sorting.
    candidates.sort(key=_stable_sort_key)
    return candidates


def _make_datapoint_id(chapter: int, sentence_id: str, token_start: int, token_end: int) -> str:
    """Deterministic, human-readable occurrence ID.

    Identity = (chapter, sentence_id, token_start, token_end). Two PPs
    in the same sentence with the same (prep, noun) get different IDs
    because their token ranges differ.
    """
    return f"dp_ch{chapter:02d}_{sentence_id}_t{token_start}-{token_end}"


def _stable_sort_key(cand: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(cand["chapter"]),
        str(cand["de_sentence_id"]),
        int(cand["de_token_start"]),
        int(cand["de_token_end"]),
    )


# --------------------------------------------------------------------- pilot


def select_pilot(
    candidates: list[dict[str, Any]],
    *,
    n_contracted: int = PILOT_DEFAULT_N_CONTRACTED,
    n_uncontracted: int = PILOT_DEFAULT_N_UNCONTRACTED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically select ``n_contracted`` + ``n_uncontracted`` candidates.

    Selection runs over candidates that have BOTH an EN and a ZH
    alignment (status == 'aligned'). Priority:

      1. ``minimal_pair`` — groups that contain both a contracted and an
         uncontracted occurrence. Each such group contributes at most one
         of each (the first by stable sort).
      2. ``author_match`` — candidates whose ``author_resource_match`` is True.
      3. ``stable_fill`` — remainder in stable sort order.

    Returns the selected list (in selection order, not stable-sort order)
    plus a summary dict describing the per-reason counts. Raises
    ``InsufficientCandidatesError`` if either bucket cannot be filled.
    """
    eligible = [
        c
        for c in candidates
        if c["en_alignment_status"] == "aligned" and c["zh_alignment_status"] == "aligned"
    ]

    by_form: dict[str, list[dict[str, Any]]] = {"contracted": [], "uncontracted": []}
    for c in eligible:
        by_form[c["de_form"]].append(c)
    # Stable-sort each bucket so 'first by stable sort' is well-defined.
    by_form["contracted"].sort(key=_stable_sort_key)
    by_form["uncontracted"].sort(key=_stable_sort_key)

    selected: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()
    counts = {"minimal_pair": 0, "author_match": 0, "stable_fill": 0}
    per_form_counts = {
        "contracted": {"minimal_pair": 0, "author_match": 0, "stable_fill": 0},
        "uncontracted": {"minimal_pair": 0, "author_match": 0, "stable_fill": 0},
    }
    target = {"contracted": n_contracted, "uncontracted": n_uncontracted}
    form_filled = {"contracted": 0, "uncontracted": 0}

    def _pick(cand: dict[str, Any], reason: str) -> None:
        if form_filled[cand["de_form"]] >= target[cand["de_form"]]:
            return
        if cand["datapoint_id"] in chosen_ids:
            return
        cand["pilot_selected"] = True
        cand["pilot_selection_reason"] = reason
        selected.append(cand)
        chosen_ids.add(cand["datapoint_id"])
        form_filled[cand["de_form"]] += 1
        counts[reason] += 1
        per_form_counts[cand["de_form"]][reason] += 1

    # ----- pass 1: minimal-pair groups -----
    groups: dict[str, list[dict[str, Any]]] = {}
    for c in eligible:
        groups.setdefault(c["minimal_pair_group"], []).append(c)
    minimal_pair_group_keys = sorted(groups.keys())
    for gk in minimal_pair_group_keys:
        members = groups[gk]
        forms_present = {m["de_form"] for m in members}
        if forms_present != {"contracted", "uncontracted"}:
            continue
        # First contracted + first uncontracted (stable order within group).
        first_c = next(m for m in members if m["de_form"] == "contracted")
        first_u = next(m for m in members if m["de_form"] == "uncontracted")
        _pick(first_c, "minimal_pair")
        _pick(first_u, "minimal_pair")
        if (
            form_filled["contracted"] >= n_contracted
            and form_filled["uncontracted"] >= n_uncontracted
        ):
            break

    # ----- pass 2: author_resource_match = True -----
    if form_filled["contracted"] < n_contracted or form_filled["uncontracted"] < n_uncontracted:
        for c in sorted(eligible, key=_stable_sort_key):
            if not c["author_resource_match"]:
                continue
            _pick(c, "author_match")
            if (
                form_filled["contracted"] >= n_contracted
                and form_filled["uncontracted"] >= n_uncontracted
            ):
                break

    # ----- pass 3: stable-fill remainder -----
    if form_filled["contracted"] < n_contracted or form_filled["uncontracted"] < n_uncontracted:
        for c in sorted(eligible, key=_stable_sort_key):
            _pick(c, "stable_fill")
            if (
                form_filled["contracted"] >= n_contracted
                and form_filled["uncontracted"] >= n_uncontracted
            ):
                break

    underfilled = {f: target[f] - form_filled[f] for f in form_filled if form_filled[f] < target[f]}
    summary = {
        "selected_total": len(selected),
        "by_reason": dict(counts),
        "by_reason_per_form": per_form_counts,
        "eligible_total": len(eligible),
        "eligible_contracted": len(by_form["contracted"]),
        "eligible_uncontracted": len(by_form["uncontracted"]),
        "target": dict(target),
        "filled": dict(form_filled),
        "underfilled": underfilled,
    }
    if underfilled:
        raise InsufficientCandidatesError(summary)
    return selected, summary


class InsufficientCandidatesError(Exception):
    """Raised when the pilot cannot reach the requested 10+10 size."""

    def __init__(self, summary: dict[str, Any]):
        self.summary = summary
        super().__init__(
            f"insufficient candidates: filled={summary['filled']}, "
            f"target={summary['target']}, eligible={summary['eligible_total']} "
            f"(contracted={summary['eligible_contracted']}, "
            f"uncontracted={summary['eligible_uncontracted']})"
        )


def select_ch1_3_annotation_pool(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the Ch.1–3 paper-eligible annotation pool.

    Applies the paper's preposition-inventory filter (Bremmers et al. 2022,
    §2.2.1): keeps candidates whose ``de_prep_normalized`` is in
    :data:`PAPER_SHARED_PREPOSITIONS` — i.e. the canonical preposition has
    both a contracted form and an uncontracted form in the language.

    The result is the **annotation pool**, not the paper's final 96.
    Membership in the 96 is a downstream human decision; this function only
    removes rows whose preposition is structurally ineligible (e.g.
    contracted ``ums`` / ``vorm`` — ``um`` and ``vor`` are not in the
    paper's PREPOSITIONS list, so they cannot pair with an uncontracted
    form).

    For Ch.1–3 every surviving occurrence is kept; the Ch.4+ minimal-pair
    restriction (contracted PPs in Ch.4+ are kept only when an uncontracted
    counterpart with the same canonical preposition + head noun lemma
    exists) does NOT apply here because Ch.4+ is out of scope. The
    minimal-pair grouping is still computed
    (see :data:`minimal_pair_groups_with_both_forms` in the summary) so the
    Ch.4+ extension can resume from this output without re-deriving it.

    Returns ``(pool, summary)``. Does not mutate the input list or the
    candidate dicts; the pool holds references to the same dict objects.

    The summary uses the keys ``pool_total`` and ``ineligible_total`` (not
    ``selected_total`` / ``dropped_total``) to avoid implying the result is
    the paper's final sample. The old names remain available as aliases for
    backward compatibility.
    """
    pool: list[dict[str, Any]] = []
    by_form: dict[str, int] = {"contracted": 0, "uncontracted": 0}
    dropped_by_form: dict[str, int] = {"contracted": 0, "uncontracted": 0}
    by_chapter: dict[int, int] = {}
    # Minimal-pair groups (canonical_prep|head_lemma) restricted to the
    # selected sample — counts groups that have BOTH forms after filtering.
    selected_groups: dict[str, set[str]] = {}
    for c in candidates:
        if c["de_prep_normalized"] in PAPER_SHARED_PREPOSITIONS:
            pool.append(c)
            by_form[c["de_form"]] = by_form.get(c["de_form"], 0) + 1
            by_chapter[c["chapter"]] = by_chapter.get(c["chapter"], 0) + 1
            selected_groups.setdefault(c["minimal_pair_group"], set()).add(c["de_form"])
        else:
            dropped_by_form[c["de_form"]] = dropped_by_form.get(c["de_form"], 0) + 1

    groups_with_both = sum(1 for forms in selected_groups.values() if len(forms) == 2)

    summary = {
        "candidate_total": len(candidates),
        "pool_total": len(pool),
        "ineligible_total": len(candidates) - len(pool),
        # Backward-compat aliases (deprecated; new code should use the keys above).
        "selected_total": len(pool),
        "dropped_total": len(candidates) - len(pool),
        "by_form": by_form,
        "dropped_by_form": dropped_by_form,
        "by_chapter": {str(k): v for k, v in sorted(by_chapter.items())},
        "shared_prepositions": sorted(PAPER_SHARED_PREPOSITIONS),
        "minimal_pair_groups_in_sample": len(selected_groups),
        "minimal_pair_groups_with_both_forms": groups_with_both,
    }
    return pool, summary


# Backward-compat alias. New code should call
# :func:`select_ch1_3_annotation_pool`; this name is retained because
# earlier tests and scripts still reference it.
select_paper_sample = select_ch1_3_annotation_pool


# --------------------------------------------------------------------- writers


def write_candidates_jsonl(candidates: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in candidates:
            # Refresh the source hash so it reflects the candidate's current
            # state (pilot selection may have flipped pilot_selected).
            c["source_row_sha256"] = compute_source_row_sha256(c)
            f.write(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def write_pilot_tsv(
    selected: list[dict[str, Any]],
    path: Path,
    *,
    scope_override: str | None = None,
) -> Path:
    """Write the human-annotation TSV. Source columns are written from the
    candidate dict; editable columns are blank (waiting for the annotator)
    except for builder-initialized QC defaults (``{en,zh}_alignment_qc
    = assumed_ok``) — see :data:`BUILDER_DEFAULT_EDITABLE`.

    The source-row SHA-256 is recomputed immediately before writing each row
    so it always reflects the candidate's current source-column state.

    If ``scope_override`` is given, every row's ``dataset_scope`` column
    is set to that value (and the source-row hash reflects the override).
    The override is applied in place on each candidate dict, matching the
    existing ``source_row_sha256`` mutation pattern; callers that need the
    original scope preserved should pass copies.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(ALL_TSV_COLUMNS),
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        w.writeheader()
        for cand in selected:
            if scope_override is not None:
                cand["dataset_scope"] = scope_override
            cand["source_row_sha256"] = compute_source_row_sha256(cand)
            row: dict[str, Any] = {}
            for col in ALL_TSV_COLUMNS:
                if col in SOURCE_COLUMNS:
                    val = cand.get(col, "")
                    if isinstance(val, list):
                        val = json.dumps(val, ensure_ascii=False)
                    elif isinstance(val, bool):
                        val = "true" if val else "false"
                    elif val is None:
                        val = ""
                    else:
                        val = str(val)
                    row[col] = val
                elif col in BUILDER_DEFAULT_EDITABLE:
                    # Builder-initialized editable default (currently only
                    # {en,zh}_alignment_qc = assumed_ok). Annotators may
                    # overwrite this cell; the validator's initial-state
                    # detection treats the default as equivalent to blank.
                    row[col] = BUILDER_DEFAULT_EDITABLE[col]
                else:
                    row[col] = ""
            w.writerow(row)
    return path


def write_summary_json(summary: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return path


def summarize_candidates(
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]] | None,
    pilot_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Aggregate counts only — safe for stdout/reports. No text lemmas,
    no surface forms, no segment IDs."""
    by_form: dict[str, int] = {"contracted": 0, "uncontracted": 0}
    by_chapter: dict[int, int] = {}
    by_author_match = {True: 0, False: 0}
    en_missing = 0
    zh_missing = 0
    en_1_to_n = 0  # 1:2, 1:3 (one DE → many EN)
    zh_1_to_n = 0
    en_n_to_1 = 0  # 2:1, 3:1 — DE side may be in a multi-DE record
    zh_n_to_1 = 0
    for c in candidates:
        by_form[c["de_form"]] = by_form.get(c["de_form"], 0) + 1
        by_chapter[c["chapter"]] = by_chapter.get(c["chapter"], 0) + 1
        by_author_match[c["author_resource_match"]] += 1
        if c["en_alignment_status"] == "missing":
            en_missing += 1
        if c["zh_alignment_status"] == "missing":
            zh_missing += 1
        en_card = c["en_alignment_cardinality"]
        zh_card = c["zh_alignment_cardinality"]
        if en_card.startswith("1:"):
            try:
                n = int(en_card.split(":")[1])
                if n >= 2:
                    en_1_to_n += 1
            except ValueError:
                pass
        if zh_card.startswith("1:"):
            try:
                n = int(zh_card.split(":")[1])
                if n >= 2:
                    zh_1_to_n += 1
            except ValueError:
                pass
        if en_card.endswith(":1") and en_card != "1:1" and en_card != "0:1":
            en_n_to_1 += 1
        if zh_card.endswith(":1") and zh_card != "1:1" and zh_card != "0:1":
            zh_n_to_1 += 1
    minimal_pair_groups = {c["minimal_pair_group"] for c in candidates}
    groups_with_both: set[str] = set()
    by_form_per_group: dict[str, set[str]] = {}
    for c in candidates:
        s = by_form_per_group.setdefault(c["minimal_pair_group"], set())
        s.add(c["de_form"])
    for gk, forms in by_form_per_group.items():
        if forms == {"contracted", "uncontracted"}:
            groups_with_both.add(gk)

    return {
        "dataset_scope": DATASET_SCOPE,
        "candidate_total": len(candidates),
        "by_form": by_form,
        "by_chapter": {str(k): v for k, v in sorted(by_chapter.items())},
        "by_author_match": {
            "true": by_author_match[True],
            "false": by_author_match[False],
        },
        "minimal_pair_group_count": len(minimal_pair_groups),
        "minimal_pair_groups_with_both_forms": len(groups_with_both),
        "missing_alignment": {
            "en": en_missing,
            "zh": zh_missing,
            "either": sum(
                1
                for c in candidates
                if c["en_alignment_status"] == "missing" or c["zh_alignment_status"] == "missing"
            ),
        },
        "multi_sentence_alignment": {
            "en_1_to_n": en_1_to_n,
            "zh_1_to_n": zh_1_to_n,
            "en_n_to_1": en_n_to_1,
            "zh_n_to_1": zh_n_to_1,
        },
        "pilot": pilot_summary,
        "pilot_selected_total": len(selected) if selected is not None else 0,
    }
