"""Reusable pure functions for building annotation context packs.

Generalizes the pilot-only logic that used to live inline in
``scripts/build_alignment_review_pack.py``. Given any annotation-target
TSV whose rows specify a German PP occurrence plus the aligned EN and ZH
sentence IDs, this module produces an enlarged TSV row with one
preceding + following context sentence per language (DE/EN/ZH), so a
human annotator can judge each candidate with surrounding prose visible.

Design points:

- Pure functions only. The CLI wrapper in
  ``scripts/build_annotation_context_pack.py`` is responsible for I/O,
  argument parsing, stdout discipline, and exit codes.
- The module knows nothing about Harry Potter or about German
  prepositions; it operates on (segment_id, lang, chapter, text) tuples
  loaded from the segmented JSONL files.
- Failures surface as :class:`ContextPackError` subclasses, each
  carrying a ``rule`` string (for stderr aggregate logging) and a row
  index (when applicable). The exception messages never include source
  novel text — only structural metadata.
- Stdout discipline is the caller's responsibility. The functions here
  never print; they return strings or raise.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from hp_corpus.provenance import MalformedParseBlockIdError, split_parse_block_id
from hp_corpus.schema import Segment

# --------------------------------------------------------------------- constants

# Input TSV must carry every column in this tuple. The columns are the
# minimum needed to look up DE/EN/ZH sentences, write the output row, and
# preserve the existing source_row_sha256 (the builder never recomputes
# the hash; it only forwards what the upstream pipeline wrote).
REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "datapoint_id",
    "chapter",
    "de_parse_block_id",
    "de_source_segment_id",
    "de_form",
    "de_token_start",
    "de_token_end",
    "de_sentence_text",
    "en_sentence_ids",
    "en_aligned_text",
    "zh_sentence_ids",
    "zh_aligned_text",
    "en_alignment_cardinality",
    "en_alignment_confidence",
    "zh_alignment_cardinality",
    "zh_alignment_confidence",
    "source_row_sha256",
)

# Output TSV columns — same shape as the existing pilot review pack.
# The last six columns are blank human-check columns the annotator fills
# (scene_match, counterpart_locatable, alignment_issue, review_notes
# plus one pair per target language).
OUTPUT_COLUMNS: tuple[str, ...] = (
    "datapoint_id",
    "chapter",
    "de_form",
    "de_parse_block_id",
    "de_source_segment_id",
    "de_token_range",
    "de_text",
    "de_context_prev",
    "de_context_next",
    "en_sentence_ids",
    "en_text",
    "en_cardinality",
    "en_confidence",
    "en_context_prev",
    "en_context_next",
    "zh_sentence_ids",
    "zh_text",
    "zh_cardinality",
    "zh_confidence",
    "zh_context_prev",
    "zh_context_next",
    # ---- blank human-check columns (annotator fills) ----
    "en_scene_match",
    "zh_scene_match",
    "en_counterpart_locatable",
    "zh_counterpart_locatable",
    "alignment_issue",
    "review_notes",
)

# Number of trailing blank human-check columns in OUTPUT_COLUMNS. Kept as
# a constant so the CLI's stdout summary line stays in sync with the
# column tuple.
N_BLANK_HUMAN_CHECK_COLUMNS = 6

# Matches the canonical segment-id pattern used by the pipeline. The
# language capture group accepts the 2-3 letter codes the schema allows
# (currently "de", "en", "zh" — three 2-letter codes — but the pattern
# also tolerates the schema's broader ``[a-z]+`` rule for forward compat).
SEGMENT_ID_RE = re.compile(r"^[a-z0-9]+_([a-z]{2,3})_ch\d{2}_p\d{4}_s\d{3}$")


# --------------------------------------------------------------------- exceptions


class ContextPackError(Exception):
    """Base class for context-pack builder errors.

    Each subclass carries:

    - ``rule``: a short stable string the CLI emits on stderr so test
      assertions can match against it without depending on message wording.
    - ``row``: 0-indexed TSV body row that triggered the error, or
      ``None`` for errors raised before any row is processed (e.g.
      missing required columns).
    - ``detail``: optional non-text structural metadata (e.g. which
      chapter / lang an unresolved segment belongs to). **Never** source
      text or segment IDs.
    """

    rule = "CONTEXT_PACK_ERROR"

    def __init__(self, *, row: int | None = None, detail: str = "") -> None:
        self.row = row
        self.detail = detail
        super().__init__(f"rule={self.rule} row={row} detail={detail!r}")


class MissingColumnError(ContextPackError):
    """One or more required input columns are absent from the header."""

    rule = "MISSING_REQUIRED_COLUMN"

    def __init__(self, column: str) -> None:
        self.column = column
        super().__init__(row=None, detail=f"column={column}")


class MalformedIdError(ContextPackError):
    """A ``de_sentence_id`` does not match the canonical segment-id pattern."""

    rule = "MALFORMED_DE_SENTENCE_ID"


class MalformedBlockIdError(ContextPackError):
    """A ``de_parse_block_id`` is not of the form ``<segment_id>#bNNN``."""

    rule = "MALFORMED_DE_PARSE_BLOCK_ID"


class ProvenanceMismatchError(ContextPackError):
    """``de_parse_block_id`` does not extend its ``de_source_segment_id``."""

    rule = "DE_PROVENANCE_MISMATCH"


class UnresolvedSegmentError(ContextPackError):
    """A segment ID listed in the row (DE, EN, or ZH) is not present in
    that chapter's segmented JSONL. The error message must NOT include
    the segment ID itself; only the row index and (optionally) the
    language + chapter."""

    rule = "UNRESOLVED_SEGMENT_ID"

    def __init__(self, *, row: int | None, lang: str = "", chapter: int | None = None) -> None:
        d = f"lang={lang}" if lang else ""
        if chapter is not None:
            d = f"{d} chapter={chapter}".strip()
        super().__init__(row=row, detail=d)


class MalformedJsonError(ContextPackError):
    """``en_sentence_ids`` or ``zh_sentence_ids`` is not valid JSON."""

    rule = "MALFORMED_SENTENCE_IDS_JSON"

    def __init__(self, *, row: int | None, field: str) -> None:
        self.field = field
        super().__init__(row=row, detail=f"field={field}")


# --------------------------------------------------------------------- helpers


def load_segments_chapter(lang: str, chapter: int, segmented_dir: Path) -> list[Segment]:
    """Load one (lang, chapter) segmented JSONL file as a list of Segments.

    Returns an empty list if the file does not exist — the caller (CLI)
    is responsible for raising :class:`UnresolvedSegmentError` if a row
    references IDs that should have lived in that file. This keeps the
    loader total (no per-row file-existence branching at call sites) and
    matches the existing pilot script's behaviour.
    """
    path = segmented_dir / f"hp1_{lang}_ch{chapter:02d}.jsonl"
    if not path.exists():
        return []
    out: list[Segment] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(Segment.model_validate_json(line))
    return out


def build_index(segs: list[Segment]) -> dict[str, int]:
    """Map each segment id → its file-order position."""
    return {s.id: i for i, s in enumerate(segs)}


def join_text(segs: list[Segment], ids: list[str]) -> str:
    """Concatenate text for ``ids`` in the given order, separated by
    ``" / "`` so multi-sentence alignments stay readable inside one TSV
    cell. Unknown ids are skipped (callers validate via
    :func:`context_around` which raises on unresolved ids).
    """
    if not ids:
        return ""
    by_id = {s.id: s for s in segs}
    parts = [by_id[i].text for i in ids if i in by_id]
    return " / ".join(parts)


def context_around(
    segs: list[Segment], ids: list[str], n: int
) -> tuple[str, str, str]:
    """Return ``(prev_text, curr_text, next_text)`` for the given ids.

    - ``prev_text`` = the ``n`` file-order sentences immediately before
      the earliest id in ``ids``.
    - ``curr_text`` = the ``" / "``-joined text of ``ids`` themselves
      (no surrounding markers; the CLI adds ``«»`` for DE only).
    - ``next_text`` = the ``n`` file-order sentences immediately after
      the latest id in ``ids``.

    Chapter boundary is the file boundary: this function never crosses
    into another (lang, chapter) file. Returns ``("", "", "")`` when
    ``ids`` is empty or none of the ids resolve.
    """
    if not ids or not segs:
        return "", "", ""
    id_to_pos = build_index(segs)
    positions = [id_to_pos[i] for i in ids if i in id_to_pos]
    if not positions:
        return "", "", ""
    lo = min(positions)
    hi = max(positions)
    prev_ids = [segs[i].id for i in range(max(0, lo - n), lo)]
    next_ids = [segs[i].id for i in range(hi + 1, min(len(segs), hi + 1 + n))]
    prev_text = join_text(segs, prev_ids)
    next_text = join_text(segs, next_ids)
    curr_text = join_text(segs, ids)
    return prev_text, curr_text, next_text


def parse_sentence_ids(
    raw: str, *, row: int | None, field: str
) -> list[str]:
    """Decode a TSV cell that should hold a JSON list of segment ids.

    Empty cells decode to ``[]`` (a row with no EN/ZH alignment is
    structurally valid — alignment_cardinality will read ``0:N`` or
    ``N:0``). Anything non-empty that fails JSON parsing raises
    :class:`MalformedJsonError`; anything that JSON-parses to a non-list
    also raises it.
    """
    if raw is None or raw == "":
        return []
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as e:
        raise MalformedJsonError(row=row, field=field) from e
    if not isinstance(decoded, list):
        raise MalformedJsonError(row=row, field=field)
    return [str(x) for x in decoded]


def assert_de_sentence_id_wellformed(
    de_sentence_id: str, *, row: int | None
) -> None:
    """Raise :class:`MalformedIdError` if the id is not canonical."""
    if not SEGMENT_ID_RE.match(de_sentence_id):
        raise MalformedIdError(row=row, detail="")


def assert_de_parse_block_provenance(
    de_parse_block_id: str, de_source_segment_id: str, *, row: int | None
) -> None:
    """Fail closed on malformed or inconsistent block-level provenance.

    Reuses the centralized parse-block-id grammar
    (:func:`hp_corpus.provenance.split_parse_block_id`): the block id
    must be a well-formed ``<segment_id>#bNNN`` whose segment part
    equals the row's ``de_source_segment_id``. Checking that both cells
    are merely non-empty is not enough — a block id pointing at a
    different segment would silently mis-join the row.
    """
    try:
        sid, _ = split_parse_block_id(de_parse_block_id)
    except MalformedParseBlockIdError as e:
        raise MalformedBlockIdError(row=row) from e
    if sid != de_source_segment_id:
        raise ProvenanceMismatchError(row=row)


def assert_ids_resolved(
    ids: list[str],
    segs: list[Segment],
    *,
    row: int | None,
    lang: str,
    chapter: int | None,
) -> None:
    """Raise :class:`UnresolvedSegmentError` if any id in ``ids`` is
    absent from ``segs``. The error never carries the offending id.
    """
    if not ids:
        return
    known = {s.id for s in segs}
    for sid in ids:
        if sid not in known:
            raise UnresolvedSegmentError(row=row, lang=lang, chapter=chapter)
