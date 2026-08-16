"""Block-level parse provenance: ``parse_block_id`` + ``source_segment_id``.

ID construction rule
--------------------

``parse_block_id = f"{source_segment_id}#b{ordinal:03d}"``

``ordinal`` is the 1-based position of the CoNLL-U block among all
blocks parsed from its source Segment, in document order — the order
the parser emitted them, which equals file order. A Segment the parser
splits into ``k`` blocks yields ``…#b001`` … ``…#b{k:03d}``. A Segment
parsed as one block yields ``…#b001``.

Determinism: identical input text + identical parser/model emits an
identical block sequence, and the ordinal depends only on that
sequence, so re-parsing the same segments reproduces the same IDs. A
pure text migration (:func:`migrate_conllu_text`) of an existing file
assigns exactly the same IDs because it numbers blocks of one segment
in file order — the order the parser wrote them.

``sent_id`` is never a standalone block key: after migration the CoNLL-U
``# sent_id`` IS the parse_block_id (unique per file), and the original
segment identifier is preserved separately as
``# source_segment_id = …``. Downstream consumers (extraction TSVs,
annotation master) carry both columns and fail closed when either is
missing or inconsistent.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

SENT_ID_PREFIX = "# sent_id = "
SOURCE_SEGMENT_KEY = "source_segment_id"

BLOCK_MARKER = "#b"
_ORDINAL_RE = re.compile(
    r"^(?P<sid>.+)" + re.escape(BLOCK_MARKER) + r"(?P<ord>\d{3,})$"
)


class ProvenanceError(Exception):
    """Base class for provenance integrity failures (fail closed)."""


class MalformedParseBlockIdError(ProvenanceError):
    """A sent_id is neither a raw segment id nor a well-formed parse_block_id."""


class MissingProvenanceError(ProvenanceError):
    """A block lacks a parse_block_id or its ``# source_segment_id`` line."""


class DuplicateParseBlockIdError(ProvenanceError):
    """The same parse_block_id occurs twice in one file."""


class InconsistentProvenanceError(ProvenanceError):
    """sent_id and ``# source_segment_id`` disagree, or the ordinal breaks
    the within-segment document order."""


def make_parse_block_id(source_segment_id: str, ordinal: int) -> str:
    """Build the block-level id for the ``ordinal``-th block of a segment."""
    if ordinal < 1:
        raise ValueError(f"ordinal must be >= 1, got {ordinal}")
    return f"{source_segment_id}{BLOCK_MARKER}{ordinal:03d}"


def split_parse_block_id(parse_block_id: str) -> tuple[str, int]:
    """Split a parse_block_id into ``(source_segment_id, ordinal)``.

    Raises :class:`MalformedParseBlockIdError` if the id is not of the
    form ``<segment_id>#bNNN``.
    """
    m = _ORDINAL_RE.match(parse_block_id)
    if not m or not m.group("sid"):
        raise MalformedParseBlockIdError(
            f"not a parse_block_id (expected '<segment_id>{BLOCK_MARKER}NNN'): "
            f"{parse_block_id!r}"
        )
    return m.group("sid"), int(m.group("ord"))


def is_parse_block_id(sent_id: str) -> bool:
    """True iff ``sent_id`` carries the block marker with a valid ordinal."""
    try:
        split_parse_block_id(sent_id)
    except MalformedParseBlockIdError:
        return False
    return True


@dataclass
class BlockProvenance:
    """Provenance of one CoNLL-U block, as read from comment lines.

    ``orphan_content=True`` marks a physical block whose token content
    was reached without any ``# sent_id`` comment — ``sent_id`` is the
    empty string for such records. They exist so that content-bearing
    blocks cannot vanish from validation just because no record would
    otherwise be created for them.
    """

    sent_id: str
    source_segment_id: str | None = None
    line_index: int = -1  # 0-based line of the ``# sent_id`` line
    migrated: bool = False  # sent_id already is a parse_block_id
    orphan_content: bool = False  # token content reached with no # sent_id


@dataclass
class MigrationStats:
    """Aggregate counts produced by a scan or migration (no text, no ids)."""

    blocks_total: int = 0
    distinct_sent_ids_before: int = 0
    duplicated_sent_ids_before: int = 0
    blocks_migrated: int = 0
    blocks_already_migrated: int = 0
    segments_total: int = 0
    multi_block_segments: int = 0
    duplicate_parse_block_ids: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "blocks_total": self.blocks_total,
            "distinct_sent_ids_before": self.distinct_sent_ids_before,
            "duplicated_sent_ids_before": self.duplicated_sent_ids_before,
            "blocks_migrated": self.blocks_migrated,
            "blocks_already_migrated": self.blocks_already_migrated,
            "segments_total": self.segments_total,
            "multi_block_segments": self.multi_block_segments,
            "duplicate_parse_block_ids": self.duplicate_parse_block_ids,
            "errors": list(self.errors),
        }


def scan_blocks(lines: Iterable[str]) -> list[BlockProvenance]:
    """Read block provenance from CoNLL-U lines (lightweight, no pyconll).

    A normal block starts at its ``# sent_id`` comment; order of
    appearance is preserved. Blank lines are physical-block boundaries:
    a physical block whose token content is reached without any
    ``# sent_id`` comment is recorded as an orphan
    (``orphan_content=True``, empty ``sent_id``) — one record per
    physical block — so it cannot silently disappear from validation.
    Comments separated from their ``# sent_id`` line by a blank line
    belong to the next physical block and do not attach to the previous
    one. Does not validate; callers validate with
    :func:`validate_blocks` or migrate with :func:`migrate_conllu_text`.
    """
    blocks: list[BlockProvenance] = []
    cur: BlockProvenance | None = None  # header-open block (sent_id seen)
    saw_sent_id = False  # current physical block carries a # sent_id
    orphan: BlockProvenance | None = None
    for i, raw in enumerate(lines):
        line = raw.rstrip("\n")
        if not line.strip():
            cur = None
            saw_sent_id = False
            orphan = None
        elif line.startswith(SENT_ID_PREFIX):
            cur = BlockProvenance(
                sent_id=line[len(SENT_ID_PREFIX) :].strip(), line_index=i
            )
            blocks.append(cur)
            saw_sent_id = True
            orphan = None
        elif line.startswith("#"):
            if cur is not None:
                key, _, value = line.lstrip("# ").partition("=")
                if key.strip() == SOURCE_SEGMENT_KEY and cur.source_segment_id is None:
                    cur.source_segment_id = value.strip()
        else:
            # Token/content line: ends the header of the current block.
            cur = None
            if not saw_sent_id and orphan is None:
                orphan = BlockProvenance(
                    sent_id="", line_index=i, orphan_content=True
                )
                blocks.append(orphan)
    for b in blocks:
        b.migrated = is_parse_block_id(b.sent_id)
    return blocks


def validate_blocks(blocks: list[BlockProvenance]) -> None:
    """Fail-closed validation of already-migrated provenance.

    Raises :class:`MissingProvenanceError` (token content with no
    ``# sent_id``, no parse_block_id, or no ``# source_segment_id``),
    :class:`DuplicateParseBlockIdError` (repeated parse_block_id), or
    :class:`InconsistentProvenanceError` (sent_id/source_segment_id
    mismatch, or an ordinal that breaks the 1..k document order within
    its segment).
    """
    seen: set[str] = set()
    expected: Counter[str] = Counter()
    for b in blocks:
        if b.orphan_content:
            raise MissingProvenanceError(
                f"physical block with token content but no # sent_id "
                f"(first content line {b.line_index}) — block identity "
                "cannot be established"
            )
        if not b.migrated:
            raise MissingProvenanceError(
                f"sent_id {b.sent_id!r} (line {b.line_index}) is not a "
                "parse_block_id; run the provenance migration first"
            )
        if b.source_segment_id is None:
            raise MissingProvenanceError(
                f"block {b.sent_id!r} (line {b.line_index}) has no "
                f"# {SOURCE_SEGMENT_KEY} comment"
            )
        sid, ordinal = split_parse_block_id(b.sent_id)
        if sid != b.source_segment_id:
            raise InconsistentProvenanceError(
                f"sent_id {b.sent_id!r} does not extend its "
                f"{SOURCE_SEGMENT_KEY} {b.source_segment_id!r}"
            )
        if b.sent_id in seen:
            raise DuplicateParseBlockIdError(
                f"duplicate parse_block_id {b.sent_id!r}"
            )
        seen.add(b.sent_id)
        expected[sid] += 1
        if ordinal != expected[sid]:
            raise InconsistentProvenanceError(
                f"parse_block_id {b.sent_id!r} has ordinal {ordinal}, but it "
                f"is block {expected[sid]} of segment {sid!r} in document order"
            )


def validate_conllu_text(text: str) -> None:
    """Validate the provenance comments of a whole CoNLL-U file."""
    validate_blocks(scan_blocks(text.splitlines()))


def _before_stats(blocks: list[BlockProvenance]) -> tuple[int, int]:
    counts = Counter(b.sent_id for b in blocks)
    distinct = len(counts)
    duplicated = sum(1 for c in counts.values() if c > 1)
    return distinct, duplicated


def migrate_conllu_text(text: str) -> tuple[str, MigrationStats]:
    """Deterministically add block-level provenance to a CoNLL-U file.

    Pre-migration input: ``# sent_id = <segment_id>`` shared by every
    block parsed from one segment. Output: ``# sent_id =
    <segment_id>#bNNN`` plus a ``# source_segment_id = <segment_id>``
    line inserted immediately after. Idempotent: blocks that already
    carry valid provenance pass through unchanged (and are validated).

    Returns ``(new_text, stats)``. Raises a :class:`ProvenanceError`
    subclass on malformed input (including physical blocks whose token
    content carries no ``# sent_id`` — their source segment is
    unknowable, so they cannot be assigned block ids); the input text
    is never partially rewritten.
    """
    lines = text.splitlines(keepends=True)
    blocks = scan_blocks(lines)
    stats = MigrationStats(blocks_total=len(blocks))
    orphans = [b for b in blocks if b.orphan_content]
    if orphans:
        raise MissingProvenanceError(
            f"{len(orphans)} physical block(s) with token content but no "
            f"# sent_id (first at line {orphans[0].line_index}); cannot "
            "migrate — the source segment is unknowable"
        )
    unmigrated = [b for b in blocks if not b.migrated]
    stats.distinct_sent_ids_before, stats.duplicated_sent_ids_before = _before_stats(
        unmigrated
    )
    if all(b.migrated for b in blocks) and blocks:
        validate_blocks(blocks)
        stats.blocks_already_migrated = len(blocks)
        stats.segments_total = len({split_parse_block_id(b.sent_id)[0] for b in blocks})
        stats.multi_block_segments = sum(
            1 for c in Counter(split_parse_block_id(b.sent_id)[0] for b in blocks).values() if c > 1
        )
        return text, stats

    if any(b.migrated for b in blocks):
        raise InconsistentProvenanceError(
            "file mixes migrated and unmigrated blocks; refusing to migrate"
        )

    ordinal: Counter[str] = Counter()
    out: list[str] = []
    block_iter = iter(blocks)
    cur = next(block_iter, None)
    seen_ids: set[str] = set()
    for i, line in enumerate(lines):
        if cur is not None and i == cur.line_index:
            ordinal[cur.sent_id] += 1
            pid = make_parse_block_id(cur.sent_id, ordinal[cur.sent_id])
            if pid in seen_ids:
                raise DuplicateParseBlockIdError(f"duplicate {pid!r} after migration")
            seen_ids.add(pid)
            newline = "\n" if line.endswith("\n") else ""
            out.append(f"{SENT_ID_PREFIX}{pid}{newline}")
            out.append(f"# {SOURCE_SEGMENT_KEY} = {cur.sent_id}{newline}")
            stats.blocks_migrated += 1
            cur = next(block_iter, None)
        else:
            out.append(line)
    stats.segments_total = len(ordinal)
    stats.multi_block_segments = sum(1 for c in ordinal.values() if c > 1)
    return "".join(out), stats


def provenance_counts(blocks: list[BlockProvenance]) -> dict[str, int]:
    """Aggregate provenance counts for a (migrated) block list."""
    validate_blocks(blocks)
    sids = [split_parse_block_id(b.sent_id)[0] for b in blocks]
    per_segment = Counter(sids)
    return {
        "blocks_total": len(blocks),
        "distinct_parse_block_ids": len({b.sent_id for b in blocks}),
        "distinct_source_segments": len(per_segment),
        "multi_block_segments": sum(1 for c in per_segment.values() if c > 1),
        "max_blocks_per_segment": max(per_segment.values(), default=0),
    }


__all__ = [
    "BLOCK_MARKER",
    "SENT_ID_PREFIX",
    "SOURCE_SEGMENT_KEY",
    "BlockProvenance",
    "MigrationStats",
    "ProvenanceError",
    "MalformedParseBlockIdError",
    "MissingProvenanceError",
    "DuplicateParseBlockIdError",
    "InconsistentProvenanceError",
    "make_parse_block_id",
    "split_parse_block_id",
    "is_parse_block_id",
    "scan_blocks",
    "validate_blocks",
    "validate_conllu_text",
    "migrate_conllu_text",
    "provenance_counts",
]
