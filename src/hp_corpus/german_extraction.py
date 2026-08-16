"""Paper-faithful German PP extraction with chapter manifests.

This module is the **single implementation** of the paper's
``time-in-translation/conll-extractor`` logic (verbatim algorithm in
:func:`extract_chapter`). The original standalone Ch.1–3 script
``scripts/run_paper_extractor.py`` was replaced by it and is now a
deprecated thin wrapper around ``run_full_novel_german_extraction.py``.
What this module provides over the legacy script:

* **Path bug fix** — input/output paths use ``f"hp1_de_ch{chapter:02d}_..."``
  zero-padding so chapters 10–17 produce the correct filename
  (``hp1_de_ch10_nomwt.conllu``), not the broken ``hp1_de_ch010_...``
  that ``ch0{ch}`` formatting yields for ``ch >= 10`` in the older
  script.
* **Fail-closed I/O** — missing/empty/corrupt inputs raise typed errors
  by default so upstream pipeline failures cannot hide behind a silent
  ``continue``. ``--allow-missing`` is opt-in for the missing case only;
  empty/corrupt inputs are always treated as upstream corruption.
* **Per-chapter manifest** — captures the SHA-256/size/sentence-count of
  each parsed input plus the extraction status and row count, so a
  downstream consumer can prove which input produced which TSV.

The paper's filter lists live in the vendored
``vendor/conll-extractor/conll_extractor/prepositions/data.py`` (gitignored
because the upstream repo carries the paper's annotation set). We import
them once at module import time and expose module-level handles that
fall back to empty containers when the vendor is absent — that lets the
unit tests monkeypatch the handles with small synthetic lists instead of
cloning the vendor. ``extract_chapter`` reads the handles at call time,
so monkeypatching ``german_extraction.CONTRACTED`` (etc.) before invoking
the function is sufficient.

Vendor prerequisite: clone ``time-in-translation/conll-extractor`` into
``vendor/conll-extractor/`` (gitignored) and ensure
``conll_extractor.prepositions.data`` is importable (e.g. by adding the
vendor path to ``PYTHONPATH`` or via ``sys.path.insert`` at the caller
site). Production CLI callers wire this; tests don't need the vendor.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pyconll

from hp_corpus.provenance import (
    DuplicateParseBlockIdError,
    InconsistentProvenanceError,
    MissingProvenanceError,
    scan_blocks,
    split_parse_block_id,
    validate_blocks,
)

# ---------------------------------------------------------------------------
# Vendor filter-list handles.
# ---------------------------------------------------------------------------
# Try the vendored conll-extractor once at module import. If the vendor
# clone is absent (CI, fresh checkout before clone), the handles fall back
# to empty containers so the module still imports. Tests monkeypatch these
# handles with synthetic lists; production runs (with the vendor present)
# get the paper's real annotation set. ``extract_chapter`` reads the
# current values of these module globals at call time, so monkeypatching
# just before invoking the function is sufficient.

CONTRACTED: list[str] = []
PREPOSITIONS: list[str] = []
DETERMINERS: list[str] = []
FILTER_CONTRACTED_123: dict[str, list[str]] = {}
FILTER_PP: dict[str, list[str]] = {}


def _try_load_vendor_lists() -> None:
    """Populate module-level filter-list handles from the vendored clone.

    Best-effort and idempotent. Falls back silently to empty defaults if
    the vendor package isn't importable. We do NOT retry this from inside
    ``extract_chapter`` so that tests can monkeypatch the handles without
    being clobbered.
    """
    global CONTRACTED, PREPOSITIONS, DETERMINERS
    global FILTER_CONTRACTED_123, FILTER_PP

    try:  # pragma: no cover — exercised by integration only
        import importlib

        mod = importlib.import_module("conll_extractor.prepositions.data")
        CONTRACTED = list(getattr(mod, "CONTRACTED", []))
        PREPOSITIONS = list(getattr(mod, "PREPOSITIONS", []))
        DETERMINERS = list(getattr(mod, "DETERMINERS", []))
        FILTER_CONTRACTED_123 = dict(getattr(mod, "FILTER_CONTRACTED_123", {}))
        FILTER_PP = dict(getattr(mod, "FILTER_PP", {}))
    except Exception:
        # Vendor clone missing — keep empty defaults so the module still
        # imports. Tests will monkeypatch the handles directly.
        pass


_try_load_vendor_lists()


# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------

EXTRACTOR_VERSION = "paper-faithful-v1"

POS_ARTICLE = "ART"
POS_NOUN = "NN"

TSV_FIELDS = [
    "parse_block_id",
    "source_segment_id",
    "prep",
    "det",
    "noun",
    "prep_token_id",
    "det_token_id",
    "noun_token_id",
    "pp_token_start",
    "pp_token_end",
    "pp_surface",
    "in_filter",
]

# Exit codes / rule names surfaced by the CLI.
RULE_MISSING = "MISSING_PARSED_INPUT"
RULE_EMPTY = "EMPTY_PARSED_INPUT"
RULE_FAILED = "EXTRACTION_FAILED"
RULE_CHAPTER_RANGE = "CHAPTER_OUT_OF_RANGE"
RULE_PROVENANCE_MISSING = "MISSING_PARSE_PROVENANCE"
RULE_PROVENANCE_DUP = "DUPLICATE_PARSE_BLOCK_ID"
RULE_PROVENANCE_INCONSISTENT = "INCONSISTENT_PARSE_PROVENANCE"

VALID_STATUSES = {
    "ok",
    "zero_hits_ok",
    "missing_input",
    "empty_input",
    "extraction_error",
}


# ---------------------------------------------------------------------------
# Typed errors.
# ---------------------------------------------------------------------------


class GermanExtractionError(Exception):
    """Base class for German-extraction failures surfaced by this module."""


class MissingParsedInputError(GermanExtractionError):
    """Raised when the requested .conllu file does not exist."""


class EmptyParsedInputError(GermanExtractionError):
    """Raised when the .conllu file exists but yields zero sentences."""


class ExtractionFailedError(GermanExtractionError):
    """Raised when pyconll or the extraction algorithm raises."""


class ProvenanceMissingError(GermanExtractionError):
    """Raised when a parsed block lacks block-level provenance.

    A block's ``sent_id`` must be a ``<segment_id>#bNNN`` parse_block_id
    carrying a matching ``# source_segment_id`` comment. A raw segment
    id as sent_id means the provenance migration has not run — refuse
    to extract rather than emit rows whose block identity is ambiguous.
    """


class ProvenanceDuplicateError(GermanExtractionError):
    """Raised when the same parse_block_id appears twice in one file."""


class ProvenanceInconsistentError(GermanExtractionError):
    """Raised when block provenance is present but self-inconsistent:
    sent_id does not extend its ``# source_segment_id``, or the block
    ordinal breaks the within-segment document order (ordinal 0, gaps,
    out-of-order numbering)."""


# ---------------------------------------------------------------------------
# Helpers (mirrors of run_paper_extractor internals).
# ---------------------------------------------------------------------------


def _is_range_id(token_id: str | None) -> bool:
    return bool(token_id) and "-" in token_id


def _reconstruct_surface(
    ordered_ids: list[str], id_to_form: dict[str, str], start: int, end: int
) -> str:
    """Join surface forms of all tokens with id in [start, end] (inclusive)."""
    parts = []
    for tid in ordered_ids:
        if _is_range_id(tid):
            continue
        tid_int = int(tid)
        if start <= tid_int <= end:
            parts.append(id_to_form.get(tid, ""))
    return " ".join(parts)


def _parsed_path_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_sentences(parsed_path: Path) -> int:
    """Count sentences in a CoNLL-U file by counting ``# sent_id`` lines.

    Tolerant: returns 0 on read errors. Used for the manifest only; the
    fail-closed ``EmptyParsedInputError`` is driven by pyconll actually
    yielding zero sentences, not by this count.
    """
    n = 0
    try:
        with open(parsed_path, encoding="utf-8") as f:
            for line in f:
                stripped = line.lstrip()
                if stripped.startswith("# sent_id") or stripped.startswith("#sent_id"):
                    n += 1
    except OSError:
        return 0
    return n


def _provenance_counts(parsed_path: Path) -> dict[str, int]:
    """Lenient provenance aggregates for the manifest (never raises).

    ``blocks_missing_provenance`` > 0 means the file has not been
    migrated (or is inconsistent); :func:`extract_chapter` is the
    fail-closed gate, the manifest just records the state.
    """
    with open(parsed_path, encoding="utf-8") as f:
        blocks = scan_blocks(f)
    migrated = [b for b in blocks if b.migrated and b.source_segment_id]
    distinct = len({b.sent_id for b in migrated})
    per_segment: dict[str, int] = {}
    for b in migrated:
        sid, _ = split_parse_block_id(b.sent_id)
        per_segment[sid] = per_segment.get(sid, 0) + 1
    return {
        "blocks_total": len(blocks),
        "distinct_parse_block_ids": distinct,
        "multi_block_segments": sum(1 for c in per_segment.values() if c > 1),
        "blocks_missing_provenance": len(blocks) - len(migrated),
    }


# ---------------------------------------------------------------------------
# Core extraction API.
# ---------------------------------------------------------------------------


def extract_chapter(parsed_path: Path, contracted: bool) -> list[dict]:
    """Paper-faithful extraction — algorithmically identical to
    ``run_paper_extractor.extract``.

    Parameters
    ----------
    parsed_path
        Path to a normalized (MWT-collapsed) DE CoNLL-U file.
    contracted
        ``True`` → search CONTRACTED forms, no determiner column.
        ``False`` → search PREPOSITIONS + DETERMINERS, fill determiner.

    Returns
    -------
    list of hit dicts (one per PP occurrence). A legitimate zero-hit
    chapter returns an empty list — that is NOT an error.

    Raises
    ------
    MissingParsedInputError
        ``parsed_path`` does not exist.
    EmptyParsedInputError
        ``parsed_path`` exists but ``pyconll`` yields zero sentences.
    ExtractionFailedError
        ``pyconll`` raised, or the extraction algorithm raised.
    """
    if not parsed_path.exists():
        raise MissingParsedInputError(str(parsed_path))

    # Fail-closed block provenance via the single centralized validator
    # (hp_corpus.provenance): missing provenance, duplicate block ids,
    # prefix mismatch, ordinal 0, ordinal gaps and invalid document order
    # all refuse extraction before any row is emitted. sent_id alone is
    # never trusted as a block key.
    try:
        with open(parsed_path, encoding="utf-8") as f:
            validate_blocks(scan_blocks(f))
    except MissingProvenanceError as exc:
        raise ProvenanceMissingError(str(exc)) from exc
    except DuplicateParseBlockIdError as exc:
        raise ProvenanceDuplicateError(str(exc)) from exc
    except InconsistentProvenanceError as exc:
        raise ProvenanceInconsistentError(str(exc)) from exc

    # Read the module-global handles at call time so tests can monkeypatch.
    forms = CONTRACTED if contracted else PREPOSITIONS
    needs_determiner = not contracted

    out: list[dict] = []
    try:
        sentences = pyconll.load_from_file(str(parsed_path))
    except Exception as exc:  # pragma: no cover — pyconll raises generic
        raise ExtractionFailedError(f"pyconll load failed: {exc}") from exc

    sentence_seen = False

    try:
        for sentence in sentences:
            sentence_seen = True

            pid = sentence.id or ""
            source_segment_id = sentence.meta_value("source_segment_id")

            ordered_ids: list[str] = []
            id_to_form: dict[str, str] = {}
            for token in sentence:
                if token.id is None or _is_range_id(token.id):
                    continue
                ordered_ids.append(token.id)
                id_to_form[token.id] = token.form or ""

            current_head: str | None = None
            current_det: str | None = None
            current_token: str | None = None
            current_prep_id: str | None = None
            current_det_id: str | None = None

            for token in sentence:
                if token.form is None or token.id is None or _is_range_id(token.id):
                    continue

                if token.form in forms:
                    current_token = token.form
                    current_head = token.head
                    current_prep_id = token.id

                if needs_determiner and current_head is not None:
                    if (
                        token.head == current_head
                        and token.xpos == POS_ARTICLE
                        and token.form in DETERMINERS
                    ):
                        current_det = token.form
                        current_det_id = token.id

                current_det_filled = current_det is not None or not needs_determiner
                current_head_filled = (
                    current_head is not None
                    and token.id == current_head
                    and token.xpos == POS_NOUN
                )

                if current_head_filled and current_det_filled:
                    pp_start = int(current_prep_id)  # type: ignore[arg-type]
                    pp_end = int(token.id)
                    surface = _reconstruct_surface(
                        ordered_ids, id_to_form, pp_start, pp_end
                    )
                    out.append(
                        {
                            "prep": current_token,
                            "det": current_det,
                            "noun": token.lemma,
                            "parse_block_id": pid,
                            "source_segment_id": source_segment_id,
                            "prep_token_id": current_prep_id,
                            "det_token_id": current_det_id,
                            "noun_token_id": token.id,
                            "pp_token_start": str(pp_start),
                            "pp_token_end": str(pp_end),
                            "pp_surface": surface,
                        }
                    )
                    current_head = None
                    current_det = None
                    current_token = None
                    current_prep_id = None
                    current_det_id = None
    except GermanExtractionError:
        raise
    except Exception as exc:
        raise ExtractionFailedError(f"extraction failed: {exc}") from exc

    if not sentence_seen:
        raise EmptyParsedInputError(str(parsed_path))

    return out


def validate_against_filters(
    hits: list[dict], contracted: bool
) -> tuple[int, int, list[dict]]:
    """Return ``(matched_count, total_count, matched_hits)``.

    QA-only — does NOT gate the manifest or the TSV. ``in_filter`` is
    written as ``"Y"``/``"N"`` per row by :func:`write_chapter_tsv`.
    """
    filt = FILTER_CONTRACTED_123 if contracted else FILTER_PP
    matched = [
        h for h in hits if h["prep"] in filt and h["noun"] in filt[h["prep"]]
    ]
    return len(matched), len(hits), matched


# ---------------------------------------------------------------------------
# Manifest + TSV writers.
# ---------------------------------------------------------------------------


def chapter_manifest(
    *,
    chapter: int,
    contracted: bool,
    parsed_path: Path,
    hits: list[dict] | None,
    status: str,
    error: str | None = None,
) -> dict:
    """Build one manifest entry.

    ``parsed_path_sha256`` / ``parsed_path_size`` /
    ``parsed_path_sentence_count`` are populated when ``parsed_path``
    exists and is readable; otherwise they are ``None``.
    ``extractor_version`` and ``row_count`` are always populated.
    ``error`` is only included when ``status`` indicates an error.
    """
    if status not in VALID_STATUSES:
        raise ValueError(f"unknown status: {status!r}")

    sha: str | None = None
    size: int | None = None
    sent_count: int | None = None
    provenance: dict | None = None

    if parsed_path.exists():
        try:
            sha = _parsed_path_sha256(parsed_path)
            size = parsed_path.stat().st_size
            sent_count = _count_sentences(parsed_path)
            provenance = _provenance_counts(parsed_path)
        except OSError:
            sha = None
            size = None
            sent_count = None
            provenance = None

    row_count = len(hits) if hits is not None else 0

    entry: dict = {
        "chapter": chapter,
        "form": "contracted" if contracted else "uncontracted",
        "parsed_path": str(parsed_path),
        "parsed_path_sha256": sha,
        "parsed_path_size": size,
        "parsed_path_sentence_count": sent_count,
        "parse_blocks": provenance["blocks_total"] if provenance else None,
        "unique_parse_block_ids": provenance["distinct_parse_block_ids"]
        if provenance
        else None,
        "multi_block_segments": provenance["multi_block_segments"]
        if provenance
        else None,
        "blocks_missing_provenance": provenance["blocks_missing_provenance"]
        if provenance
        else None,
        "extractor_version": EXTRACTOR_VERSION,
        "status": status,
        "row_count": row_count,
    }
    if status in {"missing_input", "empty_input", "extraction_error"}:
        entry["error"] = error if error is not None else ""
    return entry


def write_chapter_tsv(path: Path, hits: list[dict], matched: list[dict]) -> Path:
    """Mirror ``run_paper_extractor.write_tsv`` exactly.

    Writes a header-only file for an empty ``hits`` list so downstream
    consumers can rely on the TSV always existing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    matched_keys = {
        (h["parse_block_id"], h["prep"], h["noun"], h["pp_token_start"], h["pp_token_end"])
        for h in matched
    }
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for h in hits:
            key = (
                h["parse_block_id"],
                h["prep"],
                h["noun"],
                h["pp_token_start"],
                h["pp_token_end"],
            )
            row = {**h, "in_filter": "Y" if key in matched_keys else "N"}
            if row.get("det") is None:
                row["det"] = "-"
            if row.get("det_token_id") is None:
                row["det_token_id"] = "-"
            row = {k: v for k, v in row.items() if k in TSV_FIELDS}
            w.writerow(row)
    return path


def write_manifest_json(path: Path, manifest: list[dict]) -> Path:
    """Write per-chapter manifest entries as a JSON array."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


__all__ = [
    "EXTRACTOR_VERSION",
    "TSV_FIELDS",
    "CONTRACTED",
    "PREPOSITIONS",
    "DETERMINERS",
    "FILTER_CONTRACTED_123",
    "FILTER_PP",
    "GermanExtractionError",
    "MissingParsedInputError",
    "EmptyParsedInputError",
    "ExtractionFailedError",
    "ProvenanceMissingError",
    "ProvenanceDuplicateError",
    "ProvenanceInconsistentError",
    "RULE_PROVENANCE_INCONSISTENT",
    "extract_chapter",
    "validate_against_filters",
    "chapter_manifest",
    "write_chapter_tsv",
    "write_manifest_json",
]
