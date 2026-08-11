"""Deterministic multi-annotator batching for Step 4 master annotation TSVs.

Takes a validated master TSV (every row has a ``datapoint_id`` and a
``de_candidate_decision``) and produces per-annotator batches with a
shared calibration set and configurable overlap. Assignment is fully
deterministic from ``datapoint_id + seed`` — re-running with the same
inputs produces byte-identical output.

Routing rules
-------------

For each master row, route by ``de_candidate_decision``:

  * ``include``            → eligible for annotation batches.
  * ``exclude``            → routed to ``<out-dir>/excluded.tsv`` (or
                             dropped entirely with ``--drop-excluded``).
  * blank or ``uncertain`` → routed to ``<out-dir>/blocked_review.tsv``.

Deterministic assignment
------------------------

For each eligible (include) row, compute::

    h = sha256(f"{datapoint_id}|{seed}".encode("utf-8")).hexdigest()
    stable_key = int(h[:16], 16)

Sort all eligible rows by ``(stable_key, chapter, de_form, datapoint_id)``.
The first ``calibration_size`` rows form the shared calibration batch
(assigned to every annotator). The remaining rows are partitioned:

  * A's main batch = rows whose ``stable_key % n_annotators == A_index``.
  * To achieve overlap, A's batch also includes up to
    ``overlap_rate * len(A_main_batch)`` rows from
    ``(A_index - 1) % n_annotators`` (chosen by stable order).

Privacy
-------

Stdout carries only aggregate counts and the output directory path.
Never ``datapoint_id`` values, annotator-editable values, source text,
or hashes.

Usage::

    uv run python scripts/prepare_annotation_batches.py \\
        --master-tsv data/derived/step4/master.tsv \\
        --out-dir   data/derived/step4/batches/run01 \\
        --annotators alice bob
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from hp_corpus.step4 import ALL_TSV_COLUMNS

# Routing buckets.
_INCLUDE = "include"
_EXCLUDE = "exclude"
_UNCERTAIN = "uncertain"

# Default seed — versioned so seed rotation forces a fresh assignment.
DEFAULT_SEED = "hp-transmining-bremmers-v1"
DEFAULT_OVERLAP = 0.2
DEFAULT_CALIBRATION = 10


# --------------------------------------------------------------------- errors


class BatchError(Exception):
    """A user-facing error. ``rule`` is the stable machine-readable code;
    ``message`` is the human-readable detail."""

    def __init__(self, rule: str, message: str):
        super().__init__(message)
        self.rule = rule
        self.message = message


# --------------------------------------------------------------------- hashing


def _stable_key(datapoint_id: str, seed: str) -> int:
    """Hash (datapoint_id, seed) deterministically to a 64-bit int.

    Uses UTF-8 (the default for :meth:`str.encode`) so non-ASCII
    datapoint IDs hash identically across runs.
    """
    blob = f"{datapoint_id}|{seed}".encode()
    h = hashlib.sha256(blob).hexdigest()
    return int(h[:16], 16)


# --------------------------------------------------------------------- routing


def _route(row: dict[str, str]) -> str:
    """Return one of: 'include', 'exclude', 'blocked'.

    Blank or unknown decisions route to 'blocked' (the only safe default
    — silently treating an unknown as include would leak unvetted rows
    into annotator batches; treating it as exclude would hide data).
    """
    decision = row.get("de_candidate_decision", "").strip()
    if decision == _INCLUDE:
        return "include"
    if decision == _EXCLUDE:
        return "exclude"
    # blank, "uncertain", or anything else → blocked review.
    return "blocked"


# --------------------------------------------------------------------- types


class _Eligible:
    """An eligible row with its deterministic sort key precomputed."""

    __slots__ = ("row", "stable_key", "datapoint_id", "chapter", "de_form")

    def __init__(self, row: dict[str, str], seed: str):
        self.row = row
        self.datapoint_id = row["datapoint_id"]
        self.stable_key = _stable_key(self.datapoint_id, seed)
        # Chapter is an int in the schema; defensive parse for safety.
        try:
            self.chapter = int(row.get("chapter", "0"))
        except ValueError:
            self.chapter = 0
        self.de_form = row.get("de_form", "")

    def sort_key(self) -> tuple[Any, ...]:
        return (self.stable_key, self.chapter, self.de_form, self.datapoint_id)


# --------------------------------------------------------------------- assignment


def _assign(
    eligible: list[_Eligible],
    *,
    annotators: list[str],
    calibration_size: int,
    overlap_rate: float,
    seed: str,
) -> dict[str, Any]:
    """Compute the full assignment.

    Returns a dict with keys: ``calibration``, ``per_annotator`` (each
    entry has ``main_batch``, ``overlap``, ``calibration`` lists of
    datapoint IDs in stable order).
    """
    n = len(annotators)
    annotator_index = {name: i for i, name in enumerate(annotators)}

    # Stable-sort once. The slice that becomes calibration is the first
    # ``calibration_size`` rows in this order; the rest is partitioned.
    eligible.sort(key=lambda e: e.sort_key())

    calibration_entries = eligible[:calibration_size]
    remaining = eligible[calibration_size:]

    calibration_ids = [e.datapoint_id for e in calibration_entries]

    # Index the remaining rows by their primary annotator for O(1) lookup
    # while building overlap slices.
    by_primary: dict[int, list[_Eligible]] = {i: [] for i in range(n)}
    for e in remaining:
        primary = e.stable_key % n
        by_primary[primary].append(e)

    per_annotator: dict[str, dict[str, list[str]]] = {}
    for name in annotators:
        i = annotator_index[name]
        # Main batch — stable order (already sorted by sort_key above).
        main_entries = by_primary[i]
        main_ids = [e.datapoint_id for e in main_entries]

        # Overlap slice — rows whose primary annotator is the previous
        # annotator (i - 1 mod n). Sized to ``overlap_rate`` * main size,
        # rounded down so we never exceed the donor's main batch.
        donor_idx = (i - 1) % n
        donor_entries = by_primary[donor_idx]
        target_overlap_count = int(overlap_rate * len(main_ids))
        # Slice the donor's stable-sorted list from the front; the same
        # rows appear in every reproducible run.
        overlap_entries = donor_entries[:target_overlap_count]
        overlap_ids = [e.datapoint_id for e in overlap_entries]

        per_annotator[name] = {
            "main_batch_ids": main_ids,
            "overlap_ids": overlap_ids,
            "calibration_ids": list(calibration_ids),
        }

    return {
        "calibration_ids": calibration_ids,
        "per_annotator": per_annotator,
    }


# --------------------------------------------------------------------- writing


def _write_tsv(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    """Write rows to a TSV with the canonical header. Lineterminator is
    forced to ``\\n`` so byte-comparisons across platforms are stable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=header,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        w.writeheader()
        for r in rows:
            w.writerow({col: r.get(col, "") for col in header})


def _write_annotator_file(
    path: Path,
    *,
    calibration_rows: list[dict[str, str]],
    main_rows: list[dict[str, str]],
    overlap_rows: list[dict[str, str]],
) -> None:
    """Write one annotator's TSV: calibration + main + overlap, in that
    order. Calibration first gives every annotator the same warm-up
    context; dedup happens upstream in :func:`_build_annotator_rows`."""
    rows = list(calibration_rows) + list(main_rows) + list(overlap_rows)
    _write_tsv(path, list(ALL_TSV_COLUMNS), rows)


def _build_annotator_rows(
    rows_by_id: dict[str, dict[str, str]],
    assignment: dict[str, Any],
    annotator: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Materialize calibration / main / overlap row lists for one annotator.

    Dedup is enforced: if a row would appear in both main and overlap
    (only possible when ``overlap_rate`` is large enough that an
    annotator's main batch overlaps with its donor's main batch through
    some quirk), it stays in main only.
    """
    per = assignment["per_annotator"][annotator]
    calibration_ids = list(per["calibration_ids"])
    main_ids = list(per["main_batch_ids"])
    overlap_ids = list(per["overlap_ids"])

    calibration_rows = [rows_by_id[i] for i in calibration_ids]
    main_rows = [rows_by_id[i] for i in main_ids]

    # Dedup overlap against main AND calibration. The invariant "no row
    # appears twice in the same annotator's file" must hold.
    seen = set(calibration_ids) | set(main_ids)
    overlap_rows: list[dict[str, str]] = []
    for dp_id in overlap_ids:
        if dp_id in seen:
            continue
        overlap_rows.append(rows_by_id[dp_id])
        seen.add(dp_id)

    return calibration_rows, main_rows, overlap_rows


# --------------------------------------------------------------------- driver


def _validate_args(
    *,
    annotators: list[str],
    overlap: float,
    calibration_size: int,
    out_dir: Path,
    force_output: bool,
) -> None:
    """Validate CLI args. Raises :class:`BatchError` on failure."""
    if len(annotators) < 2:
        raise BatchError(
            "TOO_FEW_ANNOTATORS",
            f"need ≥2 unique annotators, got {len(annotators)}",
        )
    if len(set(annotators)) != len(annotators):
        raise BatchError(
            "DUPLICATE_ANNOTATORS",
            "annotator names must be unique",
        )
    if not (0.0 <= overlap < 1.0):
        raise BatchError(
            "BAD_OVERLAP",
            f"--overlap must be in [0.0, 1.0), got {overlap}",
        )
    if calibration_size < 0:
        raise BatchError(
            "BAD_CALIBRATION_SIZE",
            f"--calibration-size must be ≥0, got {calibration_size}",
        )

    if out_dir.exists() and any(out_dir.iterdir()) and not force_output:
        raise BatchError(
            "OUT_DIR_NOT_EMPTY",
            f"{out_dir} exists and is non-empty (pass --force-output to overwrite)",
        )


def _read_master(path: Path) -> list[dict[str, str]]:
    """Read the master TSV. Returns the body rows (header already stripped).

    Raises :class:`BatchError` if the file is missing, empty, or has the
    wrong header.
    """
    if not path.exists():
        raise BatchError("MASTER_NOT_FOUND", f"master TSV not found: {path}")
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        rows_raw = list(reader)
    if not rows_raw:
        raise BatchError("MASTER_EMPTY", f"master TSV is empty: {path}")
    header = rows_raw[0]
    expected = list(ALL_TSV_COLUMNS)
    if header != expected:
        raise BatchError(
            "MASTER_HEADER_MISMATCH",
            f"master TSV header has {len(header)} columns; expected {len(expected)} "
            f"columns in fixed ALL_TSV_COLUMNS order",
        )
    body = [dict(zip(header, row, strict=False)) for row in rows_raw[1:] if any(row)]
    return body


def prepare_batches(
    *,
    master_tsv: Path,
    out_dir: Path,
    annotators: list[str],
    seed: str,
    overlap: float,
    calibration_size: int,
    drop_excluded: bool,
    force_output: bool,
) -> dict[str, Any]:
    """Pure driver — does all the work, returns the manifest dict (the
    same dict written to ``batch_manifest.json``).

    Raises :class:`BatchError` on any user-facing failure. The CLI
    wrapper converts these to stderr messages + exit code 2.
    """
    _validate_args(
        annotators=annotators,
        overlap=overlap,
        calibration_size=calibration_size,
        out_dir=out_dir,
        force_output=force_output,
    )

    body = _read_master(master_tsv)

    # Route every row.
    include_rows: list[dict[str, str]] = []
    exclude_rows: list[dict[str, str]] = []
    blocked_rows: list[dict[str, str]] = []
    for r in body:
        route = _route(r)
        if route == "include":
            include_rows.append(r)
        elif route == "exclude":
            exclude_rows.append(r)
        else:
            blocked_rows.append(r)

    # Eligible check must happen before reading rows_by_id so we surface
    # CALIBRATION_TOO_LARGE before any I/O.
    eligible_count = len(include_rows)
    if calibration_size > eligible_count:
        raise BatchError(
            "CALIBRATION_TOO_LARGE",
            f"--calibration-size ({calibration_size}) exceeds eligible count "
            f"({eligible_count}); refusing to silently shrink",
        )

    # Build the eligible index.
    rows_by_id: dict[str, dict[str, str]] = {}
    for r in include_rows:
        rows_by_id[r["datapoint_id"]] = r

    eligible = [_Eligible(r, seed) for r in include_rows]

    assignment = _assign(
        eligible,
        annotators=annotators,
        calibration_size=calibration_size,
        overlap_rate=overlap,
        seed=seed,
    )

    # ----- write outputs -----
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration_ids = assignment["calibration_ids"]
    calibration_rows = [rows_by_id[i] for i in calibration_ids]

    # Per-annotator TSVs.
    per_annotator_manifest: dict[str, dict[str, Any]] = {}
    per_annotator_totals: dict[str, int] = {}
    for annotator in annotators:
        calibration_rows_a, main_rows, overlap_rows = _build_annotator_rows(
            rows_by_id, assignment, annotator
        )
        path = out_dir / f"{annotator}.tsv"
        _write_annotator_file(
            path,
            calibration_rows=calibration_rows_a,
            main_rows=main_rows,
            overlap_rows=overlap_rows,
        )
        per = assignment["per_annotator"][annotator]
        # Recompute overlap count after dedup; the manifest records what
        # actually ended up in the file.
        overlap_count = len(overlap_rows)
        per_annotator_manifest[annotator] = {
            "main_batch_ids": per["main_batch_ids"],
            "overlap_ids": [r["datapoint_id"] for r in overlap_rows],
            "calibration_ids": per["calibration_ids"],
            "total_rows": (
                len(calibration_rows_a) + len(main_rows) + overlap_count
            ),
        }
        per_annotator_totals[annotator] = (
            len(calibration_rows_a) + len(main_rows) + overlap_count
        )

    # Shared calibration.tsv.
    _write_tsv(out_dir / "calibration.tsv", list(ALL_TSV_COLUMNS), calibration_rows)

    # Excluded / blocked outputs.
    if not drop_excluded:
        _write_tsv(out_dir / "excluded.tsv", list(ALL_TSV_COLUMNS), exclude_rows)
    _write_tsv(out_dir / "blocked_review.tsv", list(ALL_TSV_COLUMNS), blocked_rows)

    # Manifest — datapoint IDs are NOT text, they are deterministic
    # identifiers derived from public chapter + sentence-id + token
    # ranges. Still, to keep this file safe for stdout/CI, we keep it as
    # structured JSON rather than printing it.
    manifest = {
        "seed": seed,
        "annotators": list(annotators),
        "calibration_size": calibration_size,
        "overlap_rate": overlap,
        "calibration_ids": calibration_ids,
        "per_annotator": per_annotator_manifest,
        "excluded_count": len(exclude_rows),
        "blocked_count": len(blocked_rows),
        "total_eligible": eligible_count,
    }
    with open(out_dir / "batch_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    # Return the manifest plus stdout-safe totals.
    return {
        "__stdout__": {
            "annotators": len(annotators),
            "calibration_size": calibration_size,
            "overlap_rate": overlap,
            "eligible": eligible_count,
            "excluded": len(exclude_rows),
            "blocked": len(blocked_rows),
            "per_annotator": per_annotator_totals,
            "output_dir": str(out_dir),
        },
        "manifest": manifest,
    }


# --------------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--master-tsv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--annotators",
        nargs="+",
        required=True,
        metavar="NAME",
        help="≥2 unique annotator names.",
    )
    ap.add_argument("--seed", default=DEFAULT_SEED)
    ap.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    ap.add_argument("--calibration-size", type=int, default=DEFAULT_CALIBRATION)
    ap.add_argument(
        "--drop-excluded",
        action="store_true",
        help="Omit excluded.tsv entirely (excluded rows are dropped).",
    )
    ap.add_argument(
        "--force-output",
        action="store_true",
        help="Required if --out-dir exists and is non-empty.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)

    try:
        result = prepare_batches(
            master_tsv=args.master_tsv,
            out_dir=args.out_dir,
            annotators=args.annotators,
            seed=args.seed,
            overlap=args.overlap,
            calibration_size=args.calibration_size,
            drop_excluded=args.drop_excluded,
            force_output=args.force_output,
        )
    except BatchError as e:
        print(f"FAIL [{e.rule}]: {e.message}", file=sys.stderr)
        return 2

    out = result["__stdout__"]
    # Privacy-disciplined stdout. NEVER print datapoint IDs or text.
    print(f"annotators: {out['annotators']}")
    print(f"calibration_size: {out['calibration_size']}")
    print(f"overlap_rate: {out['overlap_rate']}")
    print(f"eligible: {out['eligible']}")
    print(f"excluded: {out['excluded']}")
    print(f"blocked: {out['blocked']}")
    pairs = ", ".join(
        f"{name}: {count}" for name, count in out["per_annotator"].items()
    )
    print(f"per_annotator: {{{pairs}}}")
    print(f"output_dir: {out['output_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
