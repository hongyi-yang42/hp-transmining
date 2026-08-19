"""Regression gate for retrieval-context quality on reviewed datapoints.

Companion to ``data/derived/annotation/context_regression.json`` (IDs only,
no corpus text): for every reviewed datapoint × side, the master TSV's
retrieval context must either contain the human-verified expected-locus
segment id or carry ``*_context_provenance = manual_review`` — a reviewed
locus may not silently fall back to an unrelated window.

Fail-closed (exit 2, aggregate stdout only — no text, lemmas, or ids):
missing manifest or master; manifest case absent from the master; a case
side whose locus is outside the context while provenance is not
``manual_review``.

Usage::

    uv run python scripts/check_context_regression.py \\
        --manifest data/derived/annotation/context_regression.json \\
        --master-tsv data/derived/step4/full_novel_annotation_master.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

EXIT_OK = 0
EXIT_INPUT_ERROR = 2

SIDES = ("en", "zh")


def _fail(rule: str, message: str) -> int:
    print(f"FAIL: rule={rule} {message}", file=sys.stderr)
    return EXIT_INPUT_ERROR


def read_cases(manifest_path: Path) -> dict[str, dict[str, list[str]]]:
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    cases = data["cases"] if isinstance(data, dict) and "cases" in data else data
    if not isinstance(cases, dict) or not cases:
        raise ValueError("manifest carries no cases")
    return cases


def check(
    cases: dict[str, dict[str, list[str]]],
    master_rows: dict[str, dict[str, str]],
) -> tuple[list[str], dict[str, int]]:
    """Returns (rule-level errors, counts). Never leaks row content or ids."""
    errors: list[str] = []
    counts = {"cases": len(cases), "sides": 0, "in_context": 0, "manual_review": 0}
    for dp, sides in cases.items():
        master = master_rows.get(dp)
        if master is None:
            errors.append("CASE_NOT_IN_MASTER")
            continue
        for side in SIDES:
            expected = sides.get(side, [])
            if not expected:
                continue
            counts["sides"] += 1
            raw = (master.get(f"{side}_context_ids", "") or "").strip()
            try:
                ctx_ids = set(json.loads(raw)) if raw else set()
            except json.JSONDecodeError:
                errors.append("CONTEXT_IDS_MALFORMED")
                continue
            if set(expected) <= ctx_ids:
                counts["in_context"] += 1
            elif (master.get(f"{side}_context_provenance", "") or "").strip() == "manual_review":
                counts["manual_review"] += 1
            else:
                errors.append("LOCUS_NOT_COVERED")
    return errors, counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--master-tsv", type=Path, required=True)
    args = ap.parse_args(argv)

    if not args.manifest.exists():
        return _fail("MANIFEST_ABSENT", str(args.manifest))
    if not args.master_tsv.exists():
        return _fail("MASTER_TSV_ABSENT", str(args.master_tsv))

    try:
        cases = read_cases(args.manifest)
    except (json.JSONDecodeError, ValueError) as exc:
        return _fail("MANIFEST_INVALID", str(exc))

    master_rows: dict[str, dict[str, str]] = {}
    with open(args.master_tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            dp = (row.get("datapoint_id", "") or "").strip()
            if dp:
                master_rows[dp] = row

    errors, counts = check(cases, master_rows)
    print(f"cases: {counts['cases']}")
    print(f"sides checked: {counts['sides']}")
    print(f"in-context: {counts['in_context']}")
    print(f"manual_review: {counts['manual_review']}")
    print(f"violations: {len(errors)}")
    for e in errors[:20]:
        print(f"  {e}")
    return EXIT_INPUT_ERROR if errors else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
