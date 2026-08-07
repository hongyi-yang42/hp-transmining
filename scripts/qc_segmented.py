"""Aggregate QC for segmented JSONL files.

Walks ``data/segmented/hp1_*.jsonl``, prints per-file aggregate counts to
stdout (no novel text — only counts and IDs), and writes any problematic
rows to ``data/derived/qc/segmented_issues.tsv`` (gitignored) for offline
review.

Checks:
  * duplicate IDs
  * empty text
  * short fragments (< SHORT_THRESHOLD chars)
  * abnormally long sentences (> LONG_THRESHOLD chars)
  * malformed IDs (regex mismatch)
  * mixed language (filename-derived lang vs ID-derived lang)
  * mixed chapter (filename-derived chapter vs ID-derived chapter)
  * non-monotonic IDs ((page, sentence) ordinal decreases)
  * source-page discontinuities (gaps in union of source_pages)

Usage:
    uv run python scripts/qc_segmented.py [--segmented-dir data/segmented] \\
        [--output data/derived/qc/segmented_issues.tsv]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from hp_corpus.schema import Segment

SHORT_THRESHOLD = 5  # chars
LONG_THRESHOLD = 500  # chars
ID_RE = re.compile(
    r"^(?P<book>[a-z0-9]+)_(?P<lang>[a-z]{2,3})_ch(?P<ch>\d{2})_p(?P<page>\d{4})_s(?P<sent>\d{3})$"
)
FILENAME_RE = re.compile(r"^hp1_(?P<lang>[a-z]{2,3})_ch(?P<ch>\d{2})\.jsonl$")


def qc_one(path: Path) -> dict:
    """Run all QC checks on one segmented JSONL. Returns dict of aggregates
    plus a list of (issue_type, segment_id, detail) tuples for problem rows.
    """
    m_fn = FILENAME_RE.match(path.name)
    expected_lang = m_fn.group("lang") if m_fn else "?"
    expected_ch = int(m_fn.group("ch")) if m_fn else 0

    segs: list[Segment] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                segs.append(Segment.model_validate_json(line))

    issues: list[tuple[str, str, str]] = []
    seen_ids: set[str] = set()
    parsed: list[tuple[int, int] | None] = []  # (page, sentence) per segment

    for s in segs:
        if s.id in seen_ids:
            issues.append(("duplicate_id", s.id, ""))
        seen_ids.add(s.id)

        text = s.text.strip()
        if not text:
            issues.append(("empty_text", s.id, ""))
        elif len(text) < SHORT_THRESHOLD:
            issues.append(("short_fragment", s.id, f"len={len(text)}"))
        if len(s.text) > LONG_THRESHOLD:
            issues.append(("abnormally_long", s.id, f"len={len(s.text)}"))

        m = ID_RE.match(s.id)
        if not m:
            issues.append(("malformed_id", s.id, ""))
            parsed.append(None)
            continue

        id_lang = m.group("lang")
        id_ch = int(m.group("ch"))
        if id_lang != expected_lang:
            issues.append(
                ("mixed_language", s.id, f"file={expected_lang} id={id_lang}")
            )
        if id_ch != expected_ch:
            issues.append(
                ("mixed_chapter", s.id, f"file=ch{expected_ch:02d} id=ch{id_ch:02d}")
            )

        page = int(m.group("page"))
        sent = int(m.group("sent"))
        parsed.append((page, sent))

    # Non-monotonic check.
    for i in range(1, len(parsed)):
        if parsed[i] is None or parsed[i - 1] is None:
            continue
        if parsed[i] < parsed[i - 1]:
            issues.append(
                ("non_monotonic_id", segs[i].id, f"prev={parsed[i - 1]} curr={parsed[i]}")
            )

    # Source-page discontinuity.
    all_pages = sorted({p for s in segs for p in s.source_pages})
    page_gaps: list[tuple[int, int]] = []
    for i in range(1, len(all_pages)):
        if all_pages[i] - all_pages[i - 1] > 1:
            page_gaps.append((all_pages[i - 1], all_pages[i]))

    return {
        "path": path.name,
        "expected_lang": expected_lang,
        "expected_chapter": expected_ch,
        "segment_count": len(segs),
        "duplicate_id_count": sum(1 for x in issues if x[0] == "duplicate_id"),
        "empty_count": sum(1 for x in issues if x[0] == "empty_text"),
        "short_fragment_count": sum(1 for x in issues if x[0] == "short_fragment"),
        "abnormally_long_count": sum(1 for x in issues if x[0] == "abnormally_long"),
        "malformed_id_count": sum(1 for x in issues if x[0] == "malformed_id"),
        "mixed_language_count": sum(1 for x in issues if x[0] == "mixed_language"),
        "mixed_chapter_count": sum(1 for x in issues if x[0] == "mixed_chapter"),
        "non_monotonic_id_count": sum(1 for x in issues if x[0] == "non_monotonic_id"),
        "source_page_gap_count": len(page_gaps),
        "page_min": all_pages[0] if all_pages else 0,
        "page_max": all_pages[-1] if all_pages else 0,
        "issues": issues,
        "page_gaps": page_gaps,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--segmented-dir", type=Path, default=Path("data/segmented"))
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/qc/segmented_issues.tsv"),
    )
    args = ap.parse_args(argv)

    files = sorted(args.segmented_dir.glob("hp1_*.jsonl"))
    if not files:
        print(f"no segmented files in {args.segmented_dir}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "file",
        "segs",
        "dup",
        "empty",
        "short",
        "long",
        "malformed",
        "mix_lang",
        "mix_ch",
        "nonmono",
        "pggap",
        "pg_min",
        "pg_max",
    ]
    rows: list[dict] = []
    all_issues: list[tuple[str, str, str, str]] = []
    all_gaps: list[tuple[str, int, int]] = []

    for path in files:
        r = qc_one(path)
        rows.append(r)
        for issue in r["issues"]:
            all_issues.append((r["path"], issue[0], issue[1], issue[2]))
        for gap in r["page_gaps"]:
            all_gaps.append((r["path"], gap[0], gap[1]))

    # Stdout: aggregate table (counts + page ranges only; no novel text).
    print(f"files: {len(files)}")
    print(
        f"thresholds: short<{SHORT_THRESHOLD} chars, long>{LONG_THRESHOLD} chars"
    )
    print()
    fmt = "{:<28}{:>7}{:>5}{:>6}{:>6}{:>6}{:>9}{:>9}{:>8}{:>9}{:>7}{:>7}{:>7}"
    print(fmt.format(*headers))
    totals = {h: 0 for h in headers[1:]}
    for r in rows:
        line = [
            r["path"],
            r["segment_count"],
            r["duplicate_id_count"],
            r["empty_count"],
            r["short_fragment_count"],
            r["abnormally_long_count"],
            r["malformed_id_count"],
            r["mixed_language_count"],
            r["mixed_chapter_count"],
            r["non_monotonic_id_count"],
            r["source_page_gap_count"],
            r["page_min"],
            r["page_max"],
        ]
        print(fmt.format(*[str(x) for x in line]))
        for k, v in zip(headers[1:], line[1:], strict=True):
            if isinstance(v, int):
                totals[k] += v
    print()
    print(fmt.format("TOTAL", *[str(totals[h]) for h in headers[1:]]))
    print()

    # Write issues TSV (gitignored). No novel text — only issue type + segment ID.
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["file", "issue_type", "segment_id", "detail"])
        for row in all_issues:
            w.writerow(row)
        if all_gaps:
            w.writerow([])
            w.writerow(["# source-page gaps (file, from_page, to_page)"])
            for fname, a, b in all_gaps:
                w.writerow([fname, "page_gap", f"{a}->{b}", ""])

    print(f"issues → {args.output}  ({len(all_issues)} rows + {len(all_gaps)} page gaps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
