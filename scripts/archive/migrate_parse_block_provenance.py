"""Deterministic block-level provenance migration for existing CoNLL-U files.

Backfills ``parse_block_id`` / ``source_segment_id`` onto already-parsed
CoNLL-U files without re-running Stanza, using the exact ID construction
rule the parser emits (``hp_corpus.provenance``): one shared module, one
algorithm, so a text migration and a re-parse produce identical ids.

For every ``hp1_{lang}_chNN.conllu`` (raw parses; ``_nomwt`` variants
are regenerated downstream by ``normalize_conllu_mwt.py``, which passes
comment lines through verbatim):

* ``# sent_id = <segment_id>`` shared by several blocks becomes
  ``# sent_id = <segment_id>#bNNN`` (unique per file) plus a
  ``# source_segment_id = <segment_id>`` line;
* already-migrated files are validated and passed through unchanged
  (idempotent);
* any inconsistent state (mixed migrated/unmigrated, malformed ids,
  duplicate ids) fails closed with a non-zero exit — the file is never
  partially rewritten.

Stdout and the stats JSON carry aggregate counts only — no sentence
text, no segment/block ids. The stats JSON path defaults under
``data/derived/`` (gitignored); it is working evidence, not a tracked
artifact.

Usage::

    uv run python scripts/migrate_parse_block_provenance.py \\
        [--parsed-dir data/parsed] [--langs de en zh] \\
        [--stats-out data/derived/audit/parse_block_migration/stats.json] \\
        [--dry-run]

After migrating, regenerate the DE ``_nomwt`` inputs consumed by the
extractor::

    for ch in $(seq 1 17); do
        uv run python scripts/normalize_conllu_mwt.py \\
            --input data/parsed/hp1_de_ch$(printf '%02d' $ch).conllu \\
            --output data/parsed/hp1_de_ch$(printf '%02d' $ch)_nomwt.conllu
    done
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hp_corpus.provenance import (  # noqa: E402
    ProvenanceError,
    migrate_conllu_text,
    provenance_counts,
    scan_blocks,
)

RAW_CONLLU_RE = re.compile(r"^hp1_(de|en|zh)_ch\d{2}\.conllu$")


def find_raw_conllu(parsed_dir: Path, langs: list[str]) -> list[Path]:
    return sorted(
        p
        for p in parsed_dir.glob("*.conllu")
        if RAW_CONLLU_RE.match(p.name) and p.name.split("_")[1] in langs
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Add parse_block_id/source_segment_id to existing CoNLL-U files."
    )
    ap.add_argument("--parsed-dir", type=Path, default=Path("data/parsed"))
    ap.add_argument("--langs", nargs="+", default=["de", "en", "zh"], choices=["de", "en", "zh"])
    ap.add_argument(
        "--stats-out",
        type=Path,
        default=Path("data/derived/audit/parse_block_migration/stats.json"),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    files = find_raw_conllu(args.parsed_dir, args.langs)
    if not files:
        print(f"no raw hp1_*.conllu files found in {args.parsed_dir}", file=sys.stderr)
        return 2

    per_file: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
            new_text, stats = migrate_conllu_text(text)
            after = provenance_counts(scan_blocks(new_text.splitlines()))
            if not args.dry_run and new_text != text:
                tmp = path.with_suffix(".conllu.tmp")
                tmp.write_text(new_text, encoding="utf-8")
                tmp.replace(path)
            entry: dict[str, object] = {"file": path.name, **stats.as_dict(), "after": after}
        except ProvenanceError as exc:
            kind = type(exc).__name__
            errors.append({"file": path.name, "error": kind})
            entry = {"file": path.name, "error": kind}
        per_file.append(entry)

    if errors:
        for e in errors:
            print(f"FAIL: {e['file']}: {e['error']}", file=sys.stderr)
        return 1

    args.stats_out.parent.mkdir(parents=True, exist_ok=True)
    args.stats_out.write_text(
        json.dumps({"files": per_file}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    blocks = sum(f["blocks_total"] for f in per_file if isinstance(f.get("blocks_total"), int))  # type: ignore[union-attr]
    migrated = sum(
        f["blocks_migrated"] for f in per_file if isinstance(f.get("blocks_migrated"), int)  # type: ignore[attr-defined]
    )
    already = sum(
        f["blocks_already_migrated"]
        for f in per_file
        if isinstance(f.get("blocks_already_migrated"), int)  # type: ignore[attr-defined]
    )
    after_totals = [
        f["after"] for f in per_file if isinstance(f.get("after"), dict)  # type: ignore[attr-defined]
    ]
    print(f"files: {len(files)} (dry_run={args.dry_run})")
    print(f"blocks_total: {blocks} | migrated: {migrated} | already_migrated: {already}")
    print(
        "after: blocks={b} unique_parse_block_ids={u} multi_block_segments={m} "
        "max_blocks_per_segment={x}".format(
            b=sum(a["blocks_total"] for a in after_totals),
            u=sum(a["distinct_parse_block_ids"] for a in after_totals),
            m=sum(a["multi_block_segments"] for a in after_totals),
            x=max((a["max_blocks_per_segment"] for a in after_totals), default=0),
        )
    )
    print(f"stats: {args.stats_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
