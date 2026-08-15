"""Sentence-ID impact analysis: old-source vs new-source ZH Ch.1-6.

Compares the archived old-source segmented IDs
(data/derived/zh_source_audit/legacy_archive/segmented/) against the live
new-source segmented IDs (data/segmented/) and reports, per chapter:

  * retained IDs (same id both sides), split by text unchanged / changed
  * added IDs (new only) and removed IDs (old only)
  * longest common ID prefix (stability prefix) — IDs ahead of the first
    paragraph-ordinal drift are stable because IDs key on paragraph and
    sentence ordinals (schema: {book}_{lang}_ch{NN}_p{NNNN}_s{NNN}), not
    page numbers

Then scans downstream derived artifacts for references to ZH Ch.1-6 IDs and
counts how many are still valid vs now-stale (id absent from the new-source
ID set). Scan is read-only; human annotation files are counted, never
modified.

Writes data/derived/zh_source_audit/id_impact_ch01_06.json.
Stdout: aggregate counts only — no novel text.

Usage:
    uv run python scripts/zh_id_impact.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ARCHIVE = Path("data/derived/zh_source_audit/legacy_archive")
AUDIT = Path("data/derived/zh_source_audit")
_ID_RE = re.compile(r"hp1_zh_ch0[1-6]_p\d{4}_s\d{3}")

# Read-only scan targets for stale-id references (never modified).
SCAN_DIRS = ("data/derived/step4", "data/derived/qc", "data/derived/sampling")


def _load_ids_texts(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            rec = json.loads(ln)
            out[rec["id"]] = rec["text"]
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text)


def chapter_impact(ch: int) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    old = _load_ids_texts(ARCHIVE / "segmented" / f"hp1_zh_ch{ch:02d}.jsonl")
    new = _load_ids_texts(Path(f"data/segmented/hp1_zh_ch{ch:02d}.jsonl"))
    old_ids, new_ids = list(old), list(new)

    common = set(old) & set(new)
    same_text = {i for i in common if _norm(old[i]) == _norm(new[i])}
    diff_text = common - same_text
    added = set(new) - set(old)
    removed = set(old) - set(new)

    # Similarity buckets for retained-but-changed ids: distinguishes "ordinal
    # collision, text moved elsewhere" from "same sentence, minor OCR/punct
    # noise". difflib on normalized short sentences is cheap enough here.
    from difflib import SequenceMatcher

    buckets = {"ge_0p99": 0, "ge_0p95": 0, "ge_0p90": 0, "lt_0p90": 0}
    for i in diff_text:
        r = SequenceMatcher(None, _norm(old[i]), _norm(new[i]), autojunk=False).ratio()
        if r >= 0.99:
            key = "ge_0p99"
        elif r >= 0.95:
            key = "ge_0p95"
        elif r >= 0.90:
            key = "ge_0p90"
        else:
            key = "lt_0p90"
        buckets[key] += 1

    # Longest common prefix of the ordered ID lists: IDs from paragraphs
    # before the first ordinal drift are untouched.
    prefix = 0
    for a, b in zip(old_ids, new_ids, strict=False):
        if a != b:
            break
        prefix += 1

    return {
        "chapter": ch,
        "old_segments": len(old),
        "new_segments": len(new),
        "retained_total": len(common),
        "retained_same_text": len(same_text),
        "retained_text_changed": len(diff_text),
        "retained_changed_similarity": buckets,
        "added": len(added),
        "removed": len(removed),
        "stable_prefix_ids": prefix,
        "first_divergence": {
            "old": old_ids[prefix] if prefix < len(old_ids) else None,
            "new": new_ids[prefix] if prefix < len(new_ids) else None,
        },
    }, old, new


def stale_reference_scan(
    valid_ids: set[str], old_texts: dict[str, str], new_texts: dict[str, str]
) -> dict[str, Any]:
    """Bucket each ZH Ch.1-6 id reference in derived artifacts.

    A reference was written against the OLD corpus, so "safe" requires BOTH
    string survival in the new corpus AND unchanged text at that id — a
    surviving id whose text moved is a silent wrong reference, worse than a
    dangling one."""
    per_file: dict[str, Any] = {}
    for d in SCAN_DIRS:
        base = Path(d)
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix not in (".jsonl", ".tsv", ".json", ".md"):
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            refs = _ID_RE.findall(text)
            if not refs:
                continue
            text_stable = sum(
                1 for r in refs
                if r in old_texts and r in new_texts
                and _norm(old_texts[r]) == _norm(new_texts[r])
            )
            text_shifted = sum(1 for r in refs if r in valid_ids) - text_stable
            not_in_new = sum(1 for r in refs if r not in valid_ids)
            per_file[str(p)] = {
                "zh_ch01_6_refs": len(refs),
                "text_stable": text_stable,
                "string_survived_but_text_shifted": text_shifted,
                "not_in_new": not_in_new,
            }
    return per_file


def main() -> int:
    valid_ids: set[str] = set()
    old_texts: dict[str, str] = {}
    new_texts: dict[str, str] = {}
    impacts = []
    for ch in range(1, 7):
        imp, old, new = chapter_impact(ch)
        impacts.append(imp)
        fd = imp["first_divergence"]
        print(f"ch{ch:02d}: old={imp['old_segments']} new={imp['new_segments']} "
              f"retained={imp['retained_total']} (same_text={imp['retained_same_text']}, "
              f"text_changed={imp['retained_text_changed']} "
              f"sim={imp['retained_changed_similarity']}) "
              f"added={imp['added']} removed={imp['removed']} "
              f"stable_prefix={imp['stable_prefix_ids']} "
              f"diverge@ old={fd['old']} new={fd['new']}")
        old_texts |= old
        new_texts |= new
    valid_ids = set(new_texts)

    scan = stale_reference_scan(valid_ids, old_texts, new_texts)
    print("\nDownstream ZH Ch.1-6 id references (read-only scan):")
    for f, s in sorted(scan.items()):
        print(f"  {f}: refs={s['zh_ch01_6_refs']} text_stable={s['text_stable']} "
              f"shifted={s['string_survived_but_text_shifted']} "
              f"not_in_new={s['not_in_new']}")

    totals = {
        f: sum(i[f] for i in impacts)
        for f in ("old_segments", "new_segments", "retained_total", "retained_same_text",
                  "retained_text_changed", "added", "removed")
    }
    (AUDIT / "id_impact_ch01_06.json").write_text(
        json.dumps({"per_chapter": impacts, "totals": totals,
                    "stale_reference_scan": scan},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\ntotals: {totals}")
    print(f"→ {AUDIT / 'id_impact_ch01_06.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
