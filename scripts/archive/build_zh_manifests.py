"""Write per-chapter source manifests for the unified new-source ZH Ch.1-17
standard-path outputs.

One manifest per chapter under data/derived/zh_source_audit/manifests/,
recording: source PDF identity (path + sha256), page range, the exact
cleaning config (path + sha256), output paths with aggregate counts, and the
ordered segment-ID digest. Supersedes the Ch.7-17 "decision pending"
manifests written by zh_source_audit.py: Source Option A was approved
2026-08-15.

Source label discipline: the source is a calibre 3.39.1 ebook-to-PDF
conversion created 2019-02-16. Its print-edition provenance is unverified —
never describe it as a "2019 print edition". The old 2018 scan remains in
data/raw/ for copyright/version evidence only.

Stdout: aggregate counts and paths only — never novel text.

Usage:
    uv run python scripts/build_zh_manifests.py
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

MANIFEST_DIR = Path("data/derived/zh_source_audit/manifests")
ZH_SOURCE_VERSION = "中文_哈利波特与魔法石_Z_2019_calibre"
PROVENANCE_NOTE = (
    "calibre PDF created 2019-02-16; print-edition provenance unverified "
    "(NOT a 2019 print edition); Su Nong translation lineage established by "
    "17/17 TOC chapter-title match + Ch.1-4 body match vs the 2018 scan"
)
DECISION_REF = (
    "Source Option A approved by coordinator 2026-08-15; see "
    "data/derived/zh_source_audit/SOURCE_DECISION_MEMO.md"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _id_digest(ids: list[str]) -> str:
    h = hashlib.sha256()
    for sid in ids:
        h.update(sid.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def build_manifest(ch: int, pdf_sha: str) -> dict[str, Any]:
    cfg_path = Path(f"config/hp1_zh_ch{ch:02d}.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ch_cfg = cfg["chapter"]
    seg_path = Path(f"data/segmented/hp1_zh_ch{ch:02d}.jsonl")
    paras = [json.loads(ln) for ln in
             Path(f"data/text_clean/hp1_zh_ch{ch:02d}.jsonl").read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    segs = [json.loads(ln) for ln in
            seg_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    notes_p = Path(f"data/text_clean/hp1_zh_ch{ch:02d}_notes.jsonl")
    n_notes = len([ln for ln in notes_p.read_text(encoding="utf-8").splitlines() if ln.strip()]) \
        if notes_p.exists() else 0
    return {
        "book": "hp1",
        "lang": "zh",
        "chapter": ch,
        "zh_source": ZH_SOURCE_VERSION,
        "provenance_note": PROVENANCE_NOTE,
        "decision": DECISION_REF,
        "source_pdf": {
            "path": cfg["pdf_path"],
            "sha256": pdf_sha,
            "pages": cfg["total_pages"],
            "producer": "calibre 3.39.1",
            "creation_date": "2019-02-16",
            "edition_note": "ebook-to-PDF conversion; no printed-page correspondence",
        },
        "page_range": {"start": ch_cfg["start_page"], "end": ch_cfg["end_page"]},
        "cleaning_config": {
            "path": str(cfg_path),
            "sha256": _sha256(cfg_path),
            "header_patterns": cfg["clean"].get("header_patterns", []),
            "footnote_spans": cfg["clean"].get("footnote_spans"),
        },
        "outputs": {
            "ocr_raw": f"data/ocr_raw/hp1_zh_ch{ch:02d}.jsonl",
            "text_clean": f"data/text_clean/hp1_zh_ch{ch:02d}.jsonl",
            "text_clean_txt": f"data/text_clean/hp1_zh_ch{ch:02d}.txt",
            "notes": str(notes_p),
            "segmented": str(seg_path),
        },
        "counts": {
            "paragraphs": len(paras),
            "segments": len(segs),
            "body_chars": sum(len(s["text"]) for s in segs),
            "notes": n_notes,
        },
        "id_digest_sha256": _id_digest([s["id"] for s in segs]),
        "status_note": (
            "unified ZH source (Option A) standard-path output; Ch.1-3 "
            "machine alignments rebuilt on this source; NOT a final trilingual "
            "master; human annotation references NOT migrated"
        ),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }


def main() -> int:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    pdf_sha = _sha256(Path("data/raw/中文_哈利波特与魔法石 Z.pdf"))
    for ch in range(1, 18):
        m = build_manifest(ch, pdf_sha)
        out = MANIFEST_DIR / f"hp1_zh_ch{ch:02d}.manifest.json"
        out.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        c = m["counts"]
        print(f"ch{ch:02d}: pp.{m['page_range']['start']}-{m['page_range']['end']} "
              f"paras={c['paragraphs']} segs={c['segments']} chars={c['body_chars']} "
              f"notes={c['notes']} → {out.name}")
    print(f"\n{17} manifests → {MANIFEST_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
