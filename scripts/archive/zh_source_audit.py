"""Aggregate-only source audit for the new (calibre 2019) ZH ebook vs the old
(2018 23rd-printing scan) source.

Writes everything to data/derived/zh_source_audit/ — no novel text reaches
stdout; only counts and classification tallies are printed.

Outputs:
  new_source_ch{01..06}.clean.jsonl / .txt / _notes.jsonl — audit-only copies
      of the new-source Ch.1-6 cleaning (standard data/ paths are NOT touched;
      Ch.1-6 standard outputs stay on the old source).
  per_chapter_stats.json — per-chapter aggregate table, both sources.
  source_evidence.json   — PDF metadata / TOC / structure evidence.
  ch06_diff_report.json  — Ch.6 absent-region enumeration + classification.
"""

from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import fitz
import yaml

from hp_corpus.clean import _is_decorative, _looks_like_page_number, _strip_block_text, clean_blocks
from hp_corpus.render import extract_text_layer_blocks
from hp_corpus.schema import OCRBlock
from hp_corpus.segment import segment_all

AUDIT = Path("data/derived/zh_source_audit")
NEW_PDF = "data/raw/中文_哈利波特与魔法石 Z.pdf"
OLD_PDF = "data/raw/hp1_zh.pdf"
ZH_SOURCE_VERSION = "中文_哈利波特与魔法石_Z_2019_calibre"

# New-source chapter page ranges from the embedded TOC (heading page → next
# heading page - 1; trailing 脚注 p.232 excluded).
NEW_RANGES = {
    1: (5, 16), 2: (17, 25), 3: (26, 36), 4: (37, 46), 5: (47, 65),
    6: (66, 84), 7: (85, 97), 8: (98, 106), 9: (107, 121), 10: (122, 134),
    11: (135, 145), 12: (146, 161), 13: (162, 170), 14: (171, 180),
    15: (181, 195), 16: (196, 215), 17: (216, 231),
}
# New-source heading texts (from TOC), used as header_patterns so the heading
# is recorded as chapter title and kept out of the body.
NEW_HEADERS = {
    1: "第１章　大难不死的男孩", 2: "第２章　悄悄消失的玻璃", 3: "第３章　猫头鹰传书",
    4: "第４章　钥匙保管员", 5: "第５章　对角巷", 6: "第６章　从9又3/4站台开始的旅程",
    7: "第７章　分院帽", 8: "第８章　魔药课老师", 9: "第９章　午夜决斗",
    10: "第10章　万圣节前夜", 11: "第11章　魁地奇比赛", 12: "第12章　厄里斯魔镜",
    13: "第13章　尼可·勒梅", 14: "第14章　挪威脊背龙——诺伯", 15: "第15章　禁林",
    16: "第16章　穿越活板门", 17: "第17章　双面人",
}


def new_config(ch: int) -> dict[str, Any]:
    return {
        "book": "hp1", "lang": "zh",
        "chapter": {"number": ch, "start_page": NEW_RANGES[ch][0], "end_page": NEW_RANGES[ch][1]},
        "ocr": {"engine": "pymupdf", "lang": "zh"},
        "clean": {
            "header_patterns": [NEW_HEADERS[ch]],
            "remove_page_numbers": True,
            "footnote_spans": {"marker_max_size": 7.0, "note_max_size": 9.0},
            "merge_line_breaks": True,
            "paragraph_detection": {"indent_threshold": 2, "dialogue_markers": ["“", "”"]},
        },
        "segment": {"lang": "zh", "split_on": ["。", "！", "？", "；"], "preserve_ellipsis": True},
    }


def classify_drops(blocks: list[OCRBlock], header_patterns: list[str]) -> dict[str, int]:
    """Re-count the per-block drop reasons the clean stage applies."""
    counts = {"headers": 0, "page_numbers": 0, "decorative": 0, "low_conf": 0}
    for b in blocks:
        t = _strip_block_text(b.text)
        if any(t == p or re.sub(r"\s+", "", t) == re.sub(r"\s+", "", p) for p in header_patterns):
            counts["headers"] += 1
        elif _looks_like_page_number(t):
            counts["page_numbers"] += 1
        elif _is_decorative(t):
            counts["decorative"] += 1
        elif b.confidence < 0.4:
            counts["low_conf"] += 1
    return counts


def audit_new_source() -> dict[int, dict[str, Any]]:
    """Process new-source Ch.1-17 (audit copies for Ch.1-6; standard outputs
    already exist for Ch.7-17)."""
    per_ch: dict[int, dict[str, Any]] = {}
    for ch in range(1, 18):
        cfg = new_config(ch)
        blocks = extract_text_layer_blocks(NEW_PDF, *NEW_RANGES[ch])
        drops = classify_drops(blocks, cfg["clean"]["header_patterns"])
        marker_count = sum(
            1 for b in blocks for line in b.lines for s in line.spans
            if s.size < 7.0 and s.text.strip().isdigit()
        )
        result = clean_blocks(blocks, cfg)
        segs = segment_all(result.sentences, "hp1", "zh", ch, cfg)
        entry: dict[str, Any] = {
            "source": ZH_SOURCE_VERSION,
            "pages": NEW_RANGES[ch][1] - NEW_RANGES[ch][0] + 1,
            "raw_blocks": len(blocks),
            "paragraphs": len(result.sentences),
            "segments": len(segs),
            "body_chars": sum(len(s.text) for s in segs),
            "notes": len(result.footnotes),
            "marker_digits": marker_count,
            "drops": drops,
        }
        entry["absent_regions"] = None  # filled by diff pass for ch1-6
        if ch <= 6:  # audit-only copies; never touch data/text_clean for ch1-6
            AUDIT.joinpath(f"new_source_ch{ch:02d}.txt").write_text(
                "\n".join(s.text for s in result.sentences), encoding="utf-8"
            )
            AUDIT.joinpath(f"new_source_ch{ch:02d}.notes.jsonl").write_text(
                "".join(json.dumps({"text": n.text}, ensure_ascii=False) + "\n"
                        for n in result.footnotes),
                encoding="utf-8",
            )
            AUDIT.joinpath(f"new_source_ch{ch:02d}.clean.jsonl").write_text(
                "".join(s.model_dump_json() + "\n" for s in result.sentences), encoding="utf-8"
            )
        per_ch[ch] = entry
    return per_ch


def audit_old_source() -> dict[int, dict[str, Any]]:
    per_ch: dict[int, dict[str, Any]] = {}
    for ch in range(1, 7):
        name = f"hp1_zh_ch{ch:02d}" if ch >= 2 else "hp1_zh"
        cfg = yaml.safe_load(Path(f"config/{name}.yaml").read_text())
        raw = [OCRBlock.model_validate_json(ln) for ln in
               Path(f"data/ocr_raw/hp1_zh_ch{ch:02d}.jsonl").read_text().splitlines() if ln.strip()]
        drops = classify_drops(raw, cfg["clean"].get("header_patterns", []))
        segs = [json.loads(ln) for ln in
                Path(f"data/segmented/hp1_zh_ch{ch:02d}.jsonl").read_text().splitlines()
                if ln.strip()]
        paras = [json.loads(ln) for ln in
                 Path(f"data/text_clean/hp1_zh_ch{ch:02d}.jsonl").read_text().splitlines()
                 if ln.strip()]
        notes_p = Path(f"data/text_clean/hp1_zh_ch{ch:02d}_notes.jsonl")
        notes = len(notes_p.read_text().splitlines()) if notes_p.exists() else 0
        per_ch[ch] = {
            "source": "hp1_zh_2018_scan_paddleocr",
            "pages": cfg["chapter"]["end_page"] - cfg["chapter"]["start_page"] + 1,
            "raw_blocks": len(raw),
            "paragraphs": len(paras),
            "segments": len(segs),
            "body_chars": sum(len(s["text"]) for s in segs),
            "notes": notes,
            "drops": drops,
            "mean_conf": round(sum(b.confidence for b in raw) / len(raw), 4) if raw else 0,
        }
    return per_ch


_WS_PUNCT_RE = re.compile(r"[\s　，。！？；：、""''（）　-〿""·—…《》“”‘’]+")


def norm_stream(text: str) -> str:
    return _WS_PUNCT_RE.sub("", text)


def chapter_diff(ch: int) -> dict[str, Any]:
    """Enumerate regions present in the new source but absent from the old
    source's cleaned text, then classify each against the old raw OCR.

    Run for Ch.1-6: Ch.1-5 act as controls (editions identical → near-zero
    regions), Ch.6 is the chapter under diagnosis."""
    new_paras = [json.loads(ln)["text"] for ln in
                 AUDIT.joinpath(f"new_source_ch{ch:02d}.clean.jsonl").read_text().splitlines()
                 if ln.strip()]
    old_txt = Path(f"data/text_clean/hp1_zh_ch{ch:02d}.txt").read_text(encoding="utf-8")
    old_raw = [OCRBlock.model_validate_json(ln) for ln in
               Path(f"data/ocr_raw/hp1_zh_ch{ch:02d}.jsonl").read_text().splitlines() if ln.strip()]

    new_stream, starts = "", []
    for p in new_paras:
        starts.append(len(new_stream))
        new_stream += norm_stream(p)
    old_stream = norm_stream(old_txt)
    old_raw_stream = norm_stream("".join(b.text for b in old_raw))

    sm = SequenceMatcher(None, old_stream, new_stream, autojunk=False)
    regions: list[dict[str, Any]] = []
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag not in ("insert", "replace"):
            continue
        if j2 - j1 < 20:  # ignore sub-sentence noise
            continue
        lo = max(i for i, s in enumerate(starts) if s <= j1)
        hi = max(i for i, s in enumerate(starts) if s < j2)
        region_new = new_stream[j1:j2]
        in_old_clean = _best_ratio(old_stream, region_new)
        in_old_raw = _best_ratio(old_raw_stream, region_new)
        regions.append({
            "tag": tag,
            "new_stream_len": j2 - j1,
            "para_range": [lo, hi],
            "para_texts": new_paras[lo:hi + 1],  # file-only, never stdout
            "best_ratio_old_clean": round(in_old_clean, 3),
            "best_ratio_old_raw": round(in_old_raw, 3),
            "verdict": _verdict(in_old_clean, in_old_raw),
        })
    return {
        "old_clean_chars": len(old_stream),
        "new_clean_chars": len(new_stream),
        "old_raw_blocks": len(old_raw),
        "regions": regions,
        "region_count": len(regions),
        "verdict_counts": _tally(regions),
    }


def _best_ratio(haystack: str, needle: str, window: int = 300) -> float:
    """Best local similarity of needle against a sliding window of haystack."""
    if not needle:
        return 0.0
    step = max(1, window // 3)
    best = 0.0
    for i in range(0, max(1, len(haystack) - window + 1), step):
        r = SequenceMatcher(None, haystack[i:i + window], needle[:window], autojunk=False).ratio()
        if r > best:
            best = r
        if best > 0.95:
            break
    return best


def _verdict(clean: float, raw: float) -> str:
    if clean >= 0.8:
        return "present_in_old_clean (no real loss)"
    if raw >= 0.8:
        return "old_raw_has_clean_dropped"
    if raw >= 0.55:
        return "old_raw_partial (OCR degraded)"
    return "absent_from_old_source"


def _tally(regions: list[dict[str, Any]]) -> dict[str, int]:
    t: dict[str, int] = {}
    for r in regions:
        t[r["verdict"]] = t.get(r["verdict"], 0) + 1
    return t


def _physical_lines(page: fitz.Page, dpi: int = 300) -> int:
    """Count text-line bands on a rendered page via row projection profile."""
    import numpy as np

    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    row_has = (arr < 128).sum(axis=1) > 3
    bands, start = [], None
    for i, v in enumerate(row_has):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= 8:
                bands.append((start, i))
            start = None
    if start is not None:
        bands.append((start, len(row_has)))
    return len(bands)


def line_deficit() -> dict[int, dict[str, Any]]:
    """Per-chapter physical-line vs OCR-line counts for the old scan.

    The projection profile counts every physical text band (body lines, the
    running header, the page number), so the baseline is not zero even for a
    perfect page; the signal is the *excess* deficit and the worst pages.
    """
    old_ranges = {1: (7, 18), 2: (19, 28), 3: (29, 40), 4: (41, 52), 5: (53, 73), 6: (74, 92)}
    out: dict[int, dict[str, Any]] = {}
    with fitz.open(OLD_PDF) as doc:
        for ch, (a, b) in old_ranges.items():
            raw = [OCRBlock.model_validate_json(ln) for ln in
                   Path(f"data/ocr_raw/hp1_zh_ch{ch:02d}.jsonl").read_text().splitlines()
                   if ln.strip()]
            pages = []
            for p in range(a, b + 1):
                phys = _physical_lines(doc[p - 1])
                ocr = len([x for x in raw if x.page == p])
                pages.append(
                    {"page": p, "physical": phys, "ocr": ocr, "missed": max(0, phys - ocr)}
                )
            out[ch] = {
                "total_missed": sum(x["missed"] for x in pages),
                "worst_pages": sorted(
                    [{"page": x["page"], "missed": x["missed"]} for x in pages if x["missed"] >= 5],
                    key=lambda x: -x["missed"],
                ),
                "pages": pages,
            }
    return out


def source_evidence() -> dict[str, Any]:
    ev: dict[str, Any] = {}
    with fitz.open(NEW_PDF) as doc:
        ev["new_pdf"] = {
            "path": NEW_PDF,
            "sha256": hashlib.sha256(Path(NEW_PDF).read_bytes()).hexdigest(),
            "metadata": {k: v for k, v in doc.metadata.items() if v},
            "pages": doc.page_count,
            "toc": [{"page": p, "title_len": len(t), "title": t} for _, t, p in doc.get_toc()],
            "front_matter": ("pp.1-3 image-only (no text layer); p.4 character list; "
                             "no copyright text anywhere"),
            "text_layer_pages": sum(1 for i in range(doc.page_count) if doc[i].get_text().strip()),
        }
    with fitz.open(OLD_PDF) as doc:
        ev["old_pdf"] = {
            "path": OLD_PDF,
            "sha256": hashlib.sha256(Path(OLD_PDF).read_bytes()).hexdigest(),
            "pages": doc.page_count,
            "metadata": {k: v for k, v in doc.metadata.items() if v},
            "text_layer_pages": sum(1 for i in range(doc.page_count) if doc[i].get_text().strip()),
            "edition_note": ("People's Lit. Su Nong translation, "
                             "2018 23rd printing (per config annotation)"),
        }
    return ev


def write_manifests(new_stats: dict[int, dict[str, Any]]) -> None:
    """Per-chapter source manifests for the new-source Ch.7-17 standard-path
    outputs. Records source version + page range so downstream consumers can
    tell these apart from any future old-source regeneration. Ch.1-6 get no
    manifest: their standard-path outputs are still the old source's."""
    import datetime

    mdir = AUDIT / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    pdf_sha = hashlib.sha256(Path(NEW_PDF).read_bytes()).hexdigest()
    for ch in range(7, 18):
        seg_path = Path(f"data/segmented/hp1_zh_ch{ch:02d}.jsonl")
        seg_ids = [json.loads(ln)["id"] for ln in seg_path.read_text().splitlines() if ln.strip()]
        manifest = {
            "book": "hp1",
            "lang": "zh",
            "chapter": ch,
            "zh_source": ZH_SOURCE_VERSION,
            "source_pdf": {
                "path": NEW_PDF,
                "sha256": pdf_sha,
                "pages": 232,
                "producer": "calibre 3.39.1",
                "creation_date": "2019-02-16",
                "edition_note": ("Su Nong translation, ebook-to-PDF conversion; "
                                 "no printed-page correspondence"),
            },
            "page_range": {"start": NEW_RANGES[ch][0], "end": NEW_RANGES[ch][1]},
            "config": f"config/hp1_zh_ch{ch:02d}.yaml",
            "outputs": {
                "ocr_raw": f"data/ocr_raw/hp1_zh_ch{ch:02d}.jsonl",
                "text_clean": f"data/text_clean/hp1_zh_ch{ch:02d}.jsonl",
                "segmented": str(seg_path),
            },
            "counts": {
                "paragraphs": new_stats[ch]["paragraphs"],
                "segments": new_stats[ch]["segments"],
                "body_chars": new_stats[ch]["body_chars"],
                "notes": new_stats[ch]["notes"],
            },
            "id_digest_sha256": hashlib.sha256("\n".join(seg_ids).encode()).hexdigest(),
            "status_note": ("new-source Ch.7-17 output; NOT part of a final trilingual "
                            "master — source decision pending (see SOURCE_DECISION_MEMO.md)"),
            "generated_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        }
        (mdir / f"hp1_zh_ch{ch:02d}.manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2)
        )


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    ev = source_evidence()
    ev["old_pdf"]["copyright_page_ocr"] = {
        "edition": "2000年9月北京第1版",
        "printing": "2018年3月第23次印刷",
        "isbn": "978-7-02-010329-4",
        "cip": "中国版本图书馆CIP数据核字(2014)第054413号",
        "print_run": "1842501—2042500",
        "log": "old_pdf_copyright_ocr.log",
    }
    (AUDIT / "source_evidence.json").write_text(json.dumps(ev, ensure_ascii=False, indent=2))

    new_stats = audit_new_source()
    old_stats = audit_old_source()

    diffs = {}
    for ch in range(1, 7):
        diffs[ch] = chapter_diff(ch)
        new_stats[ch]["absent_regions"] = diffs[ch]["region_count"]
        (AUDIT / f"ch{ch:02d}_diff_report.json").write_text(
            json.dumps(diffs[ch], ensure_ascii=False, indent=2)
        )

    stats = {"new_source": new_stats, "old_source": old_stats}
    (AUDIT / "per_chapter_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))

    deficit = line_deficit()
    (AUDIT / "line_deficit.json").write_text(json.dumps(deficit, ensure_ascii=False, indent=2))

    write_manifests(new_stats)
    print("\nManifests written →", AUDIT / "manifests")

    # stdout: aggregates only
    hdr = (f"{'ch':>3} {'src':>4} {'pages':>5} {'blocks':>6} {'paras':>5} {'segs':>5}"
           f" {'chars':>6} {'notes':>5} {'markers':>7}")
    print(hdr)
    for ch in range(1, 18):
        s = new_stats[ch]
        row = (f"{ch:>3} {'new':>4} {s['pages']:>5} {s['raw_blocks']:>6} {s['paragraphs']:>5}"
               f" {s['segments']:>5} {s['body_chars']:>6} {s['notes']:>5} {s['marker_digits']:>7}")
        print(row)
        if ch in old_stats:
            o = old_stats[ch]
            row = (f"{ch:>3} {'old':>4} {o['pages']:>5} {o['raw_blocks']:>6} {o['paragraphs']:>5}"
                   f" {o['segments']:>5} {o['body_chars']:>6} {o['notes']:>5} {'-':>7}")
            print(row)
    print("\nAbsent-region diff (new vs old cleaned, >=20 chars):")
    for ch in range(1, 7):
        d = diffs[ch]
        print(f"  ch{ch}: old={d['old_clean_chars']} new={d['new_clean_chars']} "
              f"regions={d['region_count']} verdicts={d['verdict_counts']}")
    print("\nOld-scan physical-line deficit (projection profile vs OCR lines):")
    for ch in range(1, 7):
        d = deficit[ch]
        print(f"  ch{ch}: total_missed={d['total_missed']} "
              f"worst_pages={[(w['page'], w['missed']) for w in d['worst_pages']]}")


if __name__ == "__main__":
    main()
