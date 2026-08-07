"""Build a dual-model review TSV from two alignment JSONL files.

Runs e5-base and MiniLM alignments on the same segmented inputs, joins them
by EN segment ID, and writes a TSV sorted for human review:

    primary sort: agree=N first  (models disagree → needs eyeball)
    secondary:    conf_minilm ASC (MiniLM is more discriminative than e5,
                                  which tends to score everything >0.7)

Usage:
    uv run python scripts/build_review_tsv.py \\
        --en data/segmented/hp1_en_ch01.jsonl \\
        --zh data/segmented/hp1_zh_ch01.jsonl \\
        --output data/aligned/hp1_en_zh_ch01_review.tsv

Reads e5 from data/aligned/hp1_en_zh_ch01.jsonl; runs MiniLM in a temp
file and joins. Pass --e5 and --minilm explicitly to skip the MiniLM run
(both pre-computed).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

E5_MODEL = "intfloat/multilingual-e5-base"
MINILM_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_alignments(path: Path) -> dict[str, dict]:
    """Map en_id → alignment record (first alignment that references it)."""
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a = json.loads(line)
            for en_id in a.get("en", []):
                out.setdefault(en_id, a)
    return out


def load_segments(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            out[s["id"]] = s["text"]
    return out


def run_minilm_align(en: Path, zh: Path, output: Path) -> Path:
    """Invoke hp-corpus align with MiniLM model."""
    cmd = [
        "uv",
        "run",
        "hp-corpus",
        "align",
        "--src",
        str(en),
        "--tgt",
        str(zh),
        "--output",
        str(output.parent),
        "--out-name",
        output.name,
        "--model",
        MINILM_MODEL,
    ]
    print(f"running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        sys.exit(result.returncode)
    # Output is hp1_en_zh_ch01.jsonl in --output dir — rename to avoid clash
    default_out = output.parent / "hp1_en_zh_ch01.jsonl"
    if default_out != output:
        default_out.rename(output)
    return output


def fmt_ids(ids: list[str]) -> str:
    return ",".join(ids) if ids else "-"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--en", required=True, type=Path, help="Segmented EN JSONL")
    ap.add_argument("--zh", required=True, type=Path, help="Segmented ZH JSONL")
    ap.add_argument("--output", required=True, type=Path, help="Output TSV path")
    ap.add_argument(
        "--e5",
        type=Path,
        default=None,
        help="Pre-computed e5 alignment JSONL (default: data/aligned/hp1_en_zh_ch01.jsonl)",
    )
    ap.add_argument(
        "--minilm",
        type=Path,
        default=None,
        help="Pre-computed minilm JSONL (default: run align with --model MiniLM)",
    )
    args = ap.parse_args()

    e5_path = args.e5 or (args.output.parent / "hp1_en_zh_ch01.jsonl")
    if not e5_path.exists():
        print(f"e5 alignment not found: {e5_path}", file=sys.stderr)
        return 1

    if args.minilm:
        minilm_path = args.minilm
    else:
        minilm_path = args.output.parent / "hp1_en_zh_ch01_minilm.jsonl"
        run_minilm_align(args.en, args.zh, minilm_path)

    e5 = load_alignments(e5_path)
    minilm = load_alignments(minilm_path)
    en_text = load_segments(args.en)
    zh_text = load_segments(args.zh)

    all_en_ids = sorted(set(e5.keys()) | set(minilm.keys()))
    rows: list[dict] = []
    for en_id in all_en_ids:
        e5_a = e5.get(en_id, {})
        ml_a = minilm.get(en_id, {})
        e5_zh = e5_a.get("zh", [])
        ml_zh = ml_a.get("zh", [])
        agree = "Y" if e5_zh and ml_zh and set(e5_zh) == set(ml_zh) else "N"
        # Pick representative zh_id: prefer e5's, fall back to minilm's
        zh_ids = e5_zh or ml_zh
        # For the en_id column, list all en IDs in the alignment (could be 2+ for 2:1 etc.)
        e5_en = e5_a.get("en", [en_id])
        if en_id not in e5_en:
            e5_en = [en_id]
        rows.append(
            {
                "align_id": e5_a.get("align_id", ml_a.get("align_id", "?")),
                "en_id": fmt_ids(e5_en),
                "zh_id": fmt_ids(zh_ids),
                "type_e5": e5_a.get("type", "-"),
                "conf_e5": f"{e5_a.get('confidence', 0.0):.3f}" if e5_a else "0.000",
                "type_minilm": ml_a.get("type", "-"),
                "conf_minilm": f"{ml_a.get('confidence', 0.0):.3f}" if ml_a else "0.000",
                "agree": agree,
                "en_text": en_text.get(en_id, ""),
                "zh_text": zh_text.get(zh_ids[0], "") if zh_ids else "",
            }
        )

    # Sort: disagree first, then conf_minilm ascending
    rows.sort(key=lambda r: (r["agree"] == "Y", float(r["conf_minilm"])))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "align_id",
        "en_id",
        "zh_id",
        "type_e5",
        "conf_e5",
        "type_minilm",
        "conf_minilm",
        "agree",
        "en_text",
        "zh_text",
    ]
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            # Escape newlines/tabs in text fields
            row = {**r}
            for k in ("en_text", "zh_text"):
                row[k] = row[k].replace("\t", " ").replace("\n", " ").replace("\r", " ")
            f.write("\t".join(row[c] for c in cols) + "\n")

    n_agree = sum(1 for r in rows if r["agree"] == "Y")
    print(
        f"wrote {args.output}: {len(rows)} rows, {n_agree} agree / {len(rows) - n_agree} disagree"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
