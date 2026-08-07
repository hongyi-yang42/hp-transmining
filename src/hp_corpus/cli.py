"""hp-corpus command-line interface.

Subcommands:
  validate  — manifest + SHA-256 + page-count + text-layer checks
  render    — render PDF page range to PNG (for OCR)
  ocr       — extract text (PaddleOCR or PyMuPDF text layer) → OCRBlock JSONL
  clean     — strip headers/page numbers/footnotes → CleanSentence JSONL + .txt
  segment   — sentence segmentation with stable IDs → Segment JSONL
  align     — EN–ZH alignment via LaBSE + DP
  run       — render → ocr → clean → segment for one language config

All commands print metadata + counts only; no novel text reaches the terminal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from . import __version__
from .align import (
    AlignmentConfig,
    align_segments,
    alignment_summary,
    load_segments,
    write_alignments_jsonl,
)
from .clean import clean_blocks, write_clean_outputs
from .ocr import print_confidence_table, read_jsonl, run_ocr, write_jsonl
from .parse import parse_segments
from .render import load_config, render_pdf, validate_source
from .schema import CleanSentence
from .segment import segment_all, write_segments_jsonl

_console = Console()


def _data_root() -> Path:
    return Path("data")


def _pages_dir(book: str, lang: str) -> Path:
    return _data_root() / "pages" / f"{book}_{lang}"


def _ocr_raw_path(book: str, lang: str, chapter: int) -> Path:
    return _data_root() / "ocr_raw" / f"{book}_{lang}_ch{chapter:02d}.jsonl"


def _text_clean_dir() -> Path:
    return _data_root() / "text_clean"


def _segmented_path(book: str, lang: str, chapter: int) -> Path:
    return _data_root() / "segmented" / f"{book}_{lang}_ch{chapter:02d}.jsonl"


def _parsed_path(book: str, lang: str, chapter: int) -> Path:
    return _data_root() / "parsed" / f"{book}_{lang}_ch{chapter:02d}.conllu"


def _aligned_dir() -> Path:
    return _data_root() / "aligned"


def _embeddings_dir() -> Path:
    return _data_root() / "embeddings"


def _print_validation(report: Any) -> None:
    table = Table(title=f"Validation — {report.document_id}", show_header=True)
    table.add_column("field")
    table.add_column("value")
    table.add_row("path", report.path)
    table.add_row("size_bytes", str(report.size_bytes))
    table.add_row("expected_sha256", report.expected_sha256[:16] + "…")
    table.add_row("actual_sha256_prefix", report.actual_sha256_prefix + "…")
    table.add_row("sha256_ok", str(report.sha256_ok))
    table.add_row("expected_total_pages", str(report.expected_total_pages))
    table.add_row("actual_total_pages", str(report.actual_total_pages))
    table.add_row("page_count_ok", str(report.page_count_ok))
    table.add_row(
        "has_text_layer", f"{report.expected_has_text_layer} → {report.actual_has_text_layer}"
    )
    _console.print(table)


def cmd_validate(args: argparse.Namespace) -> int:
    configs = []
    if args.config:
        configs = [load_config(args.config)]
    else:
        for p in sorted(Path("config").glob("hp1_*.yaml")):
            configs.append(load_config(p))

    rc = 0
    for cfg in configs:
        try:
            report = validate_source(cfg)
            _print_validation(report)
        except Exception as e:  # noqa: BLE001
            _console.print(f"[red]FAIL[/] {cfg.get('book')}_{cfg.get('lang')}: {e}")
            rc = 1
    return rc


def cmd_render(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    pages_dir = _pages_dir(cfg["book"], cfg["lang"])
    count, has_layer = render_pdf(
        cfg["pdf_path"],
        pages_dir,
        cfg["book"],
        cfg["lang"],
        dpi=args.dpi or cfg.get("scan", {}).get("dpi", 300),
        start_page=cfg["chapter"]["start_page"],
        end_page=cfg["chapter"]["end_page"],
    )
    _console.print(f"rendered [cyan]{count}[/] pages → {pages_dir} (has_text_layer={has_layer})")
    return 0


def cmd_ocr(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    pages_dir = _pages_dir(cfg["book"], cfg["lang"])
    blocks = run_ocr(cfg, pages_dir)
    out = _ocr_raw_path(cfg["book"], cfg["lang"], cfg["chapter"]["number"])
    write_jsonl(blocks, out)
    _console.print(f"ocr → {out}  ([cyan]{len(blocks)}[/] blocks)")
    print_confidence_table(blocks, f"{cfg['book']}_{cfg['lang']}")
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    raw = _ocr_raw_path(cfg["book"], cfg["lang"], cfg["chapter"]["number"])
    blocks = read_jsonl(raw)
    result = clean_blocks(blocks, cfg)
    paths = write_clean_outputs(
        result,
        _text_clean_dir(),
        cfg["book"],
        cfg["lang"],
        cfg["chapter"]["number"],
    )
    _console.print(
        f"clean → {paths['jsonl']}  "
        f"([cyan]{len(result.sentences)}[/] paragraphs, "
        f"[yellow]{len(result.footnotes)}[/] footnotes)"
    )
    if result.chapter_title:
        _console.print("  detected chapter title: [dim](suppressed)[/]")
    return 0


def cmd_segment(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    clean_jsonl = (
        _text_clean_dir() / f"{cfg['book']}_{cfg['lang']}_ch{cfg['chapter']['number']:02d}.jsonl"
    )
    with open(clean_jsonl, encoding="utf-8") as f:
        cleans = [CleanSentence.model_validate_json(line) for line in f if line.strip()]
    segments = segment_all(
        cleans,
        cfg["book"],
        cfg["lang"],
        cfg["chapter"]["number"],
        cfg,
    )
    out = _segmented_path(cfg["book"], cfg["lang"], cfg["chapter"]["number"])
    write_segments_jsonl(segments, out)
    _console.print(
        f"segment → {out}  ([cyan]{len(segments)}[/] sentences "
        f"from [cyan]{len(cleans)}[/] paragraphs)"
    )
    return 0


def cmd_parse(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    seg = _segmented_path(cfg["book"], cfg["lang"], cfg["chapter"]["number"])
    out = _parsed_path(cfg["book"], cfg["lang"], cfg["chapter"]["number"])
    result = parse_segments(seg, cfg["lang"], out)
    _console.print(
        f"parse → {out}  ([cyan]{result['output_sentences']}[/] sentences, "
        f"[cyan]{result['output_tokens']}[/] tokens, "
        f"[yellow]{result['n_mwt']}[/] MWT expansions from "
        f"[cyan]{result['input_segments']}[/] input segments)"
    )
    return 0


def _lang_from_path(path: str) -> str:
    """Extract the language code from a segmented-file path like
    'data/segmented/hp1_en_ch01.jsonl' → 'en'."""
    import re

    m = re.search(r"_([a-z]{2,3})_ch\d+", path)
    return m.group(1) if m else "x"


def cmd_align(args: argparse.Namespace) -> int:
    src = load_segments(args.src)
    tgt = load_segments(args.tgt)
    config = AlignmentConfig(
        embed_cache_dir=_embeddings_dir(),
        vecalign_dir=Path("vendor/vecalign") if Path("vendor/vecalign").exists() else None,
        model_name=args.model,
        locality_band=args.band,
    )
    alignments = align_segments(src, tgt, config)
    # Output name derived from input languages: hp1_<src>_<tgt>_ch01.jsonl
    src_lang = _lang_from_path(args.src)
    tgt_lang = _lang_from_path(args.tgt)
    out_name = args.out_name or f"hp1_{src_lang}_{tgt_lang}_ch01.jsonl"
    out = Path(args.output) / out_name
    write_alignments_jsonl(alignments, out)
    summary = alignment_summary(alignments)
    _console.print(f"align → {out}  ([cyan]{summary['count']}[/] records)")
    table = Table(title="Alignment summary", show_header=True)
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in summary.items():
        if isinstance(v, dict):
            table.add_row(k, ", ".join(f"{kk}={vv}" for kk, vv in v.items()))
        elif isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
        else:
            table.add_row(k, str(v))
    _console.print(table)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run render → ocr → clean → segment for one language config."""
    cfg = load_config(args.config)
    # Synthesize a Namespace with the fields each subcommand expects.
    shared = argparse.Namespace(config=args.config, dpi=None)
    for step in (cmd_render, cmd_ocr, cmd_clean, cmd_segment):
        rc = step(shared)
        if rc != 0:
            return rc
    _console.print(f"[green]run complete[/] for {cfg['book']}_{cfg['lang']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hp-corpus", description=__doc__)
    parser.add_argument("--version", action="version", version=f"hp-corpus {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_validate = sub.add_parser("validate", help="Validate source PDFs against manifests")
    p_validate.add_argument(
        "--config", help="Path to a single config; default validates all in config/"
    )
    p_validate.set_defaults(func=cmd_validate)

    p_render = sub.add_parser("render", help="Render PDF pages to PNG")
    p_render.add_argument("--config", required=True)
    p_render.add_argument("--dpi", type=int, default=None)
    p_render.set_defaults(func=cmd_render)

    p_ocr = sub.add_parser("ocr", help="Run OCR (or text-layer extraction)")
    p_ocr.add_argument("--config", required=True)
    p_ocr.set_defaults(func=cmd_ocr)

    p_clean = sub.add_parser("clean", help="Clean OCR JSONL")
    p_clean.add_argument("--config", required=True)
    p_clean.set_defaults(func=cmd_clean)

    p_segment = sub.add_parser("segment", help="Sentence segmentation")
    p_segment.add_argument("--config", required=True)
    p_segment.set_defaults(func=cmd_segment)

    p_parse = sub.add_parser("parse", help="UD parsing → CoNLL-U (requires Stanza models)")
    p_parse.add_argument("--config", required=True)
    p_parse.set_defaults(func=cmd_parse)

    p_align = sub.add_parser("align", help="Sentence-level alignment between two languages")
    p_align.add_argument("--src", required=True, help="Path to source-language segmented JSONL")
    p_align.add_argument("--tgt", required=True, help="Path to target-language segmented JSONL")
    p_align.add_argument("--output", required=True, help="Output directory")
    p_align.add_argument(
        "--out-name",
        default=None,
        help="Output filename (default: hp1_<src>_<tgt>_ch01.jsonl, derived from inputs)",
    )
    p_align.add_argument(
        "--model",
        default="intfloat/multilingual-e5-base",
        help="sentence-transformers model name (default: e5-base ~1.1GB; "
        "fallback: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 ~470MB)",
    )
    p_align.add_argument(
        "--band",
        type=float,
        default=0.15,
        help="Diagonal-band width for global DP (fraction of length). 0.15 = ±15%% drift.",
    )
    p_align.set_defaults(func=cmd_align)

    p_run = sub.add_parser("run", help="Render → OCR → clean → segment for one config")
    p_run.add_argument("--config", required=True)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
