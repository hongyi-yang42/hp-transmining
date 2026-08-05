"""Unified OCR: PaddleOCR for image-only scans, PyMuPDF for embedded text layers.

Output is OCRBlock JSONL regardless of engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .render import extract_text_layer_blocks, has_text_layer
from .schema import OCRBlock

_console = Console()


def _make_paddleocr(lang: str):
    """Construct a PaddleOCR instance. Imported lazily so unit tests don't require it."""
    from paddleocr import PaddleOCR  # type: ignore[import-not-found]

    return PaddleOCR(lang=lang, use_angle_cls=True, show_log=False)


def ocr_image(paddle_instance, image_path: str | Path, page: int) -> list[OCRBlock]:
    """Run PaddleOCR on one image; return OCRBlocks with bbox + confidence."""
    result = paddle_instance.ocr(str(image_path), cls=True)
    blocks: list[OCRBlock] = []
    # PaddleOCR returns [[line, [(bbox, (text, conf))], ...]] — version-dependent shape.
    raw = result[0] if result and isinstance(result, list) else []
    for block_idx, entry in enumerate(raw):
        if entry is None:
            continue
        bbox, (text, conf) = entry
        blocks.append(
            OCRBlock(
                page=page,
                block_idx=block_idx,
                text=text,
                bbox=[float(c) for c in bbox[0] if c is not None][:4]
                if isinstance(bbox, list) and bbox and isinstance(bbox[0], list)
                else [float(c) for c in bbox][:4],
                confidence=float(conf),
                engine="paddleocr",
            )
        )
    return blocks


def ocr_pages_paddleocr(
    image_paths: list[Path],
    lang: str,
) -> list[OCRBlock]:
    """Run PaddleOCR over a list of pre-rendered page images."""
    paddle = _make_paddleocr(lang)
    all_blocks: list[OCRBlock] = []
    for img in image_paths:
        page_no = int(img.stem.split("_p")[-1])
        all_blocks.extend(ocr_image(paddle, img, page_no))
    return all_blocks


def ocr_pdf_text_layer(
    pdf_path: str | Path,
    start_page: int,
    end_page: int,
) -> list[OCRBlock]:
    """Extract blocks via PyMuPDF's text layer."""
    return extract_text_layer_blocks(pdf_path, start_page, end_page)


def write_jsonl(blocks: list[OCRBlock], output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for b in blocks:
            f.write(b.model_dump_json() + "\n")


def read_jsonl(input_path: str | Path) -> list[OCRBlock]:
    blocks: list[OCRBlock] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            blocks.append(OCRBlock.model_validate_json(line))
    return blocks


def confidence_summary(blocks: list[OCRBlock]) -> dict[str, float | int]:
    """Aggregate statistics. Returns only numbers — no text content."""
    if not blocks:
        return {"count": 0}
    confs = [b.confidence for b in blocks]
    sorted_c = sorted(confs)
    n = len(sorted_c)
    return {
        "count": n,
        "mean": sum(confs) / n,
        "median": sorted_c[n // 2],
        "p10": sorted_c[max(0, int(n * 0.10) - 1)],
        "below_0p6": sum(1 for c in confs if c < 0.6),
    }


def print_confidence_table(blocks: list[OCRBlock], label: str) -> None:
    s = confidence_summary(blocks)
    if s["count"] == 0:
        _console.print(f"[yellow]{label}: no blocks[/]")
        return
    table = Table(title=f"OCR confidence — {label}", show_header=True)
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in s.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
        else:
            table.add_row(k, str(v))
    _console.print(table)


def run_ocr(config: dict[str, Any], pages_dir: str | Path) -> list[OCRBlock]:
    """Dispatch OCR based on config. Uses pre-rendered PNGs for paddleocr."""
    engine = config["ocr"]["engine"]
    lang = config["ocr"]["lang"]
    start = config["chapter"]["start_page"]
    end = config["chapter"]["end_page"]
    pdf_path = config["pdf_path"]

    if engine == "pymupdf":
        return ocr_pdf_text_layer(pdf_path, start, end)

    if engine == "paddleocr":
        return ocr_pages_paddleocr(
            _select_page_images(pages_dir, config["book"], lang, start, end),
            lang,
        )

    if engine == "auto":
        import fitz

        with fitz.open(pdf_path) as doc:
            return (
                ocr_pdf_text_layer(pdf_path, start, end)
                if has_text_layer(doc)
                else ocr_pages_paddleocr(
                    _select_page_images(pages_dir, config["book"], lang, start, end),
                    lang,
                )
            )

    raise ValueError(f"Unknown ocr.engine: {engine}")


def _select_page_images(
    pages_dir: str | Path,
    book: str,
    lang: str,
    start: int,
    end: int,
) -> list[Path]:
    pages = Path(pages_dir)
    wanted = set(range(start, end + 1))
    images = sorted(pages.glob(f"{book}_{lang}_p*.png"))
    selected = [p for p in images if int(p.stem.split("_p")[-1]) in wanted]
    if not selected:
        raise FileNotFoundError(
            f"No rendered PNGs found in {pages} for pages {start}-{end}. "
            "Run `hp-corpus render` first."
        )
    return selected


__all__ = [
    "OCRBlock",
    "ocr_pdf_text_layer",
    "ocr_pages_paddleocr",
    "write_jsonl",
    "read_jsonl",
    "confidence_summary",
    "print_confidence_table",
    "run_ocr",
]
